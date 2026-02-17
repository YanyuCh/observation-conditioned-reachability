import isaacgym
assert isaacgym
import torch
import numpy as np
import pickle as pkl
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

def load_policy(logdir):
    # Load the body and adaptation module
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        # Forward pass: History -> Latent -> Action
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(label, friction_value, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [friction_value, friction_value] # Force specific friction
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    #from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy

def run_stress_test(label, friction_values=[0.25, 0.5, 1.0], v_cmds=[0.5, 1.0, 1.5, 2.0], w_cmds=[-2.0, 0.0, 2.0]):
    """
    Systematically tests the policy across a grid of frictions and commands.
    """

    results = {}

    print(f"Starting Stress Test on: {label}")
    print(f"Frictions: {friction_values}")
    print(f"Velocities: {v_cmds}")
    print(f"Yaws: {w_cmds}")
    print("-" * 50)

    # 4. The Test Loop
    # ----------------
    for mu in friction_values:
        print(f"\n>>> Testing Friction mu = {mu}")
        
        # load the env
        env, policy = load_env(label, friction_value=mu, headless=True)
        
        for v_target in v_cmds:
            for w_target in w_cmds:
                
                # Reset Env
                obs = env.reset()
                
                # Set Command
                # [lin_vel_x, lin_vel_y, ang_vel_yaw]
                env.commands[:, 0] = v_target
                env.commands[:, 1] = 0.0       # No lateral velocity command for now
                env.commands[:, 2] = w_target
                
                # Run Episode (e.g. 5 seconds = 250 steps at 0.02s dt)
                # Note: 'walk-these-ways' often uses dt=0.02 for control
                num_steps = 250 
                survival_steps = 0
                measured_vels = [] # Store velocities
                
                for i in range(num_steps):
                    with torch.no_grad():
                        actions = policy(obs)
                    
                    obs, rew, done, info = env.step(actions)
                    
                    # Capture actual forward velocity (base_lin_vel is usually in body frame)
                    # We access the raw env to get the ground truth state
                    actual_v = env.base_lin_vel[0, 0].item()
                    measured_vels.append(actual_v)
                    
                    # Log Stability Metrics
                    # projected_gravity is usually [sin(pitch), -sin(roll)cos(pitch), -cos(roll)cos(pitch)]
                    # or simply check base_quat. 
                    # Here we approximate from projected_gravity if available, or just check 'done'
                    
                    # If the environment terminates (falls), stop
                    if done.any():
                         break
                    
                    survival_steps += 1
                    avg_vel = np.mean(measured_vels)
                    tracking_error = abs(v_target - avg_vel)
                
                # Determine Result
                survived = (survival_steps == num_steps)
                key = (mu, v_target, w_target)
                results[key] = {
                    "survived": survived,
                    "steps": survival_steps,
                    "avg_vel": avg_vel
                }
                
                status = "PASS" if survived else "FAIL"
                print(f"  CMD [v={v_target:.1f}, w={w_target:.1f}] -> {status} (Steps: {survival_steps}/{num_steps})"
                      f"| Actual v={avg_vel:.2f} (Err: {tracking_error:.2f})")

    return results

def plot_results(results, filename="stability_matrix.png"):
    # Visualize the Safe/Unsafe regions
    frictions = sorted(list(set([k[0] for k in results.keys()])))
    
    fig, axes = plt.subplots(1, len(frictions), figsize=(5*len(frictions), 4), sharey=True)
    if len(frictions) == 1: axes = [axes]
    
    for i, mu in enumerate(frictions):
        ax = axes[i]
        
        # Extract data for this friction
        data = [k for k in results.keys() if k[0] == mu]
        vs = [k[1] for k in data]
        ws = [k[2] for k in data]
        colors = ['green' if results[k]['survived'] else 'red' for k in data]
        
        ax.scatter(ws, vs, c=colors, s=100)
        ax.set_title(f"Friction $\mu$ = {mu}")
        ax.set_xlabel("Yaw Cmd $\omega$ (rad/s)")
        if i == 0: ax.set_ylabel("Linear Cmd $v$ (m/s)")
        ax.grid(True)
    
    plt.suptitle("Stability Region: Green=Stable, Red=Fell Over")
    plt.tight_layout()
    # SAVE instead of SHOW
    print(f"Saving plot to {filename}...")
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # Replace with your actual label
    label = "gait-conditioned-agility/pretrain-v0/train" 
    
    results = run_stress_test(
        label, 
        friction_values=[0.8], # Focus on low friction
        v_cmds=[0.5, 1.0, 1.5, 2.0],          # Your velocity range
        w_cmds=[-2.0, -1.0, 0.0, 1.0, 2.0]    # Your yaw range
    )
    
    plot_results(results)