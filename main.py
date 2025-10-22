"""
Main Execution Script
====================
Runs the complete policy iteration pipeline:
1. Create environment
2. Run policy iteration
3. Visualize results
4. Test policy robustness
5. Record video
"""

import os
from datetime import datetime
from gymnasium.wrappers import RecordVideo
import numpy as np

from environment import GridMazeEnv
from policy_iteration import policy_iteration
from visualization import (
    print_policy_visualization, 
    print_value_function,
    simulate_policy, 
    test_policy_robustness
)


def main():
    """Main execution function"""
    print("=" * 60)
    print("GRID MAZE - POLICY ITERATION WITH STOCHASTIC TRANSITIONS")
    print("=" * 60)
    
    # Create environment with fixed seed for reproducibility
    seed = None
    env = GridMazeEnv(n=5, seed_nr=seed, render_mode=None)
    
    print(f"\n📍 Environment Setup (seed={seed}):")
    print(f"   Start position: {env.start_pos}")
    print(f"   Goal position: {env.goal_pos}")
    print(f"   Bad cells: {env.bad_cells}")
    
    # Run Policy Iteration
    policy, V, iterations = policy_iteration(env, gamma=0.95, theta=1e-6)
    
    # Visualize policy
    print_policy_visualization(env, policy)
    
    # Print value function
    print_value_function(env, V)
    
    # Test policy robustness
    success_rate, successes, failures = test_policy_robustness(
        env, policy, num_trials=50, max_steps=100
    )
    
    # Run single simulation with visualization
    print("\n" + "=" * 60)
    print("SINGLE EPISODE SIMULATION (with rendering)")
    print("=" * 60)
    
    play_env = GridMazeEnv(n=5, seed_nr=seed, render_mode='human')
    play_env.start_pos = env.start_pos
    play_env.goal_pos = env.goal_pos
    play_env.bad_cells = env.bad_cells
    play_env.agent_pos = play_env.start_pos
    
    steps, reward, path = simulate_policy(play_env, policy, max_steps=100)
    
    play_env.close()
    
    # Record video
    print("\n" + "=" * 60)
    print("🎥 RECORDING VIDEO")
    print("=" * 60)
    
    os.makedirs('./videos', exist_ok=True)
    
    # Create unique video name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f'grid_maze_{timestamp}'
    
    # Create environment for video recording
    video_base_env = GridMazeEnv(n=5, seed_nr=seed, render_mode='rgb_array')
    video_base_env.start_pos = env.start_pos
    video_base_env.goal_pos = env.goal_pos
    video_base_env.bad_cells = env.bad_cells
    
    # Wrap with RecordVideo
    video_env = RecordVideo(
        video_base_env,
        video_folder='./videos',
        episode_trigger=lambda x: True,
        name_prefix=video_name
    )
    
    # Run episode with video recording
    obs, info = video_env.reset()
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 100:
        x, y = obs[0], obs[1]
        action = np.argmax(policy[x, y, :])
        obs, reward, terminated, truncated, info = video_env.step(action)
        step_count += 1
    
    # Close and save video
    video_env.close()
    del video_env  # Ensure wrapper is properly deleted to finalize video
    
    print(f"✓ Video saved in ./videos/ directory")
    print(f"✓ Episode in video: {step_count} steps, Goal reached: {'YES ✓' if terminated else 'NO ✗'}")
    print("\n" + "=" * 60)
    print("✅ SIMULATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
