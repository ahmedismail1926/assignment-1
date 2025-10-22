"""
Visualization and Simulation Functions
======================================
Functions for visualizing policies, running simulations, and testing robustness.
"""

import numpy as np
from collections import Counter


def print_policy_visualization(env, policy):
    """
    Print ASCII visualization of the policy
    
    Shows the optimal action for each cell using arrow symbols.
    """
    n = env.n
    action_symbols = {0: '→', 1: '↑', 2: '←', 3: '↓'}
    
    print("\n" + "=" * 50)
    print("OPTIMAL POLICY VISUALIZATION")
    print("=" * 50)
    
    for i in range(n):
        row = ""
        for j in range(n):
            if (i, j) == env.goal_pos:
                row += " G "
            elif (i, j) in env.bad_cells:
                row += " X "
            else:
                best_action = np.argmax(policy[i, j, :])
                row += f" {action_symbols[best_action]} "
        print(row)
    
    print("=" * 50)


def print_value_function(env, V):
    """
    Print the value function in a grid format
    
    Args:
        env: Environment instance
        V: Value function (n x n array)
    """
    print("\nValue Function (rounded to 1 decimal):")
    print("=" * 50)
    
    for i in range(env.n):
        row = ""
        for j in range(env.n):
            if (i, j) == env.goal_pos:
                row += "   G   "
            elif (i, j) in env.bad_cells:
                row += "   X   "
            else:
                row += f"{V[i, j]:6.1f} "
        print(row)
    
    print("=" * 50)


def simulate_policy(env, policy, max_steps=100):
    """
    Simulate the learned policy and record the path
    
    Args:
        env: Environment instance
        policy: Learned policy
        max_steps: Maximum steps before termination
    
    Returns:
        step_count: Number of steps taken
        total_reward: Total accumulated reward
        path: List of visited states
    """
    print("\n🎬 Running simulation with learned policy...")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0
    
    path = [env.agent_pos]
    actions_taken = []
    rewards_received = []
    
    while not (terminated or truncated) and step_count < max_steps:
        # Get current position from observation
        x, y = obs[0], obs[1]
        
        # Choose action based on policy (greedy)
        action = np.argmax(policy[x, y, :])
        actions_taken.append(action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards_received.append(reward)
        step_count += 1
        
        path.append(env.agent_pos)
        
        # Render
        if env.render_mode:
            env.render()
    
    action_symbols = {0: '→', 1: '↑', 2: '←', 3: '↓'}
    
    print(f"✓ Episode finished in {step_count} steps")
    print(f"✓ Total reward: {total_reward:.2f}")
    print(f"✓ Reached goal: {'YES ✓' if terminated else 'NO ✗ (hit max steps)'}")
    print(f"✓ Path taken ({len(path)} states): {' → '.join([str(p) for p in path[:10]])}" + 
          (" ..." if len(path) > 10 else ""))
    
    # Diagnostic: Check if agent got stuck
    if not terminated and step_count >= max_steps:
        print(f"\n⚠️  CONVERGENCE FAILURE DETECTED!")
        print(f"   Final position: {env.agent_pos}")
        print(f"   Goal position: {env.goal_pos}")
        print(f"   Distance from goal: {abs(env.agent_pos[0] - env.goal_pos[0]) + abs(env.agent_pos[1] - env.goal_pos[1])} cells")
        print(f"   Times hit bad cells: {sum(1 for r in rewards_received if r == -50.0)}")
        print(f"   Bad cell penalty: {sum(r for r in rewards_received if r == -50.0)}")
        
        # Check for loops (visiting same state multiple times)
        unique_states = len(set(path))
        if unique_states < len(path) * 0.5:
            print(f"   ⚠️  Agent appears stuck in a loop! (visited {unique_states} unique states out of {len(path)})")
            
            # Find most visited state
            state_counts = Counter(path)
            most_common = state_counts.most_common(3)
            print(f"   Most visited states:")
            for state, count in most_common:
                print(f"      {state}: {count} times (action: {action_symbols[np.argmax(policy[state[0], state[1], :])]})")
    
    return step_count, total_reward, path


def test_policy_robustness(env, policy, num_trials=20, max_steps=100):
    """
    Test the learned policy over multiple trials to measure robustness
    
    Args:
        env: Environment instance
        policy: Learned policy
        num_trials: Number of test episodes
        max_steps: Maximum steps per episode
    
    Returns:
        success_rate: Percentage of successful episodes
        successes: Number of successful episodes
        failures: Number of failed episodes
    """
    print("\n" + "=" * 50)
    print(f"🧪 TESTING POLICY ROBUSTNESS ({num_trials} trials)")
    print("=" * 50)
    
    successes = 0
    failures = 0
    total_steps_list = []
    total_rewards_list = []
    bad_cell_hits_list = []
    
    for trial in range(num_trials):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        bad_cell_hits = 0
        
        while not (terminated or truncated) and step_count < max_steps:
            x, y = obs[0], obs[1]
            action = np.argmax(policy[x, y, :])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if reward == -50.0:
                bad_cell_hits += 1
        
        if terminated:
            successes += 1
            total_steps_list.append(step_count)
            total_rewards_list.append(total_reward)
        else:
            failures += 1
        
        bad_cell_hits_list.append(bad_cell_hits)
    
    success_rate = (successes / num_trials) * 100
    
    print(f"\n📊 RESULTS:")
    print(f"   Success rate: {success_rate:.1f}% ({successes}/{num_trials})")
    print(f"   Failure rate: {100-success_rate:.1f}% ({failures}/{num_trials})")
    
    if successes > 0:
        print(f"\n   Successful episodes:")
        print(f"      Average steps: {np.mean(total_steps_list):.1f}")
        print(f"      Average reward: {np.mean(total_rewards_list):.1f}")
        print(f"      Min steps: {min(total_steps_list)}")
        print(f"      Max steps: {max(total_steps_list)}")
    
    avg_bad_hits = np.mean(bad_cell_hits_list)
    print(f"\n   Average bad cell hits per episode: {avg_bad_hits:.2f}")
    
    if success_rate < 80:
        print(f"\n⚠️  WARNING: Low success rate indicates:")
        print(f"      - Bad cells may be blocking optimal paths")
        print(f"      - Stochastic transitions causing divergence")
        print(f"      - Goal may be hard to reach from certain positions")
    
    print("=" * 50)
    
    return success_rate, successes, failures
