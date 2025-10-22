"""
Policy Iteration Algorithm
==========================
Implements policy iteration with stochastic transitions for MDPs.
Includes both optimized (NumPy vectorized) and legacy (commented) implementations.
"""

import numpy as np


def policy_iteration(env, gamma=0.95, theta=1e-6):
    """
    Main policy iteration algorithm
    
    Args:
        env: GridMazeEnv instance
        gamma: Discount factor
        theta: Convergence threshold
    
    Returns:
        policy: Optimal policy (n x n x 4)
        value_function: Optimal value function (n x n)
        iterations: Number of iterations to converge
    """
    n = env.n
    # Initialization
    V = np.zeros((n, n))
    policy = np.ones((n, n, 4)) / 4  # Uniform random policy
    iteration_count = 0
    policy_stable = False
    
    print("Starting Policy Iteration...")
    print("=" * 50)
    
    while not policy_stable:
        iteration_count += 1
        
        # Policy Evaluation
        V = policy_evaluation(env, policy, V, gamma, theta)
        
        # Policy Improvement
        policy_stable = policy_improvement(env, policy, V, gamma)
        
        print(f"Iteration {iteration_count}: Policy {'stable ✓' if policy_stable else 'updated'}")
    
    print("=" * 50)
    print(f"✓ Policy converged in {iteration_count} iterations!")
    
    return policy, V, iteration_count


def policy_evaluation(env, policy, V, gamma, theta):
    """
    Evaluate the current policy using stochastic transitions - OPTIMIZED with NumPy
    
    This version uses vectorized operations for better performance.
    Pre-computes all transitions and rewards, then applies them in batch.
    """
    n = env.n
    
    # Pre-compute terminal state mask
    terminal_mask = np.zeros((n, n), dtype=bool)
    terminal_mask[env.goal_pos] = True
    for bad_cell in env.bad_cells:
        terminal_mask[bad_cell] = True
    
    # Pre-compute transition matrices and rewards for all states and actions
    # Shape: (n, n, 4, 3) - for each state and action, store 3 possible next states
    next_states = np.zeros((n, n, 4, 3, 2), dtype=np.int32)  # Last dim: [x, y]
    rewards = np.zeros((n, n, 4, 3))  # Rewards for each transition
    
    for x in range(n):
        for y in range(n):
            for action in range(4):
                # 70% - intended direction
                next_x, next_y = get_next_state(x, y, action, n)
                next_states[x, y, action, 0] = [next_x, next_y]
                rewards[x, y, action, 0] = get_reward_for_state(next_x, next_y, env)
                
                # 15% - perpendicular left
                left_action = (action - 1) % 4
                next_x, next_y = get_next_state(x, y, left_action, n)
                next_states[x, y, action, 1] = [next_x, next_y]
                rewards[x, y, action, 1] = get_reward_for_state(next_x, next_y, env)
                
                # 15% - perpendicular right
                right_action = (action + 1) % 4
                next_x, next_y = get_next_state(x, y, right_action, n)
                next_states[x, y, action, 2] = [next_x, next_y]
                rewards[x, y, action, 2] = get_reward_for_state(next_x, next_y, env)
    
    # Transition probabilities
    probs = np.array([0.70, 0.15, 0.15])
    
    delta = float('inf')
    
    while delta > theta:
        V_new = V.copy()
        
        # Vectorized Bellman update for all non-terminal states
        for x in range(n):
            for y in range(n):
                if terminal_mask[x, y]:
                    V_new[x, y] = 0
                    continue
                
                # Calculate expected value for all actions at once
                expected_values = np.zeros(4)
                for action in range(4):
                    # Get next state values for all 3 possible transitions
                    next_vals = np.array([
                        V[next_states[x, y, action, 0, 0], next_states[x, y, action, 0, 1]],
                        V[next_states[x, y, action, 1, 0], next_states[x, y, action, 1, 1]],
                        V[next_states[x, y, action, 2, 0], next_states[x, y, action, 2, 1]]
                    ])
                    
                    # Expected value = sum of prob * (reward + gamma * next_value)
                    expected_values[action] = np.sum(probs * (rewards[x, y, action] + gamma * next_vals))
                
                # Value = sum over actions weighted by policy
                V_new[x, y] = np.sum(policy[x, y] * expected_values)
        
        # Compute max change
        delta = np.max(np.abs(V_new[~terminal_mask] - V[~terminal_mask]))
        V = V_new
    
    return V


# =====================
# OLD IMPLEMENTATION (Loop-based) - KEPT FOR REFERENCE
# =====================
# def policy_evaluation_old(env, policy, V, gamma, theta):
#     """
#     Evaluate the current policy using stochastic transitions
#     
#     This is the original loop-based implementation.
#     Commented out in favor of optimized NumPy version above.
#     """
#     n = env.n
#     delta = float('inf')
#     
#     while delta > theta:
#         delta = 0
#         V_new = V.copy()
#         
#         for x in range(n):
#             for y in range(n):
#                 # Skip terminal states (goal and bad cells)
#                 if (x, y) == env.goal_pos or (x, y) in env.bad_cells:
#                     V_new[x, y] = 0
#                     continue
#                 
#                 # Calculate value using Bellman equation
#                 v = 0
#                 for action in range(4):
#                     action_prob = policy[x, y, action]
#                     
#                     # Calculate expected value considering stochastic transitions
#                     expected_value = 0
#                     
#                     # 70% - intended direction
#                     next_x, next_y = get_next_state(x, y, action, n)
#                     reward = get_reward_for_state(next_x, next_y, env)
#                     expected_value += 0.70 * (reward + gamma * V[next_x, next_y])
#                     
#                     # 15% - perpendicular left
#                     left_action = (action - 1) % 4
#                     next_x, next_y = get_next_state(x, y, left_action, n)
#                     reward = get_reward_for_state(next_x, next_y, env)
#                     expected_value += 0.15 * (reward + gamma * V[next_x, next_y])
#                     
#                     # 15% - perpendicular right
#                     right_action = (action + 1) % 4
#                     next_x, next_y = get_next_state(x, y, right_action, n)
#                     reward = get_reward_for_state(next_x, next_y, env)
#                     expected_value += 0.15 * (reward + gamma * V[next_x, next_y])
#                     
#                     v += action_prob * expected_value
#                 
#                 delta = max(delta, abs(V_new[x, y] - v))
#                 V_new[x, y] = v
#         
#         V = V_new
#     
#     return V


def policy_improvement(env, policy, V, gamma):
    """
    Improve policy by acting greedily with respect to value function - OPTIMIZED with NumPy
    
    This version pre-computes all Q-values and uses vectorized operations
    for faster policy updates.
    """
    n = env.n
    
    # Pre-compute terminal state mask
    terminal_mask = np.zeros((n, n), dtype=bool)
    terminal_mask[env.goal_pos] = True
    for bad_cell in env.bad_cells:
        terminal_mask[bad_cell] = True
    
    # Store old policy for comparison
    old_policy = policy.copy()
    
    # Pre-compute all Q-values for all states and actions
    q_values = np.zeros((n, n, 4))
    probs = np.array([0.70, 0.15, 0.15])
    
    for x in range(n):
        for y in range(n):
            if terminal_mask[x, y]:
                continue
            
            for action in range(4):
                # Get next states and rewards for all 3 possible transitions
                next_states = []
                next_rewards = []
                
                # 70% - intended direction
                next_x, next_y = get_next_state(x, y, action, n)
                next_states.append((next_x, next_y))
                next_rewards.append(get_reward_for_state(next_x, next_y, env))
                
                # 15% - perpendicular left
                left_action = (action - 1) % 4
                next_x, next_y = get_next_state(x, y, left_action, n)
                next_states.append((next_x, next_y))
                next_rewards.append(get_reward_for_state(next_x, next_y, env))
                
                # 15% - perpendicular right
                right_action = (action + 1) % 4
                next_x, next_y = get_next_state(x, y, right_action, n)
                next_states.append((next_x, next_y))
                next_rewards.append(get_reward_for_state(next_x, next_y, env))
                
                # Vectorized Q-value calculation
                next_vals = np.array([V[ns[0], ns[1]] for ns in next_states])
                rewards_arr = np.array(next_rewards)
                q_values[x, y, action] = np.sum(probs * (rewards_arr + gamma * next_vals))
    
    # Vectorized policy update for all non-terminal states
    best_actions = np.argmax(q_values, axis=2)
    new_policy = np.zeros_like(policy)
    
    for x in range(n):
        for y in range(n):
            if not terminal_mask[x, y]:
                new_policy[x, y, best_actions[x, y]] = 1.0
    
    # Check if policy changed (only for non-terminal states)
    policy_stable = np.array_equal(
        np.argmax(old_policy[~terminal_mask], axis=1),
        np.argmax(new_policy[~terminal_mask], axis=1)
    )
    
    # Update policy in place
    policy[:] = new_policy
    
    return policy_stable


# =====================
# OLD IMPLEMENTATION (Loop-based) - KEPT FOR REFERENCE
# =====================
# def policy_improvement_old(env, policy, V, gamma):
#     """
#     Improve policy by acting greedily with respect to value function
#     
#     This is the original loop-based implementation.
#     Commented out in favor of optimized NumPy version above.
#     """
#     n = env.n
#     policy_stable = True
#     
#     for x in range(n):
#         for y in range(n):
#             # Skip terminal states (goal and bad cells)
#             if (x, y) == env.goal_pos or (x, y) in env.bad_cells:
#                 continue
#             
#             old_action = np.argmax(policy[x, y, :])
#             
#             # Calculate Q-values for all actions (considering stochastic transitions)
#             q_values = np.zeros(4)
#             
#             for action in range(4):
#                 # 70% - intended direction
#                 next_x, next_y = get_next_state(x, y, action, n)
#                 reward = get_reward_for_state(next_x, next_y, env)
#                 q_values[action] += 0.70 * (reward + gamma * V[next_x, next_y])
#                 
#                 # 15% - perpendicular left
#                 left_action = (action - 1) % 4
#                 next_x, next_y = get_next_state(x, y, left_action, n)
#                 reward = get_reward_for_state(next_x, next_y, env)
#                 q_values[action] += 0.15 * (reward + gamma * V[next_x, next_y])
#                 
#                 # 15% - perpendicular right
#                 right_action = (action + 1) % 4
#                 next_x, next_y = get_next_state(x, y, right_action, n)
#                 reward = get_reward_for_state(next_x, next_y, env)
#                 q_values[action] += 0.15 * (reward + gamma * V[next_x, next_y])
#             
#             # Update policy to be greedy
#             best_action = np.argmax(q_values)
#             policy[x, y, :] = 0
#             policy[x, y, best_action] = 1.0
#             
#             # Check if policy changed
#             if old_action != best_action:
#                 policy_stable = False
#     
#     return policy_stable


# =====================
# Helper Functions
# =====================

def get_next_state(x, y, action, n):
    """
    Get next state given current state and action
    
    Args:
        x, y: Current position
        action: Action to take (0=right, 1=up, 2=left, 3=down)
        n: Grid size
    
    Returns:
        (next_x, next_y): Next position after action
    """
    if action == 0:  # right
        return x, min(n - 1, y + 1)
    elif action == 1:  # up
        return max(0, x - 1), y
    elif action == 2:  # left
        return x, max(0, y - 1)
    elif action == 3:  # down
        return min(n - 1, x + 1), y


def get_reward_for_state(x, y, env):
    """
    Get reward for a given state
    
    Args:
        x, y: Position
        env: Environment instance
    
    Returns:
        reward: Reward value for the position
    """
    if (x, y) == env.goal_pos:
        return 100.0
    elif (x, y) in env.bad_cells:
        return -50.0
    else:
        return -1.0
