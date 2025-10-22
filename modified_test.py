import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.wrappers import RecordVideo


# =====================
# Grid Maze Gym Environment
# =====================
class GridMazeEnv(gym.Env):
    """
    - Stochastic movement: 70% intended, 15% perpendicular directions
    - Observation: 8 integers [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, n=5, seed_nr=None, render_mode=None):
        super(GridMazeEnv, self).__init__()
        
        # Set seed for reproducibility
        if seed_nr is not None:
            random.seed(seed_nr)
            np.random.seed(seed_nr)
        self.n = n
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 8 integers (4 coordinate pairs)
        self.observation_space = spaces.Box(
            low=0, 
            high=n-1, 
            shape=(8,), 
            dtype=np.int32
        )
        
        # Generate random positions for S, G, and 2 X cells
        all_positions = [(i, j) for i in range(n) for j in range(n)]
        random.shuffle(all_positions)
        
        self.start_pos = all_positions[0]
        self.goal_pos = all_positions[1]
        self.bad_cells = [all_positions[2], all_positions[3]]
        
        # Current agent position
        self.agent_pos = self.start_pos
        
        # Rendering
        self.cell_size = 100
        self.screen = None
        self.clock = None
        
        # Store initial state for reset
        self._initial_positions = {
            'start': self.start_pos,
            'goal': self.goal_pos,
            'bad': self.bad_cells.copy()
        }

    def reset(self, seed=None, options=None):
        """Reset environment to starting position"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.agent_pos = self.start_pos
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def _get_obs(self):
        """Return observation: [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]"""
        obs = np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.bad_cells[0][0], self.bad_cells[0][1],
            self.bad_cells[1][0], self.bad_cells[1][1]
        ], dtype=np.int32)
        return obs

    def step(self, action):
        """
        70% intended direction, 15% each perpendicular direction
        Action mapping:
        0 = right  1 = up   2 = left  3 = down
        """
        # Stochastic action selection
        rand_val = random.random()
        
        if rand_val < 0.70:
            # 70% - intended direction
            actual_action = action
        else:
            # 30% - perpendicular directions (15% each)
            if action == 0 or action == 2:  # right or left
                # Perpendiculars are up (1) and down (3)
                perpendiculars = [1, 3]
            else:  # action == 1 or action == 3  # up or down
                # Perpendiculars are right (0) and left (2)
                perpendiculars = [0, 2]
            
            # Choose one of the two perpendiculars randomly (50% each = 15% overall)
            actual_action = random.choice(perpendiculars)
        
        # Execute actual action
        x, y = self.agent_pos
        # wall boundaries to prevent moving outside the grid
        if actual_action == 0:  # right
            y = min(self.n - 1, y + 1)
        elif actual_action == 1:  # up
            x = max(0, x - 1)
        elif actual_action == 2:  # left
            y = max(0, y - 1)
        elif actual_action == 3:  # down
            x = min(self.n - 1, x + 1)
        self.agent_pos = (x, y)
        
        reward = self._get_reward()
        
        # Check if episode is done
        terminated = (self.agent_pos == self.goal_pos) or (self.agent_pos in self.bad_cells)
        truncated = False
        
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_reward(self):
        """
        Reward function:
        - Goal: +100 (encourage reaching goal)
        - Bad cells: -50 (strong penalty to avoid)
        - Normal cells: -1 (small penalty to encourage efficiency)
        """
        if self.agent_pos == self.goal_pos:
            return 100.0
        elif self.agent_pos in self.bad_cells:
            return -50.0
        else:
            return -1.0

    def render(self):
        """Render the environment using PyGame"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.n * self.cell_size, self.n * self.cell_size)
                )
                pygame.display.set_caption("Grid Maze - Policy Iteration")
            else:  # rgb_array
                self.screen = pygame.Surface(
                    (self.n * self.cell_size, self.n * self.cell_size)
                )
            self.clock = pygame.time.Clock()
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 200, 0)
        RED = (200, 0, 0)
        BLUE = (0, 100, 255)
        YELLOW = (255, 215, 0)
        GRAY = (200, 200, 200)
        
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw grid cells
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(
                    j * self.cell_size, 
                    i * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Color based on cell type
                if (i, j) == self.goal_pos:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif (i, j) in self.bad_cells:
                    pygame.draw.rect(self.screen, RED, rect)
                elif (i, j) == self.start_pos:
                    pygame.draw.rect(self.screen, YELLOW, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, BLACK, rect, 2)
                
                # Draw labels
                font = pygame.font.Font(None, 36)
                if (i, j) == self.goal_pos:
                    text = font.render('G', True, BLACK)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                elif (i, j) in self.bad_cells:
                    text = font.render('X', True, WHITE)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
                elif (i, j) == self.start_pos and self.agent_pos != self.start_pos:
                    text = font.render('S', True, BLACK)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        
        # Draw agent (blue circle)
        agent_x, agent_y = self.agent_pos
        center = (
            agent_y * self.cell_size + self.cell_size // 2,
            agent_x * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, BLUE, center, self.cell_size // 3)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Close the rendering window"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# =====================
# Policy Iteration Algorithm
# =====================
def policy_iteration(env, gamma=0.95, theta=1e-6):
    """
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


# # OLD IMPLEMENTATION (Loop-based)
# def policy_evaluation_old(env, policy, V, gamma, theta):
#     """
#     Evaluate the current policy using stochastic transitions
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


# # OLD IMPLEMENTATION (Loop-based)
# def policy_improvement_old(env, policy, V, gamma):
#     """
#     Improve policy by acting greedily with respect to value function
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


def get_next_state(x, y, action, n):
    """
    Get next state given current state and action
    0=right, 1=up, 2=left, 3=down
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
    if (x, y) == env.goal_pos:
        return 100.0
    elif (x, y) in env.bad_cells:
        return -50.0
    else:
        return -1.0


# =====================
# Visualization Functions
# =====================
def print_policy_visualization(env, policy):
    """Print ASCII visualization of the policy"""
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


def simulate_policy(env, policy, max_steps=100):
    """
    Simulate the learned policy and record video
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
            from collections import Counter
            state_counts = Counter(path)
            most_common = state_counts.most_common(3)
            print(f"   Most visited states:")
            for state, count in most_common:
                print(f"      {state}: {count} times (action: {action_symbols[np.argmax(policy[state[0], state[1], :])]})")
    
    return step_count, total_reward, path
def test_policy_robustness(env, policy, num_trials=20, max_steps=100):
    """
    Test the learned policy over multiple trials to measure robustness
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


# =====================
# Main Execution
# =====================
if __name__ == "__main__":
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
    print("\nValue Function (rounded to 2 decimals):")
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
    
    # Test policy robustness
    success_rate, successes, failures = test_policy_robustness(env, policy, num_trials=50, max_steps=100)
    
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
    
    import os
    from datetime import datetime
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