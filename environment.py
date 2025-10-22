"""
Grid Maze Gym Environment
=========================
A custom Gymnasium environment for a grid-based maze with:
- Stochastic movement: 70% intended, 15% perpendicular directions
- Observation: 8 integers [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
"""

import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import pygame


class GridMazeEnv(gym.Env):
    """
    Grid Maze Environment with stochastic transitions
    
    Args:
        n (int): Grid size (n x n)
        seed_nr (int): Random seed for reproducibility
        render_mode (str): 'human' or 'rgb_array'
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
        Execute action with stochastic transitions
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
