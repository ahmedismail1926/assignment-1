# Grid Maze - Policy Iteration 🎮

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python implementation of the **Policy Iteration** algorithm for solving a stochastic grid maze environment using Gymnasium. This project demonstrates reinforcement learning concepts with visual simulations and comprehensive testing.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 🎯 **Stochastic Environment**: 70% intended action, 15% perpendicular movements
- ⚡ **Optimized Implementation**: NumPy vectorization for faster computation
- 📊 **Comprehensive Testing**: Multi-trial robustness evaluation
- 🎥 **Video Recording**: Automatic policy visualization with PyGame
- 📈 **Detailed Diagnostics**: Path tracking, convergence analysis, loop detection
- 🧪 **Modular Design**: Clean separation of concerns for easy extension

## 🎬 Demo

The agent learns to navigate from start (S) to goal (G) while avoiding bad cells (X):

```
Grid Layout:        Learned Policy:
S . . . G           → → → → G
. X . . .           → X ↑ ↑ ↑
. . . X .           ↓ → → ↑ ↑
. . . . .           ↓ ↓ ↓ → ↑
. . . . .           → → → → ↑
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/grid-maze-policy-iteration.git
   cd grid-maze-policy-iteration
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Create a 5×5 grid maze environment
2. Run policy iteration to find the optimal policy
3. Visualize the learned policy and value function
4. Test policy robustness over 50 trials
5. Record a video of the agent following the policy

## 📁 Project Structure

```
├── environment.py          # Grid Maze Gymnasium environment
├── policy_iteration.py     # Policy iteration algorithm (optimized + legacy)
├── visualization.py        # Visualization and simulation utilities
├── main.py                 # Main execution script
├── modified_test.py        # Original monolithic implementation (legacy)
└── videos/                 # Recorded simulation videos
```

## File Descriptions

### `environment.py`
- Contains the `GridMazeEnv` class (Gymnasium environment)
- Implements stochastic transitions (70% intended, 15% perpendicular each)
- Handles rendering with PyGame
- Observation space: 8 integers [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]

### `policy_iteration.py`
- Implements the policy iteration algorithm
- **Optimized versions** using NumPy vectorization for faster computation
- **Legacy versions** commented out for reference
- Helper functions: `get_next_state()`, `get_reward_for_state()`

### `visualization.py`
- `print_policy_visualization()`: ASCII visualization of learned policy
- `print_value_function()`: Display value function in grid format
- `simulate_policy()`: Run single episode with detailed diagnostics
- `test_policy_robustness()`: Test policy over multiple trials

### `main.py`
- Orchestrates the entire pipeline
- Creates environment, runs policy iteration, visualizes results
- Records video of learned policy

### `modified_test.py`
- Original monolithic implementation (kept for backward compatibility)
- All functionality has been modularized into separate files above

## Usage

Run the main script:
```bash
python main.py
```

Or use individual modules:
```python
from environment import GridMazeEnv
from policy_iteration import policy_iteration
from visualization import print_policy_visualization

env = GridMazeEnv(n=5, seed_nr=42)
policy, V, iterations = policy_iteration(env)
print_policy_visualization(env, policy)
```

## Features

- **Stochastic transitions**: 70% intended action, 15% each perpendicular
- **Optimized computation**: NumPy vectorization for faster policy evaluation
- **Comprehensive testing**: Robustness testing with multiple trials
- **Video recording**: Automatic video generation of learned policy
- **Detailed diagnostics**: Path tracking, loop detection, convergence analysis

## Reward Structure

- Goal cell: +100
- Bad cells: -50
- Normal cells: -1 (step penalty)

## 🧮 Algorithm Details

### Policy Iteration Process

1. **Initialization**: Start with a random policy
2. **Policy Evaluation**: Calculate value function for current policy
3. **Policy Improvement**: Update policy greedily based on values
4. **Repeat** until policy converges (typically 3-6 iterations)

### Stochastic Transitions

- **70%** probability: Agent moves in intended direction
- **15%** probability: Agent moves perpendicular left
- **15%** probability: Agent moves perpendicular right

### Reward Structure

| State | Reward | Purpose |
|-------|--------|---------|
| Goal cell | +100 | Encourage reaching goal |
| Bad cells | -50 | Strong penalty to avoid |
| Normal cells | -1 | Step penalty for efficiency |

## 💻 Usage Examples

### Basic Usage

```python
from environment import GridMazeEnv
from policy_iteration import policy_iteration
from visualization import print_policy_visualization

# Create environment
env = GridMazeEnv(n=5, seed_nr=42)

# Run policy iteration
policy, V, iterations = policy_iteration(env, gamma=0.95, theta=1e-6)

# Visualize results
print_policy_visualization(env, policy)
```

### Custom Configuration

```python
# Larger grid
env = GridMazeEnv(n=10, seed_nr=123, render_mode='human')

# Different discount factor
policy, V, iterations = policy_iteration(env, gamma=0.99, theta=1e-8)

# More robust testing
success_rate, _, _ = test_policy_robustness(env, policy, num_trials=100)
```

## 📊 Performance

- **5×5 grid**: Converges in ~3-5 iterations, <1 second
- **10×10 grid**: Converges in ~4-7 iterations, ~2-5 seconds
- **Success rate**: Typically 80-100% depending on maze layout

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) framework
- Inspired by Sutton & Barto's "Reinforcement Learning: An Introduction"
- Visualization powered by [PyGame](https://www.pygame.org/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: See [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md) for detailed instructions on uploading this project to GitHub.
