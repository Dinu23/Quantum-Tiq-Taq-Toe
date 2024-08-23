# Reinforcement Learning for Quantum Tic-Tac-Toe

This repository contains the code and documentation for exploring reinforcement learning (RL) techniques applied to Quantum Tic-Tac-Toe. The project aims to integrate quantum computing concepts with RL algorithms to create an innovative approach to the classic game of Tic-Tac-Toe.

## Introduction

Quantum Tic-Tac-Toe \cite{goff2006quantum} is an extended version of the classic Tic-Tac-Toe game that incorporates quantum mechanics, such as superposition and entanglement, into the gameplay. This project explores the application of reinforcement learning methods to Quantum Tic-Tac-Toe, providing insights into how RL can be leveraged in quantum environments.

Despite the availability of Quantum Chess \cite{youvan2024sequential,cantwell2019quantum}, which is more complex, Quantum Tic-Tac-Toe serves as a more accessible testbed for combining quantum computing and RL. Our approach includes the following:

- **Two Versions of the Game**: Each with different rules regarding entanglement moves.
- **Reinforcement Learning Agents**: Implementing and comparing different RL strategies.
- **Analysis**: Evaluating the performance of agents under various game rules.

## Methodology

### Game Versions

We investigate two distinct versions of Quantum Tic-Tac-Toe:

1. **Version 1 (V1)**: Entanglement moves are restricted to pairs containing at least one empty cell.
2. **Version 3 (V3)**: Any pair of cells can be used for entanglement moves.

For a detailed explanation of the game's rules and state representations, refer to the [Appendix](#appendix).

### Representation of the State

Quantum Tic-Tac-Toe features challenges such as partial observability and exponential state complexity. We use two methods to represent the game state:

- **Measurements**: A 3x3 matrix of state probabilities.
- **Move History**: A 9x9 matrix tracking historical entanglement relations.

### Reinforcement Learning

We use Proximal Policy Optimization (PPO) \cite{schulman2017proximal,tang2020implementing,liu2021self} to train agents in Quantum Tic-Tac-Toe. Agents are compared based on their performance with different types of information (measurement matrices, historical entanglement records).

## Results

### Version 1 (V1)

The first set of results shows that the first player tends to have an advantage due to the constraints on entanglement moves. This suggests that certain strategies may be more effective, even within the randomness of the quantum environment.

![V1 Reward](figures/reward_v1.png)
*Average reward on 100 games during the training of different agents.*

![V1 Results](figures/results_V1.png)
*Pitting best agents: X-Wins, O-Wins, Draws.*

### Version 3 (V3)

In Version 3, where entanglement constraints are removed, combining measurement matrices with historical entanglement records leads to better performance and more balanced outcomes.

![V3 Reward](figures/reward_v3.png)
*Average reward on 100 games during the training of different agents.*

![V3 Results](figures/results_V3.png)
*Pitting best agents: X-Wins, O-Wins, Draws.*

## Discussion

Quantum Tic-Tac-Toe offers a valuable testbed for RL in quantum settings. The results highlight the importance of comprehensive information (both measurements and historical records) for effective decision-making. Future work could explore more advanced RL techniques and methods to better address partial observability, such as using recurrent neural networks or transformers.

## Installation

To get started with this project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Dinu23/Quantum-Tiq-Taq-Toe.git
   cd Quantum-Tiq-Taq-Toe
   ```

2. **Install Dependencies**

   Make sure you have Python installed, then install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**

   Execute the main script to start training and evaluating RL agents:

   ```bash
   python main.py
   ```


## Contact

For questions or feedback, please contact:

- **Catalin Dinu**: [viorel.dinu00@gmail.com](viorel.dinu00@gmail.com)

## Appendix

### A. Environment

Quantum Tic-Tac-Toe is played on a 3x3 board where each cell can be in a superposition of three states: empty, X, or O. The state of the game is represented as a linear combination of all possible classical states.

**Action Space**: Moves include standard X/O placements and quantum entanglements between pairs of cells.

**State Collapsing**: This occurs when the game board is fully populated, collapsing the quantum state to one of the possible classical states.

### B. Observation Space

**Measurements**: Estimated probabilities of each cell being in a specific state based on simulations.

**Move History**: Matrices tracking past moves, including entanglements and classical moves.
