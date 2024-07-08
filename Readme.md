# PPO Algorithm for Task Scheduling

This repository contains an implementation of the Proximal Policy Optimization algorithm for task scheduling in a High-Performance Computing (HPC) environment. The PPO algorithm is a reinforcement learning technique that learns to schedule tasks efficiently on available resources while considering objectives such as minimizing makespan and energy consumption.


## Usage

1. Prepare your task data file in the required format.
2. Run the script with appropriate arguments: python main.py --csv_file data/file/path --num_node <nCPU> --nsga_time <nsga_makespan> --mode <train/test>
3. The script will train the PPO agent. 
4. Upon completion, the script will generate CSV files containing the optimal schedule for makespan and energy consumption, as well as a Pareto list for each iteration.




## Files

- `main.py`: The main script to run the PPO algorithm.
- `environment.py`: Contains the environment definition for the task scheduling problem.
- `train.py`: Training Implementation of the PPO algorithm.
- `test.py`: Testing Implementation of the PPO algorithm.
- `network.py`: Defines the neural network models used by the PPO agent.
- `utils.py`: Utility functions for the environment.
- `requirements.txt`: List of required Python packages.
- `models.py`: It download model inside Models Folder.


## To test the PPO algorithm,  agent file need to be given as input in agent.cfg file
