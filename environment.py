import gym
from gym import spaces
import pygame
import numpy as np
from utils import *
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('./environment.cfg')
server_data = config['Servers']

class HPCTaskSchedulingEnv(gym.Env):
    def __init__(self, num_clusters, num_nodes, data_file, reward_type, nsga_time):
        """Initialize the HPC Environment """
        
        self.num_cluster = num_clusters  
        self.p = self.num_nodes = num_nodes 
        self.task_data = readCSV(data_file) 
        self.num_tasks = len(self.task_data)
        self.nsga_time = nsga_time
        self.reward = reward_type
        # Define the observation space
        self.observation_space = spaces.Dict({
            "tasks": spaces.Box(
                low=0,
                high=self.num_tasks - 1,
                shape=(self.num_tasks,),
                dtype=int
            ),
            "clusters": spaces.Box(
                low=0,
                high=self.num_cluster - 1,
                shape=(self.num_tasks,),
                dtype=int
            )
        })
               
        self.action_space = spaces.Discrete(self.num_cluster)
        self.current_proc = [0 for i in range(self.num_cluster)]
        self.reset()


    def reset(self):
        """Reset the observation space"""

        self.running = [-1 * np.ones(self.p) for i in range(self.num_cluster)]
        self.running_task2proc = {}
        self.ready_proc = [np.zeros(self.p) for i in range(self.num_cluster)]
        # self.ready_tasks.append(0)
        self.current_proc = [0 for i in range(self.num_cluster)]
        self.task_proc_map = []       
        self.tasks_ = self.task_data.copy()
        self.processed = {}
        self.compeur_task = 0
        for t in self.tasks_:
            t.status = 1 
        self.Each_iteration_time = 0
    
        return self.tasks_



    def step(self, makespan, energy, tasks, waiting_time):
        """Check if all tasks are done or not and calculate reward for this step."""

        if self._isdone(tasks):                                                             
            done = True
        else:
            done = False   

        # reward = (self.heft_time - makespan)/self.heft_time if done else 0
        if self.reward == "makespan":   
            # reward = - makespan
            reward =  ((self.nsga_time - makespan)/self.nsga_time)+2  if done else 0
            
        elif self.reward == "energy":    
            reward =  ((self.nsga_time - energy)/self.nsga_time)+2 if done else 0
        elif self.reward == "weighted":
            reward = (0.5*((self.nsga_time - makespan)/self.nsga_time) + 0.5*((self.nsga_time - makespan)/self.nsga_time)) if done else 0         
        # reward = - makespan
        reward += (waiting_time/100)
        if waiting_time <= 0:
            reward += 0.5

        return reward, done   



    def _isdone(self, tasks):
        '''Check if all tasks are done'''
        if len(tasks) == 0:
            return True
        return False     
    

    def getpwr(self, processor_name, CPU):
        '''calculate power consumption'''
        coef1 = float(config['coefficients']['coef1'])
        coef2 = float(config['coefficients']['coef2'])
        coef3 = float(config['coefficients']['coef3'])
        processor_data = config['power_ratings']['processor{}'.format(processor_name)]
        idle_power, peak_power = map(float, processor_data.split(','))
        total_power = idle_power + (peak_power - idle_power) * (coef1 * CPU + coef2 * np.sqrt(CPU) + (CPU ** coef3))
        
        return total_power 