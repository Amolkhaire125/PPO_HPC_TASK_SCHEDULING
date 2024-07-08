import configparser
config2 = configparser.ConfigParser()
config2.read('./agent.cfg')
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from collections import deque
from torch.optim import Adam
import time
import os
import random
import gym
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO_TEST:
    def __init__(self, actor_model, policy_class, env):
        """Initialize PPO agent.
        :param config: Configuration dictionary
        :param env: Gym environment
        :param model: PyTorch model
        :param writer: TensorBoard writer (optional)
        """
        self.config = config2
        self.env = env
        self.time = 0
        self.obs_dim = env.num_cluster
        self.act_dim = env.num_cluster
        
        # Load actor network
        self.actor = policy_class(self.obs_dim, self.act_dim).to(device)                                  
        self.actor.load_state_dict(torch.load(actor_model))

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}


    def find_available_proc(self):
        """
    This function finds the next available processing unit (proc) in each cluster.
    If a proc is currently running a task, it will move to the next proc.
    If all procs in a cluster are busy, it will call the _forward_in_time_ function.

    Parameters:
    None

    Returns:
    None
    """
        for i in range(self.env.num_cluster):
            while (self.env.current_proc[i] < self.env.p) and (self.env.running[i][self.env.current_proc[i]] > -1):
                self.env.current_proc[i] += 1 
        
        if self.env.current_proc[self.C] == self.env.p: 
            # self.env.current_proc[self.C] = 0
            self._forward_in_time_()
        # print("self.env.current_proc..............", self.env.current_proc)  


    def select_task_runtime(self, task, Cluster):
        """
    This function selects the runtime of a task based on the cluster it belongs to.

    Parameters:
    task (Task): The task object for which the runtime needs to be selected.
    Cluster (int): The cluster number (0, 1, or 2) for which the runtime needs to be selected.

    Returns:
    float: The runtime of the task for the specified cluster.
    """
        task_runtime = 0
        if Cluster == 0:
            task_runtime = task.Runtime_C1
        elif Cluster == 1:
            task_runtime = task.Runtime_C2
        elif Cluster == 2:
            task_runtime = task.Runtime_C3    
        return task_runtime
    

    def _forward_in_time_(self):  
        """
    This function moves the simulation forward in time for the current cluster.
    It finds the minimum time for the next task to be ready and updates the ready_proc and running arrays accordingly.
    It also deletes the finished tasks from the running_task2proc dictionary.

    Parameters:
    None

    Returns:
    None
    """
        if len(self.env.ready_proc[self.C][self.env.ready_proc[self.C] > self.time]) > 0:
            min_time = np.min(self.env.ready_proc[self.C][self.env.ready_proc[self.C] > self.time])
        else:
            min_time = 0

        self.time_c = min_time                       
        # self.time = max(min_time, self.time)
        # print("self.time_c......", self.time_c)
        # print("self.env.ready_proc_1", self.env.ready_proc)
        for i in range(len(self.env.ready_proc)):
            self.env.ready_proc[self.C][self.env.ready_proc[self.C] < self.time_c] = self.time_c  
            tasks_finished = self.env.running[self.C][np.logical_and(self.env.ready_proc[self.C] == self.time_c, self.env.running[self.C] > -1)].copy()
            self.env.running[self.C][self.env.ready_proc[self.C] == self.time_c] = -1
            self.env.current_proc[self.C] = np.argmin(self.env.running[self.C])
        # print("self.env.ready_proc_2", self.env.ready_proc)
        # print("self.env.running...........", self.env.running)
        # tasks_finished = self.env.running[self.C][np.logical_and(self.env.ready_proc[self.C] == self.time, self.env.running[self.C] > -1)].copy()
        # print("tasks_finished", tasks_finished)
        # self.env.running[self.C][self.env.ready_proc[self.C] == self.time] = -1
        # self.env.running[self.env.ready_proc == self.time] = -1
        for task in tasks_finished:
            del self.env.running_task2proc[task]  


    def _forward_in_time(self):  
        """
        This function moves the simulation forward in time for all clusters.
        It finds the minimum time for the next task to be ready and updates the ready_proc and running arrays accordingly.
        It also deletes the finished tasks from the running_task2proc dictionary.

        Parameters:
        None

        Returns:
        None
        """
        min_time = float('inf')
        print("self.time............................", self.time)
        for i in range(self.env.num_cluster):  
            if len(self.env.ready_proc[i][self.env.ready_proc[i] > self.time]) > 0:
                min_time_c = np.min(self.env.ready_proc[i][self.env.ready_proc[i] > self.time])
            else:
                min_time_c = 0

            if min_time_c > 0:
                min_time = min(min_time_c, min_time)

            print("min_time_BLACKSHIP........", min_time_c)    
            print("min_time_c........", min_time)    

        self.time = min_time                        
        for i in range(len(self.env.ready_proc)):
            self.env.ready_proc[i][self.env.ready_proc[i] < self.time] = self.time  
            tasks_finished = self.env.running[self.C][np.logical_and(self.env.ready_proc[self.C] == self.time, self.env.running[self.C] > -1)].copy()
            self.env.running[i][self.env.ready_proc[i] == self.time] = -1
            self.env.current_proc[i] = np.argmin(self.env.running[i])
        
        for task in tasks_finished:
            del self.env.running_task2proc[task]

        if self.env.current_proc[self.C] == self.env.p: 
            self._forward_in_time_()  


    def calculate_task_start_time(self, task, parent_end_times):
        """ Calculate starting time for each task """
        print("task.parent_task[0]", task.parent_task[0])
        if not task.parent_task or task.parent_task[0] == 0:
            return 0  # No dependencies or first task, so start time
        max_dependency_end_time = 0
        print("taskID....", task.taskID)
        for parent_task_id in task.parent_task:
            parent_task_end_time = parent_end_times[parent_task_id]
            max_dependency_end_time = max(max_dependency_end_time, parent_task_end_time)
        return max_dependency_end_time


    def evaluate_test(self):
        """
        Evaluate current policy.

        This function simulates the scheduling process for a given set of tasks,
        considering the dependencies between tasks and the available processing units.
        It calculates the energy consumption, makespan, and other relevant metrics.

        Parameters:
        None

        Returns:
        task_completion_list_ (list): A list of tuples representing the completion of each task.
        Make_Span (float): The maximum end time of all tasks.
        rounded_energy (float): The total energy consumption, rounded to two decimal places.
        energy_makespan_list (list): A list of tuples containing the makespan and energy consumption for each iteration.
        node_count_dict (dict): A dictionary representing the number of tasks scheduled on each processing unit.
        """
        self.time = 0
        self.Each_iteration_energy = 0
        self.task_completion_list_iteration = []
        energy_makespan_list = []
        self.node_count_dict = {}
        self.task_scheduled = []
        self.rejected_job = []
        start_time = time.time()
        self.tasks = self.env.reset()  

        parent_end_times = {}
        self.task_completion_list_ = []
        with torch.no_grad():
            while len(self.tasks) > 0:
                schedulable_tasks = []
                for t in self.tasks:
                    if t.status == 1 and t.taskID not in self.task_scheduled and t.jobID not in self.rejected_job:
                        parents_scheduled = True
                        for parent in t.parent_task:
                            if parent not in self.task_scheduled and parent != 0:
                                parents_scheduled = False
                                break
                        if parents_scheduled:
                            schedulable_tasks.append(t)   
                
                if not schedulable_tasks:
                    # No tasks can be scheduled in this iteration; break out of the loop
                    break   

                for t in schedulable_tasks.copy():
                    observation = np.array([t.Runtime_C1, t.Runtime_C2, t.Runtime_C3])  
                    observation_tensor = torch.FloatTensor(observation)
                    action_probs = self.actor(observation_tensor)
                    policy = F.softmax(action_probs, dim=-1)
                    action = policy.argmax().cpu().numpy()
                    # ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
                    # action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw]
                    self.C = action
                    # reward, done = self.env.step(action)
                    task_runtime = self.select_task_runtime(t, action)
                    self.N = self.env.current_proc[self.C]
                    parent_task_finishtime = self.calculate_task_start_time(t, parent_end_times)
                    t.starttime = max(self.env.ready_proc[self.C][self.N], parent_task_finishtime)
                    self.env.ready_proc[self.C][self.N] = t.starttime
                    print("self.starttime..... ", t.starttime)
                    self.env.ready_proc[self.C][self.N] += task_runtime
                    t.end_time = self.env.ready_proc[self.C][self.N]
                    print("t.end_time.......", t.end_time)
                    Power = self.env.getpwr(action, t.CPU)
                    enrgy_for_task = (Power*task_runtime)/3600
                    self.Each_iteration_energy += enrgy_for_task
                    self.env.running_task2proc[t.taskID] = [self.C, self.N]
                    self.env.running[self.C][self.N] = t.taskID
                    self.find_available_proc()
                    if t.taskID not in parent_end_times:
                        parent_end_times[t.taskID] = t.end_time     
                    if self.C+1 not in  self.node_count_dict:
                        self.node_count_dict[self.C+1] = [self.N+1]
                    elif self.N+1 not in self.node_count_dict[self.C+1]:
                        self.node_count_dict[self.C+1].append(self.N+1)    
                    self.task_scheduled.append(t.taskID)
                    self.task_completion_list_.append((t.taskID, self.C+1, self.N+1, t.starttime, t.end_time, t.jobID, t.task_type))
                    schedulable_tasks.remove(t)
                    self.tasks.remove(t)
                    if len(schedulable_tasks) == 0:
                        self._forward_in_time()
        
        jobs = organize_tasks_by_job(self.task_completion_list_)
        for jobID, job in jobs.items():                
            max_end_time = max(job, key=lambda x: x[4])
            Make_Span = max_end_time[4]    
    
        rounded_energy = round(self.Each_iteration_energy, 2)
        energy_makespan_list.append((Make_Span, rounded_energy)) 

        return self.task_completion_list_, Make_Span, rounded_energy, energy_makespan_list, self.node_count_dict
  