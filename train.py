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
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class PPO:
    def __init__(self, policy_class, env, num_node):
        """Initialize PPO agent.
        :param config: Configuration dictionary
        :param env: Gym environment
        :param model: PyTorch model
        :param writer: TensorBoard writer (optional)
        """
        self.num_node = num_node
        self.config = config2
        self.env = env
        self.gamma = float(config2['Hyperparameters']['gamma'])
        self.num_epochs = int(config2['Hyperparameters']['num_epochs'])
        self.clip_ratio = float(config2['Hyperparameters']['clip_ratio'])
        self.entropy_coef = float(config2['Hyperparameters']['entropy_coef'])
        self.initial_entropy_coeff = float(config2['Hyperparameters']['initial_entropy_coeff'])
        self.final_entropy_coeff = float(config2['Hyperparameters']['final_entropy_coeff'])
        self.grad_clip = float(config2['Hyperparameters']['grad_clip'])
        self.num_minibatches = int(config2['Hyperparameters']['num_minibatches'])
        self.lam = float(config2['Hyperparameters']['lam'])
        self.n_iterations = int(config2['Hyperparameters']['n_iterations'])
        self.noise = 0
        self.time = 0
        self.reward_log = deque(maxlen=10)
        self.time_log = deque(maxlen=10)
        self.obs_dim = env.num_cluster
        self.act_dim = env.num_cluster
        
        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim).to(device)                                  
        self.critic = policy_class(self.obs_dim, 1).to(device)

		# Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=float(config2['Hyperparameters']['LR']))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(config2['Hyperparameters']['LR']))

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}


    def _returns_advantages(self, rewards, dones, values, next_value):
        """ Compute returns and advantages using GAE."""
        # Convert inputs to numpy arrays
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        next_value = np.float32(next_value)

        # Ensure gamma and lam are float32
        gamma = np.float32(self.gamma)
        lam = np.float32(self.lam)

        # Initialize returns and advantages
        returns = np.zeros(len(rewards) + 1, dtype=np.float32)
        returns[-1] = next_value
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lambda = np.float32(0)


        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = np.float32(1.0 - dones[t])
                next_val = next_value
            else:
                next_non_terminal = np.float32(1.0 - dones[t])
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lambda = delta + gamma * lam * next_non_terminal * last_gae_lambda
            returns[t] = advantages[t] + values[t]

        return returns[:-1], advantages


    def optimize_model(self, observations, actions, log_probs, returns, advantages, step=None):
        """ Optimize model using PPO loss."""
        # Convert inputs to PyTorch tensors
        observations = torch.tensor(np.array(observations), dtype=torch.float, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        
        # Handle log_probs carefully
        if isinstance(log_probs, (list, np.ndarray)):
            log_probs = torch.tensor(np.array(log_probs), dtype=torch.float, device=device)
        elif isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.to(device)
        elif isinstance(log_probs, (float, np.float32, np.float64)):
            log_probs = torch.tensor([log_probs], dtype=torch.float, device=device)
        else:
            raise TypeError(f"Unexpected type for log_probs: {type(log_probs)}")

        returns = torch.tensor(np.array(returns), dtype=torch.float, device=device).unsqueeze(1)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float, device=device)

        # Ensure all tensors have the same first dimension
        batch_size = observations.shape[0]
        if log_probs.shape[0] != batch_size:
            log_probs = log_probs.expand(batch_size)

        # Forward pass
        policies = self.actor(observations)
        values = self.critic(observations)
        policies = F.softmax(policies, dim=-1)

        # Compute new log probabilities
        new_log_probs = torch.log(policies.gather(1, actions.unsqueeze(1))).squeeze(1)

        # Compute ratio and surrogate objectives
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

        # Compute losses
        loss_actor = -torch.min(surr1, surr2).mean()
        loss_critic = F.mse_loss(values, returns)
        loss_entropy = -(policies * torch.log(policies)).sum(-1).mean()

        # Optimize actor
        self.actor_optim.zero_grad()
        actor_loss = loss_actor - self.entropy_coef * loss_entropy
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        # Optimize critic
        self.critic_optim.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        # Log losses if logger is available
        self.logger['actor_losses'].append(loss_actor.item())

        # If you want to use TensorBoard for logging
        if hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar('critic_loss', loss_critic.item(), step)
            self.writer.add_scalar('actor_loss', loss_actor.item(), step)
            self.writer.add_scalar('entropy', loss_entropy.item(), step)


    def _forward_in_time_(self):  
        """
        Update the ready_proc and running arrays to reflect the progression of time.
        This method is called when a certain condition is met, such as when all tasks
        assigned to a specific cluster and processor are completed.

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
        Update the time and process states based on the minimum ready time.
        If the current process is full, call the _forward_in_time_() method.

        Parameters:
        None

        Returns:
        None
        """
        min_time = float('inf')
        for i in range(self.env.num_cluster):  
            if len(self.env.ready_proc[i][self.env.ready_proc[i] > self.time]) > 0:
                min_time_c = np.min(self.env.ready_proc[i][self.env.ready_proc[i] > self.time])
            else:
                min_time_c = 0

            if min_time_c > 0:
                min_time = min(min_time_c, min_time)

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


    def select_task_runtime(self, task, Cluster):
        """
        Select the runtime of a task based on the cluster it is assigned to.

        Parameters:
        task (Task): The task for which the runtime needs to be selected.
        Cluster (int): The cluster to which the task is assigned.

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


    def find_available_proc(self):
        """
        Find the next available processor for each cluster. If all processors are busy,
        call the `_forward_in_time_` method to move the time forward.
        """
        for i in range(self.env.num_cluster):
            while (self.env.current_proc[i] < self.env.p) and (self.env.running[i][self.env.current_proc[i]] > -1):
                self.env.current_proc[i] += 1 
        
        if self.env.current_proc[self.C] == self.env.p: 
            # self.env.current_proc[self.C] = 0
            self._forward_in_time_()
        # print("self.env.current_proc..............", self.env.current_proc)  


    def calculate_task_start_time(self, task, parent_end_times):
        """ Calculate starting time for each task """
        if not task.parent_task or task.parent_task[0] == 0:
            return 0  # No dependencies or first task, so start time
        max_dependency_end_time = 0
        print("taskID....", task.taskID)
        for parent_task_id in task.parent_task:
            parent_task_end_time = parent_end_times[parent_task_id]
            max_dependency_end_time = max(max_dependency_end_time, parent_task_end_time)
        return max_dependency_end_time


    def evaluate(self, render=False):
        """Evaluate current policy."""
        total_reward = 0
        self.time = 0
        start_time = time.time()
        self.tasks = self.env.reset()      
        parent_end_times = {}
        self.task_scheduled = []
        self.task_completion_list_ = []
        self.rejected_job = []
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
        return Make_Span


    def current_entropy_coeff(self, num_of_iterations, iterations):
        """
        Calculate the current entropy coefficient based on the number of iterations and current iteration.

        Parameters:
        num_of_iterations (int): The total number of iterations for training.
        iterations (int): The current iteration number.

        Returns:
        float: The current entropy coefficient.
        """
        return self.initial_entropy_coeff - (self.initial_entropy_coeff - self.final_entropy_coeff) * (iterations / num_of_iterations)


    def training_batch(self):
        """Perform a training by batch
        Parameters
        ----------
        steps : int
            Number of steps
        batch_size : int
            The size of a batch
        """
        start = time.time()
        reward_log = deque(maxlen=10)
        time_log = deque(maxlen=10)
        num_of_iterations = int(config2['Hyperparameters']['no_of_iterations'])
        batch_size = int(config2['Hyperparameters']['batch_size'])

        actions = np.empty((batch_size,), dtype=int)
        dones = np.empty((batch_size,), dtype=bool)
        rewards, values = np.empty((2, batch_size), dtype=float)              
        log_ratio = 0
        best_time = float('inf')
        n_step = 0
        iteration_counter = 0
        updations = []
        best_actor_path = None
        best_critic_path = None

        make_span_curve = []
        power_list = []
        time_list = []
        energy_makespan_list = []
        task_completion_list = []
        node_engaged = []
        waiting_time_list = []
        for i in range(num_of_iterations):
            self.time = 0
            self.tasks = self.env.reset()
            self.Each_iteration_energy = 0
            self.task_completion_list_iteration = []
            self.node_count_dict = {}
            self.task_scheduled = []
            self.rejected_job = []
            parent_end_times = {}
            waiting_time_iteration = []
            observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            
            while len(self.tasks) > 0:
                task_runtime = 0   
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
                #print("schedulable_tasks", [t.taskID for t in schedulable_tasks] )
                if not schedulable_tasks:
                    # No tasks can be scheduled in this iteration; break out of the loop
                    break   
                
                for t in schedulable_tasks.copy():
                    """Collect trajectories using current policy.""" 
                    observation = np.array([t.Runtime_C1, t.Runtime_C2, t.Runtime_C3])     
                    observations.append(observation)
                    # policy, value = self.network(observation)
                    # policy = F.softmax(policy, dim=-1)
                    # action = np.random.choice(len(policy), p=policy)
                    observation_tensor = torch.FloatTensor(observation)
                    action_probs = self.actor(observation_tensor)
                    value = self.critic(observation_tensor)
                    policy = F.softmax(action_probs, dim=-1)
                    action = torch.multinomial(policy, 1).item()


                    # action_probs = self.actor(observation)
                    # print("action_probs", action_probs)
                    # value = self.critic(observation) 
                    # print("value", value)              
                    # policy = F.softmax(action_probs, dim=-1)
                    # print("policy", policy)
                    # action = torch.multinomial(policy, 1).detach().cpu().numpy()
                    self.C = action
                    log_prob = torch.log(policy[action])
                    task_runtime = self.select_task_runtime(t, action)
                    self.N = self.env.current_proc[self.C]
                    print("parent_end_times", parent_end_times)
                    parent_task_finishtime = self.calculate_task_start_time(t, parent_end_times)
                    print("parent_task_finishtime", parent_task_finishtime)
                    print("self.env.ready_proc", self.env.ready_proc)
                    t.starttime = max(self.env.ready_proc[self.C][self.N], parent_task_finishtime)
                    waiting_time = parent_task_finishtime - t.starttime
                    waiting_time_iteration.append(waiting_time)
                    self.env.ready_proc[self.C][self.N] = t.starttime
                    print("self.starttime ", t.starttime)
                    self.env.ready_proc[self.C][self.N] += task_runtime
                    t.end_time = self.env.ready_proc[self.C][self.N]
                    print("t.end_time", t.end_time)
                    Power = self.env.getpwr(self.C, t.CPU)
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
                    self.task_completion_list_iteration.append((t.taskID, self.C+1, self.N+1, t.starttime, t.end_time, t.jobID, t.task_type))
                    t.status = 2 
                    # ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
                    # action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw]
                    schedulable_tasks.remove(t)
                    self.tasks.remove(t)
                    makespan = max([np.max(arr) for arr in self.env.ready_proc])
                    # print("makespan", makespan)
                    reward, done = self.env.step(makespan, self.Each_iteration_energy, self.tasks, waiting_time)

                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value.detach().cpu().numpy())
                    log_probs.append(log_prob.detach().cpu().numpy())
                    if len(schedulable_tasks) == 0:
                        self._forward_in_time() 
                    n_step += 1
                    print("n_step", n_step)
                    if done:
                        self.env.reset()
                        self.reward_log.append(reward)
                        # current_time = np.mean([self.evaluate() for _ in range(3 if self.noise > 0 else 1)])                        
                        # print("current time and best time: ", current_time, best_time)
                        # Create the models directory if it doesn't exist
                        models_dir = os.path.join(os.getcwd(), 'models')
                        os.makedirs(models_dir, exist_ok=True)
                        # os.makedirs('./models', exist_ok=True)
                        if makespan < best_time:
                            best_time = makespan
                            best_actor_path = os.path.join(models_dir, f'actor_best.pt')
                            best_critic_path = os.path.join(models_dir, f'critic_best.pt')
                            torch.save(self.actor.state_dict(), best_actor_path)
                            torch.save(self.critic.state_dict(), best_critic_path)
                            updations.append(i)
                            print(f"New best model saved with test time: {best_time}")


                    if done and iteration_counter == self.n_iterations:
                        iteration_counter = 0
                        # observations, actions, rewards, dones, values, log_probs = self.collect_trajectories(batch_size)
                        # next_value = self.network(observations[-1])[1].detach().cpu().numpy()[0]
                        next_value = self.critic(torch.FloatTensor(observations[-1])).detach().cpu().numpy()[0]
                        returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        for _ in range(self.num_epochs):
                            indices = np.random.permutation(len(observations))
                            for start in range(0, len(observations), self.num_minibatches):
                                end = start + self.num_minibatches
                                mb_indices = indices[start:end]
                                mb_observations = [observations[i] for i in mb_indices]
                                mb_actions = [actions[i] for i in mb_indices]
                                mb_log_probs = [log_probs[i] for i in mb_indices]
                                mb_returns = returns[mb_indices]
                                mb_advantages = advantages[mb_indices]

                                self.optimize_model(mb_observations, mb_actions, mb_log_probs, mb_returns, mb_advantages, step=n_step)
                        print("rewards...........", rewards)
                        # Reset the lists after optimization
                        observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

                        log_ratio += 1
                        # current_time = np.mean([self.evaluate() for _ in range(3 if self.noise > 0 else 1)])
                        
                        # print("current time and best time: ", current_time, best_time)
                        # # Create the models directory if it doesn't exist
                        # models_dir = os.path.join(os.getcwd(), 'models')
                        # os.makedirs(models_dir, exist_ok=True)
                        # # os.makedirs('./models', exist_ok=True)
                        # if current_time < best_time:
                        #     best_time = current_time
                        #     best_actor_path = os.path.join(models_dir, f'actor_best.pt')
                        #     best_critic_path = os.path.join(models_dir, f'critic_best.pt')
                        #     torch.save(self.actor.state_dict(), best_actor_path)
                        #     torch.save(self.critic.state_dict(), best_critic_path)
                        #     print(f"New best model saved with test time: {best_time}")


                        if self.reward_log:
                            end = time.time()
                            print(f'step {n_step}, reward: {np.mean(self.reward_log)}, FPS: {int(n_step / (end - start))}')

                        # if self.scheduler is not None:
                        #     self.scheduler.step(int(n_step/batch_size))
                 
            # self.entropy_coeff = self.current_entropy_coeff(num_of_iterations, i+1)  
            print("updations............", updations)          
            print("iteration_counter", iteration_counter)
            waiting_time_list.append(waiting_time_iteration)
            iteration_counter += 1
            task_completion_list.append(self.task_completion_list_iteration)   
            node_engaged.append(self.node_count_dict)             
            jobs = organize_tasks_by_job(self.task_completion_list_iteration)
            for jobID, job in jobs.items():                
                max_end_time = max(job, key=lambda x: x[4])
                #print("max_end_time", max_end_time)
                Make_Span = max_end_time[4] 
                     
            make_span_curve.append(Make_Span) 
            power_list.append(self.Each_iteration_energy) 
            # time_list.append(self.Each_iteration_time) 
            rounded_energy = round(self.Each_iteration_energy, 2)
            energy_makespan_list.append((Make_Span, rounded_energy)) 
    
        # print("waiting_time_list", waiting_time_list)
        if best_actor_path and best_critic_path:
            # self.actor.load_state_dict(torch.load(best_actor_path))
            # self.critic.load_state_dict(torch.load(best_critic_path))
            print(f"Loaded best model with test time: {best_time}")
            now = datetime.datetime.now() 
            # Optionally, you can rename the best model files
            os.rename(best_actor_path, os.path.join(models_dir, f'actor_final_{task_completion_list[0][0][5]}-PE({self.num_node*3}).pt'))
            # os.rename(best_actor_path, os.path.join(models_dir, f'actor_final.pt'))
            os.rename(best_critic_path, os.path.join(models_dir, f'critic_final_{task_completion_list[0][0][5]}-PE({self.num_node*3}).pt'))
            # os.rename(best_critic_path, os.path.join(models_dir, f'critic_final.pt'))

        return task_completion_list, make_span_curve, self.rejected_job, power_list, self.task_scheduled, node_engaged, energy_makespan_list                  
                    

            