import numpy as np
import configparser
config1 = configparser.ConfigParser()
config2 = configparser.ConfigParser()

config1.read('./environment.cfg')
config2.read('./agent.cfg')
import argparse
import logger, logging
from train import *
from test import *
from test import PPO_TEST
from utils import *
from environment import HPCTaskSchedulingEnv
from network import *
import os
import datetime


parser = argparse.ArgumentParser(description='Train the DQN model for task scheduling.')
parser.add_argument('-f', '--csv_file', type=str, help='Path to the CSV file containing task data.')
parser.add_argument('--num_node', type=int, help='Number of nodes in the environment.')
parser.add_argument('--num_jobs', type=int, help='Number of jobs to be schedule.', default=config1['General']['num_jobs'])
parser.add_argument('--num_cluster', type=int, help='Number of clusters in the environment.', default=config1['General']['num_cluster'])
parser.add_argument("-l", "--loglevel", help="The log level to be used in this module. Default: INFO", type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument('--nsga_time', type=int, help='Number of nodes in the environment.')
parser.add_argument("--showDAG", help="Switch used to enable display of the incoming task DAG", dest="showDAG", action="store_true")
parser.add_argument("--showGantt", help="Switch used to enable display of the final scheduled Gantt chart", dest="showGantt", action="store_true")
parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode to run the script in: train or test', required=True)

# Training settings
parser.add_argument('--input_dims', type=int, help='Input dimensions for the model.', default=...)  # Replace with default value
parser.add_argument('--n_actions', type=int, help='Number of actions available to the agent.', default=...)  # Replace with default value
parser.add_argument('--batch_size', type=int, help='Batch size for training.', default=64)
parser.add_argument('--gamma', type=float, help='Discount factor for future rewards.', default=0.99)
parser.add_argument('--lr', type=float, help='Learning rate for the model.', default=1e-3)


args = parser.parse_args()
num_node = args.num_node
csv_file = args.csv_file
num_cluster = args.num_cluster
nsga_time = args.nsga_time
mode = args.mode
train_type = "makespan"
actor_filename = config2['Hyperparameters']['actor_filename']
# actor_model = "C:/Users/Amol/DQN CDAC/Optimizations algorithms/models/actor_final_Epigenomes(544)-PE(180).pt"
# critic_model = "C:/Users/Amol/DQN CDAC/Optimizations algorithms/models/critic_final_Epigenomes(544)-PE(180).pt"
config_enhanced = vars(args)

def train(num_cluster, num_node, csv_file, train_type, nsga_time):
    """
    This function is used to train the PPO model for task scheduling in a HPC environment.

    Parameters:
    num_cluster (int): The number of clusters in the HPC environment.
    num_node (int): The number of nodes in each cluster.
    csv_file (str): The path to the CSV file containing the task information.
    train_type (str): The type of training to be performed (e.g., 'makespan', 'energy').
    nsga_time (int): The time for NSGA algorithm.
    actor_model (str): The path to the actor model file.
    critic_model (str): The path to the critic model file.

    Returns:
    None
    """
    env = HPCTaskSchedulingEnv(num_cluster, num_node, csv_file, train_type, nsga_time)
    model = PPO(policy_class=FeedForwardNN, env = env, num_node=num_node)
    # if actor_model != '' and critic_model != '':
    #     model.actor.load_state_dict(torch.load(actor_model))
    #     model.critic.load_state_dict(torch.load(critic_model))
    #     print("Successfully loaded pre-trained models.")
    # else:
    #     print("No pre-trained models found. Training from scratch.")
    task_completion_list, make_span_curve, rejected_job, power_list, task_scheduled, node_engaged, energy_makespan_list = model.training_batch()
    min_index_makespan = make_span_curve.index(min(make_span_curve))
    min_index_power = power_list.index(min(power_list))
    count_makespan = 0
    for list_ in node_engaged[min_index_makespan].values():
        for num in list_:
            count_makespan += 1

    count_energy = 0
    for list_ in node_engaged[min_index_power].values():
        for num in list_:
            count_energy += 1        

    now = datetime.datetime.now() 
    with open(f"{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_Optimal_task_completion_list.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)    
        # Write the header
        header = ['Task_ID', 'Cluster_ID', 'Node_ID', 'Start_Time', 'End_Time', 'Task_Name', 'Resource_Type']
        csv_writer.writerow(header)    
        # Write the data
        csv_writer.writerows(task_completion_list[min_index_makespan])
        csv_writer.writerow([])
        csv_writer.writerow([])
        csv_writer.writerow(['#summary'])
        csv_writer.writerow(['#Node_engaged for optimal iteration:'])
        csv_writer.writerow([node_engaged[min_index_makespan]])
        csv_writer.writerow(['#Node_engaged count:'])
        csv_writer.writerow([count_makespan])
        csv_writer.writerow(['#Optimal ite make span:'])
        csv_writer.writerow([make_span_curve[min_index_makespan]])
        csv_writer.writerow(['#Optimal ite energy:'])
        csv_writer.writerow([power_list[min_index_makespan]])

    # now = datetime.datetime.now()
    # # output_dir = f"Out/DQN/CSV/last"
    # # if not os.path.exists(os.path.dirname(output_dir)):
    # #     os.makedirs(os.path.dirname(output_dir))  
    # #csv_file_path = f'output_dir'
    # # Write the list to the CSV file
    # with open(f"{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_last_iter_task_completion_list.csv", 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)    
    #     # Write the header
    #     header = ['Task_ID', 'Cluster_ID', 'Node_ID', 'Start_Time', 'End_Time', 'Task_Name', 'Resource_Type']
    #     csv_writer.writerow(header)    
    #     # Write the data
    #     csv_writer.writerows(task_completion_list[int(config2['Hyperparameters']['no_of_iterations'])-1])   
    #     csv_writer.writerow([])
    #     csv_writer.writerow([])
    #     csv_writer.writerow(['#summary'])
    #     csv_writer.writerow(['#Node_engaged for last iteration:'])
    #     csv_writer.writerow([node_engaged[min_index_power]])
    #     csv_writer.writerow(['#Node_engaged count:'])
    #     csv_writer.writerow([count_energy])
    #     csv_writer.writerow(['#last ite make span:'])
    #     csv_writer.writerow([make_span_curve[min_index_power]])
    #     csv_writer.writerow(['#last ite energy:'])
    #     csv_writer.writerow([power_list[min_index_power]])
        

    # now = datetime.datetime.now()
        #output_dir = f"Out/Pareto_data"
        # if not os.path.exists(os.path.dirname(output_dir)):
        #     os.makedirs(os.path.dirname(output_dir))  
        #csv_file_path = f'output_dir'
        # Write the list to the CSV file
    with open(f"{task_completion_list[0][0][5]}-PE({num_node*3}).csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)    
        # Write the header
        header = ['Makespan', 'Energy']
        csv_writer.writerow(header)    
        # Write the data
        csv_writer.writerows(energy_makespan_list)


    # plot_learning_curve(power_list, "Energy")
    # plot_learning_curve(make_span_curve, "Make Span")


    # plot_gantt_chart_per_cluster_and_type(task_completion_list[min_index_makespan], "optimal")
    # plot_gantt_chart_per_job(task_completion_list[min_index_makespan], "optimal") 


    print("make_span_curve_list", make_span_curve)
    print("MakeSpan for  optimal makespan iteration is :", make_span_curve[min_index_makespan])
    print("Energy for optimal makespan iteration is :", power_list[min_index_makespan])
    print("Node_engaged for optimal makespan iteration", node_engaged[min_index_makespan])
    print("count_makespan", count_makespan)
    print("MakeSpan for optimal energy iteration is :", make_span_curve[min_index_power])    
    print("Energy for optimal energy iteration is :", power_list[min_index_power])   
    print("Node_engaged for optimal energy iteration", node_engaged[min_index_power])   
    print("count_energy", count_energy)



def test(num_cluster, num_node, csv_file, train_type, nsga_time):
    """
    This function is used to test the PPO model for task scheduling in a HPC environment.

    Parameters:
    num_cluster (int): The number of clusters in the HPC environment.
    num_node (int): The number of nodes in each cluster.
    csv_file (str): The path to the CSV file containing the task information.
    train_type (str): The type of training to be performed (e.g., 'makespan', 'energy').
    nsga_time (int): The time for NSGA algorithm.

    Returns:
    None
    """
    env = HPCTaskSchedulingEnv(num_cluster, num_node, csv_file, train_type, nsga_time)
    actor_model = os.path.join('./models', actor_filename)
    model = PPO_TEST(actor_model, policy_class=FeedForwardNN, env = env)

    task_completion_list, Make_Span, rounded_energy, power_list, node_engaged = model.evaluate_test()
    count = 0
    for list_ in node_engaged.values():
        for num in list_:
            count += 1     

    now = datetime.datetime.now() 
    with open(f"{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_Optimal_task_completion_list.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)    
        # Write the header
        header = ['Task_ID', 'Cluster_ID', 'Node_ID', 'Start_Time', 'End_Time', 'Task_Name', 'Resource_Type']
        csv_writer.writerow(header)    
        # Write the data
        csv_writer.writerows(task_completion_list)
        csv_writer.writerow([])
        csv_writer.writerow([])
        csv_writer.writerow(['#summary'])
        csv_writer.writerow(['#Node_engaged for optimal iteration:'])
        csv_writer.writerow([node_engaged])
        csv_writer.writerow(['#Node_engaged count:'])
        csv_writer.writerow([count])
        csv_writer.writerow(['#Optimal ite make span:'])
        csv_writer.writerow([Make_Span])
        csv_writer.writerow(['#Optimal ite energy:'])
        csv_writer.writerow([rounded_energy])
        
    # plot_gantt_chart_per_cluster_and_type(task_completion_list[min_index_makespan], "optimal")
    plot_gantt_chart_per_job(task_completion_list, "optimal") 

    print("MakeSpan for optimal iteration is :", Make_Span)
    print("Energy for optimal iteration is :", rounded_energy)
    print("Node_engaged for optimal iteration", node_engaged)
    print("total_node_count", count)    


def main(num_cluster, num_node, csv_file, train_type, nsga_time):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# Train or test, depending on the mode specified
	if mode == 'train':
		train(num_cluster, num_node, csv_file, train_type, nsga_time)
	else:
		test(num_cluster, num_node, csv_file, train_type, nsga_time)

if __name__ == '__main__':
	main(num_cluster, num_node, csv_file, train_type, nsga_time)    