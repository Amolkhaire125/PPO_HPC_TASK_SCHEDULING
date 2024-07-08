import logging
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.cm import viridis
import networkx as nx
import datetime
import os

class Task(object):
    # Task details initializing
    def __init__(self, jobID, taskID, CPU, RAM, disk, parent_task, Runtime_C1, Runtime_C2, Runtime_C3, deadline, task_type, status):
        self.parent = []
        self.child = []
        self.jobID = jobID
        self.taskID = taskID
        self.CPU = CPU
        self.RAM = RAM
        self.disk = disk
        self.parent_task = parent_task
        self.status = status               #-1: rejected, 0: finished, 1: ready, 2: running, 3:waiting 
        self.Runtime_C1 = Runtime_C1
        self.Runtime_C2 = Runtime_C2
        self.Runtime_C3 = Runtime_C3
        self.ddl =  deadline
        self.task_type = task_type
        #self.ddl = self.runtime + time.time() + random.randint(1, 100)*1000
        self.start_time = 0  # Initialize start time
        self.end_time = 0
        self.num_task = 0


def readCSV(fname):
    # Reading csv file
    task = []    
    Network_flow = []
    task_dict = {}
    with open(fname, 'r')as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            jobID, taskID, CPU, RAM, disk, parent_task, Runtime_C1, Runtime_C2, Runtime_C3, deadline,  task_type = map(str, line)
            #logging.error("Any task has incomplete information")
            # Handle the case when parent_task is a string representation of a tuple
            if parent_task.startswith("[") and parent_task.endswith("]"):
                parent_tasks = [int(t) for t in parent_task[1:-1].split(',')]
            else:
                parent_tasks = [int(parent_task)]
            #parent_tasks = [int(t) for t in parent_task.split(',')]
            task_obj = Task(str(jobID), int(taskID), float(CPU), float(RAM), disk, parent_tasks, float(Runtime_C1), float(Runtime_C2), float(Runtime_C3), float(deadline), str(task_type), 1)
            task_dict[taskID] = task_obj
            task.append(task_obj)
    #print(Network_flow)
    
    return task


def organize_tasks_by_job2(tasks):
    # organize task accoording to job
    jobs = {}
    for task in tasks:
        if task.jobID not in jobs:
            jobs[task.jobID] = []
        jobs[task.jobID].append(task)
    print("organise tasks by job", jobs)    
    return jobs


def plot_network_flow_for_each_job(tasks):
    # plot network flow for each job separately
    jobs = organize_tasks_by_job2(tasks)
    # Create a PdfPages object to save all plots in a single PDF file
    output_dir = f"Out/DQN/DAG"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now() 
    #with PdfPages(f'{output_dir}/DAG_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.pdf') as pdf:
    for job_id, tasks in jobs.items():
        with PdfPages(f"{output_dir}/DAG_{job_id}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.pdf") as pdf:
            now = datetime.datetime.now()  
            filename = f"{output_dir}/DAG_{job_id}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.pdf"
            Network_flow = []
            for task in tasks:
                for parent in task.parent_task:
                    if parent != 0:
                        Network_flow.append((parent, task.taskID))
            plt.title(f'Network Flow: {job_id}')
            print(f"\nNetwork_flow is as follows: {Network_flow}") 
            G = nx.DiGraph()
            edge_length = 1.0
            task_types = {task.taskID: task.task_type for task in tasks}      
            unique_task_types = list(set(task_types.values()))       
            color_map = {task_type: plt.cm.get_cmap('jet')(i / len(unique_task_types))
                        for i, task_type in enumerate(unique_task_types)}
            
            G.add_nodes_from(set(node for edge in Network_flow for node in edge))  
            G.add_edges_from(Network_flow, length=edge_length)   
            # Adjust node positions to ensure a minimum distance
            node_size = 600       
            # Modify the line causing the ValueError
            node_colors = [color_map.get(task_types.get(node, 'default_color')) for node in G.nodes]
            #if show_dag:
            nx.draw(G, pos=nx.nx_pydot.graphviz_layout(G, prog='dot'), with_labels=True, node_color=node_colors, node_size=node_size)

            legend_labels = {task_type: f"{task_type}" for task_type in unique_task_types}
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                            for color in color_map.values()]

            plt.legend(handles=legend_handles, labels=list(legend_labels.values()), title='Task Types',
                    loc='upper left', bbox_to_anchor=(1, 1))
            plt.axis('off') 
            pdf.savefig()
            #pdf.savefig(filename)
            #plt.savefig(f'Network Flow for Job {job_id}.pdf')
            plt.show()
            #plt.close()      # Close the current figure to free up resources 


def plot_gantt_chart_per_cluster_and_type(task_completion_list, cat):       
    cluster_tasks = organize_tasks_by_cluster_and_type(task_completion_list) 
    # Create a PdfPages object to save all plots in a single PDF file
    max_end_time = max(task_completion_list, key=lambda x: x[4])
    #print("max_end_time", max_end_time)
    Make_Span = max_end_time[4]
    if cat == "optimal":
        output_dir = f"Out/DQN/Cluster/optimal"
    elif cat == "last":
        output_dir = f"Out/DQN/Cluster/last"    
    # os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    now = datetime.datetime.now() 
    with PdfPages(f'{output_dir}/{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_schedule-cluster.pdf') as pdf: 
        for cluster_id, job_types_data in cluster_tasks.items():
            tasks_data = job_types_data
            fig, ax = plt.subplots(figsize=(10, 5))
            y_ticks = []
            y_ticklabels = []
            y = y_initial =  0
            # Sort tasks based on start time
            sorted_tasks_data = {k: sorted(v, key=lambda x: x[0]) for k, v in tasks_data.items()}
            sorted_tasks_data = dict(sorted(tasks_data.items()))
            for node_id, tasks in sorted_tasks_data.items():
                sorted_tasks = sorted(tasks, key=lambda x: x[3])  # Sort tasks based on start time
                #print("sorted_tasks", sorted_tasks)
                y_max = 0
                #print(sorted_tasks)
                for i, task_info in enumerate(sorted_tasks):
                    task_name, _, _, start_time, end_time, jobID, _ = task_info
                    task_width = end_time - start_time
                    task_height = 1.0
                    if len(sorted_tasks) > 1 and i != 0:
                        if sorted_tasks[i][3] < sorted_tasks[i - 1][4] :
                            y += 1
                            y_max += 1
                        else:
                            y = y_initial   

                    ax.broken_barh([(start_time, task_width)], (y, task_height),
                                label=f'T{task_name} ')
                    ax.text(start_time + (end_time - start_time) / 2, y + 0.4,
                        f'T{task_name}', ha='center', va='center', color='black')
                
                y_ticks.append(y_initial)  # Adjusted for better placement of node labels
                y_ticklabels.append(f'Node {node_id}')
                y_initial += y_max + 2
                y = y_initial
            #print("Y_ticks", y_ticks)    
            #max_y = max(node_id + len(tasks) - server_offsets[node_id] * 0.5 for node_id, tasks in nodes_data.items())
            ax.axvline(Make_Span, color='red', linestyle='--', label=f'Total Makespan: {Make_Span} units')
            ax.text(Make_Span, len(tasks_data) + 1,f'Total Makespan: {Make_Span} units', ha='right', va='top', color='red')

            ax.set_xlabel('Time')
            ax.set_ylabel('Node')
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            plt.title(f'Tasks Schedule on C{cluster_id}')
            #fig.savefig()
            pdf.savefig()
            #plt.savefig(f"Scheduling chart for {cluster_id} - {job_type}.pdf")
            if cat == "optimal":
                plt.show()
            # else:
            #     plt.close()      


def plot_gantt_chart_per_job(task_completion_list, cat):
    """
    Plot task completion schedule for each job.

    Parameters:
    task_completion_list (list): A list of tuples representing task completion information.
    cat (str): A string indicating the category of the task completion list.

    Returns:
    None
    """

    jobs = organize_tasks_by_job(task_completion_list)
    for jobID, job in jobs.items():
        task_types = {task[0]: task[6] for task in job}
        #print("task_types", task_types)
        unique_task_types = list(set(task_types.values()))
        #unique_task_types = list(set(task['task_type'] for task in tasks if isinstance(task, dict)))
        #print("unique_task_types", unique_task_types)
        color_map = {task_type: plt.cm.get_cmap('jet')(i / len(unique_task_types))
                                for i, task_type in enumerate(unique_task_types)}
        #print("color_map", color_map)
        cluster_tasks = organize_tasks_by_cluster_and_type(job) 
        max_end_time = max(job, key=lambda x: x[4])
        #print("max_end_time", max_end_time)
        Make_Span = max_end_time[4]
        #print(f"makespan for job - {jobID} is {Make_Span}")
        if cat == "optimal":
            output_dir = f"Out/DQN/Schedule/optimal"
        elif cat == "last":
            output_dir = f"Out/DQN/Schedule/last"    
        # os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now() 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create a PdfPages object to save all plots in a single PDF file    
        with PdfPages(f'{output_dir}/{jobID}schedule-{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.pdf') as pdf: 
            for cluster_id, job_types_data in cluster_tasks.items():
                #print("job_types_data", job_types_data)                    
                # Check if the key is present in job_types_data
                # Extract the relevant data for the current task type
                tasks_data = job_types_data
                #print("tasks_data", tasks_data)
                fig, ax = plt.subplots(figsize=(10, 5))
                y_ticks = []
                y_ticklabels = []
                y = y_initial =  0
                # Sort tasks based on start time
                sorted_tasks_data = {k: sorted(v, key=lambda x: x[0]) for k, v in tasks_data.items()}
                sorted_tasks_data = dict(sorted(tasks_data.items()))
                #sorted_tasks_data = dict(sorted(sorted_tasks_data.items(), reverse=True))
                #sorted_tasks_data = {node_id: sorted(tasks, key=lambda x: x[3]) for node_id, tasks in tasks_data.items()}
                #print("sorted_tasks_data", sorted_tasks_data)
                for node_id, tasks in sorted_tasks_data.items():
                    sorted_tasks = sorted(tasks, key=lambda x: x[3])  # Sort tasks based on start time
                    #print("sorted_tasks", sorted_tasks)
                    y_max = 0
                    for i, task_info in enumerate(sorted_tasks):
                        task_name, _, _, start_time, end_time, jobID, task_type = task_info
                        task_width = end_time - start_time
                        task_height = 1.0
                        if len(sorted_tasks) > 1 and i != 0:
                            if sorted_tasks[i][3] < sorted_tasks[i - 1][4] :
                                y += 1
                                y_max += 1
                            else:
                                y = y_initial   

                        # Use color_map to get the color based on task_type
                        color = color_map.get(task_type, 'gray')
                        ax.broken_barh([(start_time, task_width)], (y, task_height), facecolors=color,
                                label=f'T{task_name}')
                        ax.text(start_time + (end_time - start_time) / 2, y + 0.4,
                            f'T{task_name}', ha='center', va='center', color='black')
                
                    y_ticks.append(y_initial)  # Adjusted for better placement of node labels
                    y_ticklabels.append(f'Node {node_id}')
                    y_initial += y_max + 2
                    y = y_initial
                #print("Y_ticks", y_ticks)    
                #max_y = max(node_id + len(tasks) - server_offsets[node_id] * 0.5 for node_id, tasks in nodes_data.items())
                ax.axvline(Make_Span, color='red', linestyle='--', label=f'Total Makespan: {Make_Span} units')
                ax.text(Make_Span, len(tasks_data) + 1,f'Total Makespan: {Make_Span} units', ha='right', va='top', color='red')

                ax.set_xlabel('Time')
                ax.set_ylabel('Node')
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_ticklabels)
                #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                handles, labels = ax.get_legend_handles_labels()
                #print("labels", labels)
                #legend_labels = [f'{label} ({task_types[int(label[1:])][1]})' for label in labels]
                legend_labels = {task_type: f"{task_type} ({color_map[task_type]})" for task_type in unique_task_types}
                legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                                for color in color_map.values()]
                ax.legend(legend_handles, legend_labels, title='Tasks', loc='upper left', bbox_to_anchor=(1, 0.5))
                #ax.legend(handles, labels, title='Tasks', loc='center left', bbox_to_anchor=(1, 0.5))
                ax.grid(True)
                plt.title(f'Tasks schedule: {jobID} on C{cluster_id}')
                #plt.title(f' job {jobID} ', loc='left', fontsize=12)
                pdf.savefig()
                #plt.savefig(f"Scheduling chart for {cluster_id} - {job_type}.pdf")
                if cat == "optimal":
                    plt.show()
                # else:
                #     plt.close()    



def plot_learning_curve(episode_rewards, r_type):
    # Plot the learning curve
    output_dir = f"Out/DQN/L-curve"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    now = datetime.datetime.now() 
    plt.plot(episode_rewards)
    plt.xlabel('Epoch')
    plt.ylabel(r_type)
    plt.title('Learning Curve')
    plt.savefig(f'{output_dir}/{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_{r_type}_LC.pdf')
    plt.show()


def organize_tasks_by_job(task_completion_list):
    """
    Organizes tasks based on their job ID.

    Parameters:
    task_completion_list (list): A list of task completion information. Each element in the list is a tuple representing task completion details.

    Returns:
    dict: A dictionary where the keys are job IDs and the values are lists of tasks belonging to each job.
    """
   
    jobs = {}
    for task in task_completion_list:
        job_id = task.jobID if hasattr(task, 'jobID') else task[5]  # Adjust the index based on the position of 'jobID' in the tuple
        if job_id not in jobs:
            jobs[job_id] = []
        jobs[job_id].append(task)
    return jobs

def organize_tasks_by_cluster(task_completion_list):
    # organize task according to cluster
    cluster_tasks = {}
    for task_info in task_completion_list:
        taskID, C_plus_1, Node_id, start_time, end_time,taskID, job_type = task_info
    # Check if the cluster exists in the dictionary, if not, create a new dictionary for the cluster
        if C_plus_1 not in cluster_tasks:
            cluster_tasks[C_plus_1] = {}
    # Check if the node exists in the cluster dictionary, if not, create a new list for the node
        if Node_id not in cluster_tasks[C_plus_1]:
            cluster_tasks[C_plus_1][Node_id] = []
    # Append the task information to the respective node's list
        cluster_tasks[C_plus_1][Node_id].append(task_info)
    return cluster_tasks        


def organize_tasks_by_cluster_and_type(task_completion_list):
    # organize task according to cluster and task type
    cluster_tasks = {}
    #print("task_completion_list", task_completion_list)
    for task_info in task_completion_list:
        taskID, C_plus_1, Node_id, start_time, end_time, jobID, task_type = task_info
    # Check if the cluster exists in the dictionary, if not, create a new dictionary for the cluster
        if C_plus_1 not in cluster_tasks:
            cluster_tasks[C_plus_1] = {}
    # Check if the node exists in the cluster and task type dictionary, if not, create a new list for the node
        if Node_id not in cluster_tasks[C_plus_1]:
            cluster_tasks[C_plus_1][Node_id] = []
    # Append the task information to the respective node's list
        cluster_tasks[C_plus_1][Node_id].append(task_info)
    #print("cluster_tasks", cluster_tasks)
    return cluster_tasks