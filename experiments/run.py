import simpy
import logging
import os
import pandas as pd
from mtdnetwork.component.time_network import TimeNetwork
from mtdnetwork.operation.mtd_operation import MTDOperation
from mtdnetwork.data.constants import ATTACKER_THRESHOLD, OS_TYPES
from mtdnetwork.component.adversary import Adversary
from mtdnetwork.operation.attack_operation import AttackOperation
from mtdnetwork.snapshot.snapshot_checkpoint import SnapshotCheckpoint
from mtdnetwork.statistic.evaluation import Evaluation
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.hosttopologyshuffle import HostTopologyShuffle
from mtdnetwork.mtd.portshuffle import PortShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity
from mtdnetwork.mtd.usershuffle import UserShuffle
from mtdnetwork.mtd.osdiversityassignment import OSDiversityAssignment
from mtdnetwork.mtdai.mtd_ai import create_network, update_target_model
from mtdnetwork.operation.mtd_ai_training import MTDAITraining
from mtdnetwork.operation.mtd_ai_operation import MTDAIOperation
from collections import deque
import random
import threading
import queue
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import keras
from keras.src.legacy.saving import legacy_h5_format
from mtdnetwork.statistic.security_metric_statistics import SecurityMetricStatistics
import numpy as np
# logging.basicConfig(format='%(message)s', level=logging.INFO)

mtd_strategies = [
    None,
    CompleteTopologyShuffle,
    # HostTopologyShuffle,
    IPShuffle,
    OSDiversity,
    # PortShuffle,
    # OSDiversityAssignment,
    ServiceDiversity,
    # UserShuffle
]



def save_evaluation_result(file_name, evaluations):
    current_directory = os.getcwd()
    if not os.path.exists(current_directory + '/experimental_data/results/' + file_name + '.csv'):
        pd.DataFrame(evaluations).to_csv('experimental_data/results/' + file_name + '.csv', index=False)
    else:
        pd.DataFrame(evaluations).to_csv('experimental_data/results/' + file_name + '.csv', mode='a', index=False,
                                         header=False)


def thread_function(start, end, result_queue, simulation_function, file_name=None, combination=None):
    results = []
    for i in range(start, end):
        result = simulation_function(file_name, combination)
        results.append(result)
    result_queue.put(results)


def execute_multithreading(simulation_function, iterations=10, num_threads=5, file_name=None, combination=None):
    # Define the range of the for loop
    start = 0
    end = iterations

    # Create a queue to hold the results
    result_queue = queue.Queue()

    # Create a list to store the threads
    threads = []

    # Define the number of threads you want to use
    num_threads = num_threads

    # Calculate the chunk size for each thread
    chunk_size = (end - start) // num_threads

    # Create the threads and assign them their respective chunks
    for i in range(num_threads):
        start_index = start + i * chunk_size
        end_index = start_index + chunk_size
        if i == num_threads - 1:
            end_index = end  # Make sure the last thread takes care of the remaining items
        thread = threading.Thread(target=thread_function, args=(start_index, end_index,
                                                                result_queue, simulation_function,
                                                                file_name, combination))
        threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all the threads to finish
    for thread in threads:
        thread.join()

    # Collect the results from the queue
    results = []
    while not result_queue.empty():
        results += result_queue.get()
    results_avg = construct_average_result(results)
    pd.DataFrame(results_avg).to_csv('experimental_data/results/' + file_name + '_avg.csv', index=False)
    return results_avg


def create_experiment_snapshots(network_size_list):
    snapshot_checkpoint = SnapshotCheckpoint()
    for size in network_size_list:
        time_network = TimeNetwork(total_nodes=size)
        adversary = Adversary(network=time_network, attack_threshold=ATTACKER_THRESHOLD)
        snapshot_checkpoint.save_snapshots_by_network_size(time_network, adversary)


def construct_average_result(results):
    results_avg = []
    for i in range(len(results[0])):
        mttc_i_list = [r[i]['time_to_compromise'] for r in results if r[i]['host_compromise_ratio'] != 0]
        if mttc_i_list:
            mttc_i = sum(mttc_i_list) / len(mttc_i_list)
            results_avg.append({
                'Name': results[0][i]['Name'],
                'Host Compromise ratio (compromised hosts / total hosts)': results[0][i]['host_compromise_ratio'],
                'MTD Interval': results[0][i]['mtd_interval'],
                'Network Size': results[0][i]['network_size'],
                'MTD Execution Frequency': sum([r[i]['MEF'] for r in results]) / len(results),
                'Attack Success Rate': sum([r[i]['ASR'] for r in results]) / len(results),
                'Mean Time to Compromise (s)': mttc_i,
            })
    return results_avg


def construct_experiment_result(name, mtd_interval, item, network_size):

    return {
        'Name': name,
        'mtd_interval': mtd_interval,
        'MEF': item['mtd_execution_frequency'],
        'ASR': item['attack_success_rate'],
        'time_to_compromise': item['time_to_compromise'],
        'host_compromise_ratio': item['host_compromise_ratio'],
        'network_size': network_size,
        'total_number_of_ports': item["total_number_of_ports"],
        'attack_path_exposure': item["attack_path_exposure"],
        'ROA': item['roa'],
        'risk': item['risk'],
        'shortest_path_variability': item['shortest_path_variability'],
        # 'Compromised Num': evaluation.compromised_num()
    }


# def single_mtd_simulation(file_name, combination):

def single_mtd_simulation(file_name, mtd_strategies, checkpoint= 'None', mtd_interval = [100, 200], network_size = [25, 50, 75, 100]):
    """
    Simulations for single mtd and no mtd
    """
    evaluations = []
    for mtd in mtd_strategies:
        mtd_evaluation = []
        if mtd is None:
            scheme = 'None'
            mtd_name = "NoMTD"
        else:
            mtd_name = mtd().get_name()
            scheme = 'single'
        for mtd_interval in mtd_interval:
            for network_size in network_size:
                evaluation = execute_simulation(scheme=scheme, mtd_interval=mtd_interval,
                                                custom_strategies=mtd, total_nodes=network_size)
                # if checkpoint != 'None':
                #     evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint(checkpoint)
                # else:
                #     evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint()
                evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint([0.05, 0.1, 0.15, 0.2, 0.25])
                for item in evaluation_results:
                    result = construct_experiment_result(mtd_name, mtd_interval, item, network_size)
                    evaluations.append(result)
                    mtd_evaluation.append(result)
        save_evaluation_result(file_name, mtd_evaluation)
        print(mtd_name)
    return evaluations

def mtd_ai_simulation(features, file_name,  model_path, start_time, finish_time, total_nodes, new_network, mtd_interval = [100, 200], network_size = [25, 50, 75, 100], custom_strategies = None, static_degrade_factor = 2000):
    """
    Simulations for single ai mtd
    """
    evaluations = []
   
    scheme = 'mtd_ai'
    # print(mtd_name, scheme)
    for mtd_interval in mtd_interval:
        for network_size in network_size:
             evaluation = execute_ai_model(
                 features = features,
                start_time=start_time,
                finish_time=finish_time,
                mtd_interval=mtd_interval,
                scheme= scheme,
                total_nodes=total_nodes,
                new_network=new_network,
                model_path=model_path,
                custom_strategies=custom_strategies,
                static_degrade_factor = static_degrade_factor
            )
            #  evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint(np.arange(0.01, 1.01, 0.01))
             evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint([0.05, 0.1, 0.15, 0.2, 0.25])
             for item in evaluation_results:
            
                result = construct_experiment_result('mtd_ai', mtd_interval, item, network_size)
        
                evaluations.append(result)
            

    save_evaluation_result(file_name, evaluations)
    print(scheme)
    return evaluations


def dap_mtd_simulation(file_name, combination):
    """
    Simulation for DAP MTD with different number of variants.
    """
    snapshot_checkpoint = SnapshotCheckpoint()
    os_types_list = [random.sample(OS_TYPES, 2), random.sample(OS_TYPES, 3), OS_TYPES]
    evaluations = []
    for os_types in os_types_list:
        mtd_evaluation = []
        for mtd_interval in [100, 200]:
            for network_size in [25, 50, 75, 100]:
                time_network, adversary = snapshot_checkpoint.load_snapshots_by_network_size(network_size)
                mtd = OSDiversityAssignment(network=time_network, os_types=os_types)
                evaluation = execute_simulation(scheme='single', mtd_interval=mtd_interval,
                                                custom_strategies=mtd, total_nodes=network_size)
                evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint()
                for item in evaluation_results:
                    result = construct_experiment_result(mtd.get_name(), mtd_interval, item, network_size)
                    evaluations.append(result)
                    mtd_evaluation.append(result)
        save_evaluation_result(file_name, mtd_evaluation)
        print(os_types)
    return evaluations


def multiple_mtd_simulation(file_name, combination):
    """
    simulations for multiple mtd using three different execution schemes.
    """
    evaluations = []

    for scheme in ['random', 'alternative', 'simultaneous']:
        mtd_evaluation = []
        for mtd_interval in [100, 200]:
            for network_size in [25, 50, 75, 100]:
                evaluation = execute_simulation(scheme=scheme, mtd_interval=mtd_interval,
                                                custom_strategies=combination, total_nodes=network_size)
                evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint()
                for item in evaluation_results:
                    result = construct_experiment_result(scheme, mtd_interval, item, network_size)
                    evaluations.append(result)
                    mtd_evaluation.append(result)
        save_evaluation_result(file_name, mtd_evaluation)
        print(scheme)
    return evaluations

def specific_multiple_mtd_simulation(file_name, combination, scheme, mtd_interval = [100, 200], network_size = [25, 50, 75, 100]):
    """
    simulations for multiple mtd using three different execution schemes.
    """
    evaluations = []
    mtd_evaluation = []
    for mtd_interval in mtd_interval:
        for network_size in network_size:
            evaluation = execute_simulation(scheme=scheme, mtd_interval=mtd_interval,
                                            custom_strategies=combination, total_nodes=network_size)
            # evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint()
            evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint([0.05, 0.1, 0.15, 0.2, 0.25])
            for item in evaluation_results:
                result = construct_experiment_result(scheme, mtd_interval, item, network_size)
                evaluations.append(result)
                mtd_evaluation.append(result)
    save_evaluation_result(file_name, mtd_evaluation)
    print(scheme)
    return evaluations



def execute_simulation(start_time=0, finish_time=None, scheme='random', mtd_interval=None, custom_strategies=None,
                       checkpoints=None, total_nodes=50, total_endpoints=5, total_subnets=8, total_layers=4,
                       target_layer=4, total_database=2, terminate_compromise_ratio=0.8, new_network=False):
    """

    :param start_time: the time to start the simulation, need to load timestamp-based snapshots if set start_time > 0
    :param finish_time: the time to finish the simulation. Set to None will run the simulation until
    the network reached compromised threshold (compromise ratio > 0.9)
    :param scheme: random, simultaneous, alternative, single, None
    :param mtd_interval: the time interval to trigger an MTD(s)
    :param custom_strategies: used for executing alternative scheme or single mtd strategy.
    :param checkpoints: a list of time value to save snapshots as the simulation runs.
    :param total_nodes: the number of nodes in the network (network size)
    :param total_endpoints: the number of exposed nodes
    :param total_subnets: the number of subnets (total_nodes - total_endpoints) / (total_subnets - 1) > 2
    :param total_layers: the number of layers in the network
    :param target_layer: the target layer in the network (for targetted attack scenario only)
    :param total_database: the number of database nodes used for computing DAP algorithm
    :param terminate_compromise_ratio: terminate the simulation if reached compromise ratio
    :param new_network: True: create new snapshots based on network size, False: load snapshots based on network size
    """
    # initialise the simulation
    env = simpy.Environment()
    end_event = env.event()
    snapshot_checkpoint = SnapshotCheckpoint(env=env, checkpoints=checkpoints)
    time_network = None
    adversary = None
    security_metrics_record = SecurityMetricStatistics()
    if start_time > 0:
        try:
            time_network, adversary = snapshot_checkpoint.load_snapshots_by_time(start_time)
        except FileNotFoundError:
            print('No timestamp-based snapshots available! Set start_time = 0 !')
            return
    elif not new_network:
        try:
            time_network, adversary = snapshot_checkpoint.load_snapshots_by_network_size(total_nodes)
        except FileNotFoundError:
            print('set new_network=True')
    else:
        time_network = TimeNetwork(total_nodes=total_nodes, total_endpoints=total_endpoints,
                                    total_subnets=total_subnets, total_layers=total_layers,
                                    target_layer=target_layer, total_database=total_database,
                                    terminate_compromise_ratio=terminate_compromise_ratio)
        adversary = Adversary(network=time_network,attack_threshold=ATTACKER_THRESHOLD)
        # snapshot_checkpoint.save_initialised(time_network, adversary)
        snapshot_checkpoint.save_snapshots_by_network_size(time_network, adversary)

    # start attack
    attack_operation = AttackOperation(env=env, end_event=end_event, adversary=adversary, proceed_time=0)
    attack_operation.proceed_attack()

    # start mtd
    if scheme != 'None':
        mtd_operation = MTDOperation(security_metrics_record=security_metrics_record,env=env, end_event=end_event, network=time_network, scheme=scheme,
                                        attack_operation=attack_operation, proceed_time=0,
                                        mtd_trigger_interval=mtd_interval, custom_strategies=custom_strategies, adversary=adversary)
        mtd_operation.proceed_mtd()
        security_metrics_record = mtd_operation.security_metric_record
    # print("metrics", mtd_operation.security_metric_record.get_record())

    # save snapshot by time
    if checkpoints is not None:
        snapshot_checkpoint.proceed_save(time_network, adversary)

    # start simulation
    if finish_time is not None:
        env.run(until=(finish_time - start_time))
    else:
        env.run(until=end_event)
    
    evaluation = Evaluation(network=time_network, adversary=adversary, security_metrics_record = security_metrics_record)

    # sim_item = scheme
    # if scheme == 'single':
    #     sim_item = custom_strategies().get_name()
    # elif scheme == 'None':
    #     sim_item = 'NoMTD'
    # time_network.get_mtd_stats().save_record(sim_time=mtd_interval, scheme=sim_item)
    # adversary.get_attack_stats().save_record(sim_time=mtd_interval, scheme=sim_item)

    return evaluation



def  execute_ai_training(features, start_time=0, finish_time=None, scheme='mtd_ai', mtd_interval=None, custom_strategies=None,
                       checkpoints=None, total_nodes=50, total_endpoints=5, total_subnets=8, total_layers=4,
                       target_layer=4, total_database=2, terminate_compromise_ratio=0.8, new_network=False,
                       state_size=3, action_size=5, time_series_size=3, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, train_start=1000, episodes=1000,
                       file_name=None, static_degrade_factor = 2000):
    """
    :param start_time: the time to start the simulation, need to load timestamp-based snapshots if set start_time > 0
    :param finish_time: the time to finish the simulation. Set to None will run the simulation until
    the network reached compromised threshold (compromise ratio > 0.9)
    :param scheme: random, simultaneous, alternative, single, None
    :param mtd_interval: the time interval to trigger an MTD(s)
    :param custom_strategies: used for executing alternative scheme or single mtd strategy.
    :param checkpoints: a list of time value to save snapshots as the simulation runs.
    :param total_nodes: the number of nodes in the network (network size)
    :param total_endpoints: the number of exposed nodes
    :param total_subnets: the number of subnets (total_nodes - total_endpoints) / (total_subnets - 1) > 2
    :param total_layers: the number of layers in the network
    :param target_layer: the target layer in the network (for targetted attack scenario only)
    :param total_database: the number of database nodes used for computing DAP algorithm
    :param terminate_compromise_ratio: terminate the simulation if reached compromise ratio
    :param new_network: True: create new snapshots based on network size, False: load snapshots based on network size
    :param state_size: the number of state variables
    :param action_size: the number of actions
    :param time_series_size: the number of time series variables
    :param gamma: discount factor
    :param epsilon: exploration rate
    :param epsilon_min: minimum exploration rate
    :param epsilon_decay: exploration rate decay
    :param batch_size: the size of batch for training
    :param train_start: the time to start training
    :param episodes: the number of episodes for training
    """

    # Create main and target models
    main_network = create_network(state_size, action_size, time_series_size)
    target_network = create_network(state_size, action_size, time_series_size)
    target_network.set_weights(main_network.get_weights())

    print("Static_factor", static_degrade_factor)
    print("MTD Scheme", custom_strategies)
    print("Action size(include zero which is no deployment)", action_size)

    memory = deque(maxlen=2000)
    security_metric_record = SecurityMetricStatistics()

    for episode in range(episodes):
        # initialise the simulation
        env = simpy.Environment()
        end_event = env.event()
        snapshot_checkpoint = SnapshotCheckpoint(env=env, checkpoints=checkpoints)
        time_network = None
        adversary = None

        if start_time > 0:
            try:
                time_network, adversary = snapshot_checkpoint.load_snapshots_by_time(start_time)
            except FileNotFoundError:
                print('No timestamp-based snapshots available! Set start_time = 0 !')
                return
        elif not new_network:
            try:
                time_network, adversary = snapshot_checkpoint.load_snapshots_by_network_size(total_nodes)
            except FileNotFoundError:
                print('set new_network=True')
        else:
            time_network = TimeNetwork(total_nodes=total_nodes, total_endpoints=total_endpoints,
                                    total_subnets=total_subnets, total_layers=total_layers,
                                    target_layer=target_layer, total_database=total_database,
                                    terminate_compromise_ratio=terminate_compromise_ratio)
            adversary = Adversary(network=time_network,attack_threshold=ATTACKER_THRESHOLD)
            # snapshot_checkpoint.save_initialised(time_network, adversary)
            snapshot_checkpoint.save_snapshots_by_network_size(time_network, adversary)



        # start attack
        attack_operation = AttackOperation(env=env, end_event=end_event, adversary=adversary, proceed_time=0)
        attack_operation.proceed_attack()

        # start mtd
        if scheme != 'None':
            mtd_operation = MTDAITraining(security_metric_record=security_metric_record,features = features, env=env, end_event=end_event, network=time_network, scheme=scheme,
                                        attack_operation=attack_operation, proceed_time=0,
                                        mtd_trigger_interval=mtd_interval, custom_strategies=custom_strategies, adversary=adversary,
                                        main_network=main_network, target_network=target_network, memory=memory, 
                                        gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, 
                                        batch_size=batch_size, train_start=train_start, static_degrade_factor = static_degrade_factor)
            mtd_operation.proceed_mtd()
            security_metric_record = mtd_operation.security_metric_record
        # save snapshot by time
        if checkpoints is not None:
            snapshot_checkpoint.proceed_save(time_network, adversary)

        # start simulation
        if finish_time is not None:
            env.run(until=(finish_time - start_time))
        else:
            env.run(until=end_event)

        if episode % 10 == 0:
            update_target_model(target_network, main_network)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # print(f"Episode: {episode}, Epsilon: {epsilon}")
    
    main_network.save(f'AI_model/main_network_{file_name}.h5')
    print("Training completed and model saved.")

# Define and register the custom mse function
@keras.saving.register_keras_serializable
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def  execute_ai_model(features, start_time=0, finish_time=None, scheme='mtd_ai', mtd_interval=None, custom_strategies=None,
                       checkpoints=None, total_nodes=50, total_endpoints=5, total_subnets=8, total_layers=4,
                       target_layer=4, total_database=2, terminate_compromise_ratio=0.8, new_network=False,
                       epsilon=1.0, attacker_sensitivity=1, model_path=None, static_degrade_factor = 2000):
    """
    :param start_time: the time to start the simulation, need to load timestamp-based snapshots if set start_time > 0
    :param finish_time: the time to finish the simulation. Set to None will run the simulation until
    the network reached compromised threshold (compromise ratio > 0.9)
    :param scheme: random, simultaneous, alternative, single, None
    :param mtd_interval: the time interval to trigger an MTD(s)
    :param custom_strategies: used for executing alternative scheme or single mtd strategy.
    :param checkpoints: a list of time value to save snapshots as the simulation runs.
    :param total_nodes: the number of nodes in the network (network size)
    :param total_endpoints: the number of exposed nodes
    :param total_subnets: the number of subnets (total_nodes - total_endpoints) / (total_subnets - 1) > 2
    :param total_layers: the number of layers in the network
    :param target_layer: the target layer in the network (for targetted attack scenario only)
    :param total_database: the number of database nodes used for computing DAP algorithm
    :param terminate_compromise_ratio: terminate the simulation if reached compromise ratio
    :param new_network: True: create new snapshots based on network size, False: load snapshots based on network size
    :param epsilon: exploration rate
    """

    custom_objects = {'mse': mse}

    
    try:
        main_network = legacy_h5_format.load_model_from_hdf5(model_path, custom_objects=custom_objects)  #For Mac
        print("On Mac")
    
    except:
        main_network = load_model(model_path, custom_objects=custom_objects)    #For Windows/Linux
        print("On Windows/Linux")

    main_network.compile(loss=MeanSquaredError(), optimizer=Adam())

    security_metrics_record = SecurityMetricStatistics()
    
    # initialise the simulation
    env = simpy.Environment()
    end_event = env.event()
    snapshot_checkpoint = SnapshotCheckpoint(env=env, checkpoints=checkpoints)
    time_network = None
    adversary = None

    print("Static_factor", static_degrade_factor)
    print("MTD Scheme", custom_strategies)
    print("Action size(include zero which is no deployment)", len(custom_strategies) + 1)

    if start_time > 0:
        try:
            time_network, adversary = snapshot_checkpoint.load_snapshots_by_time(start_time)
        except FileNotFoundError:
            print('No timestamp-based snapshots available! Set start_time = 0 !')
            return
    elif not new_network:
        try:
            time_network, adversary = snapshot_checkpoint.load_snapshots_by_network_size(total_nodes)
        except FileNotFoundError:
            print('set new_network=True')
    else:
        time_network = TimeNetwork(total_nodes=total_nodes, total_endpoints=total_endpoints,
                                total_subnets=total_subnets, total_layers=total_layers,
                                target_layer=target_layer, total_database=total_database,
                                terminate_compromise_ratio=terminate_compromise_ratio)
        adversary = Adversary(network=time_network,attack_threshold=ATTACKER_THRESHOLD)
        # snapshot_checkpoint.save_initialised(time_network, adversary)
        snapshot_checkpoint.save_snapshots_by_network_size(time_network, adversary)


 
    # start attack
    attack_operation = AttackOperation(env=env, end_event=end_event, adversary=adversary, proceed_time=0)
    attack_operation.proceed_attack()

    # start mtd
    if scheme != 'None':
        mtd_operation = MTDAIOperation(features, security_metrics_record, env=env, end_event=end_event, network=time_network, scheme=scheme,
                                    attack_operation=attack_operation, proceed_time=0,
                                    mtd_trigger_interval=mtd_interval, custom_strategies=custom_strategies, adversary=adversary,
                                    main_network=main_network, epsilon=epsilon, attacker_sensitivity=attacker_sensitivity, static_degrade_factor = static_degrade_factor)
        mtd_operation.proceed_mtd()
        security_metrics_record = mtd_operation.security_metrics_record

    # save snapshot by time
    if checkpoints is not None:
        snapshot_checkpoint.proceed_save(time_network, adversary)

    # start simulation
    if finish_time is not None:
        env.run(until=(finish_time - start_time))
    else:
        env.run(until=end_event)

    evaluation = Evaluation(network=time_network, adversary=adversary, security_metrics_record = security_metrics_record)
    return evaluation
        
    


#