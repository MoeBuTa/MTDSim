import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import os
import seaborn as sns
directory = os.getcwd()


class Evaluation:
    def __init__(self, network, adversary, features, security_metrics_record):

        self._network = network
        self._adversary = adversary
        self._mtd_record = network.get_mtd_stats().get_record()
        self._attack_record = adversary.get_attack_stats().get_record()
        self.features = "#".join(features)
        self.security_metrics_record = security_metrics_record
        self.create_directories()

    def create_directories(self):
        os.makedirs(directory + '/experimental_data/plots/' + self.features, exist_ok=True)
        return directory
    

    def compromised_num(self, record=None):
        if record is None:
            record = self._attack_record
        if 'compromise_host_uuid' in record.columns:
            compromised_hosts = record[record['compromise_host_uuid'] != 'None']['compromise_host_uuid'].unique()
            return len(compromised_hosts)
        else:
            return 0

    def mean_time_to_compromise_10_timestamp(self, run_index = 1):
        interval = 10
        record = self._attack_record
        step = max(record['finish_time']) / interval
        MTTC = []
        for i in range(1, interval + 1, 1):
            compromised_num = self.compromised_num_by_timestamp(step * i)
            if compromised_num == 0:
                MTTC.append({f'Mean Time to Compromise_{run_index}': None, 'Time': step * i})
                continue
            attack_action_time = record[(record['name'].isin(['SCAN_PORT', 'EXPLOIT_VULN', 'BRUTE_FORCE'])) &
                                        (record['finish_time'] <= step * i)]['duration'].sum()
            MTTC.append({f'Mean Time to Compromise_{run_index}': attack_action_time / compromised_num, f'Time_{run_index}': step * i})
    
        return MTTC
    
    # def mean_time_to_compromise_intervals(self, time_intervals):
    #     record = self._attack_record
    #     MTTC = []

    #     for time in time_intervals:
    #         compromised_num = self.compromised_num_by_timestamp(time)
    #         if compromised_num == 0:
    #             continue
    #         attack_action_time = record[(record['name'].isin(['SCAN_PORT', 'EXPLOIT_VULN', 'BRUTE_FORCE'])) &
    #                                     (record['finish_time'] <= time)]['duration'].sum()
    #         MTTC.append({'
    # ': attack_action_time / compromised_num, 'Time': time})
        
    #     return MTTC
    
    def compromised_num_by_timestamp(self, timestamp):
        record = self._attack_record
        compromised_hosts = record[(record['compromise_host_uuid'] != 'None') &
                                   (record['finish_time'] <= timestamp)]['compromise_host_uuid'].unique()
        return len(compromised_hosts)

    def mtd_execution_frequency(self):
        """
        The frequency of executing MTDs
        :return: Total number of executed MTD / Elapsed time
        """
        if len(self._mtd_record) == 0:
            return 0
        record = self._mtd_record
        return len(record) / (record.iloc[-1]['finish_time'] - record.iloc[0]['start_time'])

    def evaluation_result_by_compromise_checkpoint(self, checkpoint=None):
        """
        The time to reach compromise checkpoints

        ATTACK_ACTION: SCAN_PORT, EXPLOIT_VULN, BRUTE_FORCE
        Attack action time = The sum of the time duration of all ATTACK_ACTION
        """
        if checkpoint is None:
            checkpoint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        record = self._attack_record
        host_num = self.get_network().get_total_nodes()
        result = []
        comp_nums = {}
        for comp_ratio in checkpoint:
            comp_nums[comp_ratio] = host_num * comp_ratio

        for comp_ratio, comp_num in comp_nums.items():
            if 'cumulative_compromised_hosts' not in record.columns:
                return []

            if max(record['cumulative_compromised_hosts']) < comp_num:
                break
            sub_record = record[record['cumulative_compromised_hosts'] <= comp_num]
            time_to_compromise = sub_record[sub_record[
                'name'].isin(['SCAN_PORT', 'EXPLOIT_VULN', 'BRUTE_FORCE'])]['duration'].sum()
            attempt_hosts = sub_record[sub_record['current_host_uuid'] != -1]['current_host_uuid'].unique()
            attack_actions = sub_record[sub_record['name'].isin(['SCAN_PORT', 'EXPLOIT_VULN', 'BRUTE_FORCE'])]
            attack_event_num = 0
            for host in attempt_hosts:
                attack_event_num += len(attack_actions[(attack_actions['current_host_uuid'] == host) &
                                                       (attack_actions['name'] == 'SCAN_PORT')])
            attack_success_rate = comp_num / attack_event_num

            mtd_execution_frequency = self.mtd_execution_frequency()

            result.append({'time_to_compromise': time_to_compromise,
                           'attack_success_rate': attack_success_rate,
                           'host_compromise_ratio': comp_ratio,
                           'mtd_execution_frequency': mtd_execution_frequency})
        return result

    def compromise_record_by_attack_action(self, action=None):
        """
        :return: a list of attack record that contains hosts compromised by the given action
        """
        record = self._attack_record
        if action is None:
            return record[record['compromise_host'] != 'None']
        return record[(record['name'] == action) &
                      (record['compromise_host'] != 'None')]

    def get_metrics(self, env):
        # State metrics

        compromised_num = self.compromised_num()
        host_compromise_ratio = compromised_num/len(self._network.get_hosts()) \

        exposed_endpoints = len(self._network.get_exposed_endpoints())

        attack_path_exposure = self._network.attack_path_exposure()

        attack_stats = self._adversary.get_network().get_scorer().get_statistics()
        risk = attack_stats['Vulnerabilities Exploited']['risk'][-1] if attack_stats['Vulnerabilities Exploited']['risk'] else 0
        roa = attack_stats['Vulnerabilities Exploited']['roa'][-1] if attack_stats['Vulnerabilities Exploited']['roa'] else 0

        shortest_paths = self._network.scorer.shortest_path_record 
        shortest_path_variability = (len(shortest_paths[-1]) - len(shortest_paths[-2]))/len(shortest_paths) if len(shortest_paths) > 1 else 0

        evaluation_results = self.evaluation_result_by_compromise_checkpoint(np.arange(0.01, 1.01, 0.01))
        if evaluation_results:
            total_asr, total_time_to_compromise, total_compromises = 0, 0, 0

            for result in evaluation_results:
                if result['host_compromise_ratio'] != 0:  
                    total_time_to_compromise += result['time_to_compromise']
                    total_compromises += 1
                total_asr += result['attack_success_rate']

            overall_asr_avg = total_asr / len(evaluation_results) if evaluation_results else 0
            overall_mttc_avg = total_time_to_compromise / total_compromises if total_compromises else 0
        else:
            overall_asr_avg = 0
            overall_mttc_avg = 0


        # Time-series metrics
        time_since_last_mtd = env.now - self._network.last_mtd_triggered_time
        mtd_freq = self.mtd_execution_frequency()

        state_array = [host_compromise_ratio, exposed_endpoints, attack_path_exposure, overall_asr_avg, roa, shortest_path_variability, risk]
 

        time_series_array = [mtd_freq, overall_mttc_avg, time_since_last_mtd]
 
        return [state_array, time_series_array]


    def draw_network(self):
        """
        Draws the topology of the network while also highlighting compromised and exposed endpoint nodes.
        """
        plt.figure(1, figsize=(15, 12))
        nx.draw(self._network.graph, pos=self._network.pos, node_color=self._network.colour_map, with_labels=True)
        plt.savefig(directory + '/experimental_data/plots/' + self.features + '/network.png')
        plt.show()

    def draw_hacker_visible(self):
        """
        Draws the network that is visible for the hacker
        """
        subgraph = self._network.get_hacker_visible_graph()
        plt.figure(1, figsize=(15, 12))
        nx.draw(subgraph, pos=self._network.pos, with_labels=True)
        plt.show()

    def draw_compromised(self):
        """
        Draws the network of compromised hosts
        """
        compromised_hosts = self._adversary.get_compromised_hosts()
        subgraph = self._network.graph.subgraph(compromised_hosts)
        colour_map = []
        c_hosts = sorted(compromised_hosts)
        for node_id in c_hosts:
            if node_id in self._network.exposed_endpoints:
                colour_map.append("green")
            else:
                colour_map.append("red")

        plt.figure(1, figsize=(15, 12))
        nx.draw(subgraph, pos=self._network.pos, node_color=colour_map, with_labels=True)
        plt.show()

    def visualise_attack_operation_group_by_host(self):
        """
        visualise the action flow of attack operation group by host ids.
        """
        record = self._attack_record

        fig, ax = plt.subplots(1, figsize=(16, 5))
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
        attack_action_legend = []
        attack_action_legend_name = []
        for i, v in enumerate(record['name'].unique()):
            record.loc[record.name == v, 'color'] = colors[i]
            attack_action_legend.append(Line2D([0], [0], color=colors[i], lw=4))
            attack_action_legend_name.append(v)

        hosts = record['current_host_uuid'].unique()
        host_token = [str(x) for x in range(len(hosts))]
        for i, v in enumerate(hosts):
            record.loc[record['current_host_uuid'] == v, 'curr_host_token'] = host_token[i]

        ax.barh(record['curr_host_token'], record['duration'],
                left=record['start_time'], height=0.4, color=record['color'])

        ax.legend(attack_action_legend, attack_action_legend_name, loc='lower left')
        plt.gca().invert_yaxis()
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('Hosts', weight='bold', fontsize=18)
        fig.tight_layout()
        plt.savefig(directory + '/experimental_data/plots/' + self.features+'/attack_action_record_group_by_host.png')
        plt.show()

    def visualise_attack_operation(self):
        """
        visualise the action flow of attack operation
        """
        record = self._attack_record
        fig, ax = plt.subplots(1, figsize=(16, 5))
        ax.barh(record['name'], record['duration'],
                left=record['start_time'], height=0.1, zorder=1)

        interrupted_record = record[record['interrupted_in'] != 'None']
        interrupted_record['color'] = np.where(interrupted_record['interrupted_in'] == 'network', 'green', 'orange')
        ax.scatter(interrupted_record['finish_time'], interrupted_record['name'], color=interrupted_record['color'],
                   zorder=3)

        compromise_record = record[record['compromise_host'] != 'None']
        print("total compromised hosts: ", len(compromise_record))
        ax.scatter(compromise_record['finish_time'], compromise_record['name'], color='red', zorder=2)

        custom_lines_attack = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10), ]

        ax.legend(custom_lines_attack, ['network layer MTD', 'application layer MTD', 'compromise host'],
                  loc='upper right')

        plt.gca().invert_yaxis()
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('Attack Actions', weight='bold', fontsize=18)
        fig.tight_layout()
        plt.savefig(directory + '/experimental_data/plots/' + self.features + '/attack_record.png')
        plt.show()

    def visualise_mtd_operation(self):
        """
        visualise the action flow of mtd operation
        """
        if len(self._mtd_record) == 0:
            return
        record = self._mtd_record
        fig, ax = plt.subplots(1, figsize=(16, 6))
        colors = ['blue', 'green', 'orange']
        mtd_action_legend = []
        mtd_action_legend_name = []
        for i, v in enumerate(record['executed_at'].unique()):
            record.loc[record['executed_at'] == v, 'color'] = colors[i]
            mtd_action_legend.append(Line2D([0], [0], color=colors[i], lw=4))
            mtd_action_legend_name.append(v)
        ax.barh(record['name'].astype(str), record['duration'],
                left=record['start_time'], height=0.4, color=record['color'])
        ax.legend(mtd_action_legend, mtd_action_legend_name, loc='lower right')
        plt.gca().invert_yaxis()
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('MTD Strategies', weight='bold', fontsize=18)
        fig.tight_layout()
        plt.savefig(directory + '/experimental_data/plots/' + self.features + '/mtd_record.png')
        plt.show()



    def get_network(self):
        return self._network
    
    def visualize_host_compromise_ratio(self):
        record = self._security_metric_record
        plt.figure(1, figsize=(16, 5))
        plt.plot(record['times'], record['host_compromise_ratio'], label='Host Compromise Ratio')
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('Host Compromise Ratio', weight='bold', fontsize=18)
        plt.legend()
        plt.savefig(directory + '/experimental_data/plots/host_compromise_ratio.png')
        plt.show()

    def visualize_attack_path_exposure_score(self):
        record = self._security_metric_record
        plt.figure(1, figsize=(16, 5))
        plt.plot(record['times'], record['attack_path_exposure_score'], label='Attack Path Exposure Score')
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('Attack Path Exposure Score', weight='bold', fontsize=18)
        plt.legend()
        plt.savefig(directory + '/experimental_data/plots/attack_path_exposure_score.png')
        plt.show()
    
    def visualize_risk(self):
        record = self._security_metric_record
        plt.figure(1, figsize=(16, 5))
        plt.plot(record['times'], record['risk'], label='Risk')
        plt.xlabel('Time', weight='bold', fontsize=18)
        plt.ylabel('Risk', weight='bold', fontsize=18)
        plt.legend()
        plt.savefig(directory + '/experimental_data/plots/risk.png')
        plt.show()