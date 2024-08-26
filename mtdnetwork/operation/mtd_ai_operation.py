from mtdnetwork.component.time_generator import exponential_variates
import logging
import simpy
from mtdnetwork.component.mtd_scheme import MTDScheme
from mtdnetwork.statistic.evaluation import Evaluation
import numpy as np
from mtdnetwork.mtdai.mtd_ai import choose_action
import pandas as pd
import random
from mtdnetwork.statistic.security_metric_statistics import SecurityMetricStatistics


class MTDAIOperation:

    def __init__(self, security_metrics_record ,env, end_event, network, attack_operation, scheme, adversary, features,proceed_time=0,
                 mtd_trigger_interval=None, custom_strategies=None, main_network=None, attacker_sensitivity=None, epsilon=None):
        """
        :param env: the parameter to facilitate simPY env framework
        :param network: the simulation network
        :param attack_operation: the attack operation
        :param scheme:alternatively, simultaneously, randomly
        :param proceed_time:the time to proceed MTD simulation
        :param custom_strategies:specific MTD priority strategy for alternative scheme or single scheme
        :param mtd_trigger_interval:the interval to trigger MTD operations
        :param adversary: the adversary 
        """
        self.env = env
        self.end_event = end_event
        self.network = network
        self.attack_operation = attack_operation
        self.adversary = adversary
        self.attacker_sensitivity = attacker_sensitivity
        self.logging = False

        self._mtd_scheme = MTDScheme(network=network, scheme=scheme, mtd_trigger_interval=mtd_trigger_interval,
                                     custom_strategies=custom_strategies)
        self._proceed_time = proceed_time

        self.application_layer_resource = simpy.Resource(self.env, 1)
        self.network_layer_resource = simpy.Resource(self.env, 1)
        self.reserve_resource = simpy.Resource(self.env, 1)

        self.main_network = main_network
        self.epsilon = epsilon
        self.features = features
        self.security_metrics_record = security_metrics_record
        self.attack_dict = {
            'SCAN_HOST': 1,
            'ENUM_HOST': 2,
            'SCAN_PORT': 3,
            'SCAN_NEIGHBOR': 4,
            'EXPLOIT_VULN': 5,
            'BRUTE_FORCE': 6,
        }
        self.evaluation = Evaluation(network=network, adversary=adversary, features=features, security_metrics_record = security_metrics_record)

        self.attack_dict = {"SCAN_HOST": 1, "ENUM_HOST": 2, "SCAN_PORT": 3, "EXPLOIT_VULN": 4, "SCAN_NEIGHBOR": 5, "BRUTE_FORCE": 6}



    def proceed_mtd(self):
        if self.network.get_unfinished_mtd():
            for k, v in self.network.get_unfinished_mtd().items():
                self._mtd_scheme.suspend_mtd(v)
        self.env.process(self._mtd_trigger_action())
        

    def _mtd_trigger_action(self):
        """
        trigger an MTD strategy in a given exponential time (next_mtd) in the queue
        Select Execute or suspend/discard MTD strategy
        based on the given resource occupation condition
        """
        while True:
            # terminate the simulation if the network is compromised
            if self.network.is_compromised(compromised_hosts=self.attack_operation.get_adversary().get_compromised_hosts()):
                if not self.end_event.triggered:  # Check if the event has not been triggered yet (will crash without this check)
                    self.end_event.succeed()
                return
            
            state, time_series = self.get_state_and_time_series()
            self.network.get_security_metric_stats().append_security_metric_record(state, time_series, round(self.env.now, -2))

            # if using the mtd_ai scheme
            if self._mtd_scheme._scheme == 'mtd_ai':

                # Static network degradation factor
                if (self.env.now - self.network.get_last_mtd_triggered_time()) > 2000: # The number 100 is just a temperory threshold
                    action = 1
                else:
                    action = choose_action(state, time_series, self.main_network, 5, self.epsilon)
                
                if self.logging:
                    logging.info('Static period: %s' % (self.env.now - self.network.get_last_mtd_triggered_time()))

                if action > 0 or self.network.get_last_mtd_triggered_time() == 0:
                    self.network.set_last_mtd_triggered_time(self.env.now)
                if self.logging:
                    logging.info('Action: %s' % action)
            else:
                action = 1 #if using a different scheme other than mtd_ai set action to 1 to trigger MTD on timer

            if action > 0:
                # register an MTD
                if not self.network.get_mtd_queue():
                    if self._mtd_scheme._scheme == 'mtd_ai':
                        self._mtd_scheme.register_mtd(mtd_action=action)
                        # Register the mtd in scorer as well
                        self.network.scorer.register_mtd(self._mtd_scheme.register_mtd(action))
                    else:
                        self._mtd_scheme.register_mtd(mtd_action=None)
                        self.network.scorer.register_mtd(self._mtd_scheme.register_mtd(mtd_action=None))
                # trigger an MTD
                if self.network.get_suspended_mtd():
                    mtd = self._mtd_scheme.trigger_suspended_mtd()
                else:
                    mtd = self._mtd_scheme.trigger_mtd()
            
                if self.logging:
                    logging.info('MTD: %s triggered %.1fs' % (mtd.get_name(), self.env.now + self._proceed_time))

                resource = self._get_mtd_resource(mtd)
                if len(resource.users) == 0:
                    self.env.process(self._mtd_execute_action(env=self.env, mtd=mtd, state=state, time_series=time_series, action=action))
                else:
                    # suspend if suspended dict doesn't have the same MTD.
                    # else discard
                    if mtd.get_priority() not in self.network.get_suspended_mtd():
                        self._mtd_scheme.suspend_mtd(mtd)

                        if self.logging:
                            logging.info('MTD: %s suspended at %.1fs due to resource occupation' %
                                    (mtd.get_name(), self.env.now + self._proceed_time))

                # exponential time interval for triggering MTD operations
                yield self.env.timeout(exponential_variates(self._mtd_scheme.get_mtd_trigger_interval(),
                                                            self._mtd_scheme.get_mtd_trigger_std()))

    def _mtd_execute_action(self, env, mtd, state, time_series, action):
        """
        Action for executing MTD
        """
        # deploy mtd
        self.network.set_unfinished_mtd(mtd)
        request = self._get_mtd_resource(mtd).request()
        yield request
        start_time = env.now + self._proceed_time

        if self.logging:
            logging.info('MTD: %s deployed in the network at %.1fs.' % (mtd.get_name(), start_time))

        yield env.timeout(exponential_variates(mtd.get_execution_time_mean(),
                                               mtd.get_execution_time_std()))

        # if network is already compromised while executing mtd:
        if self.network.is_compromised(compromised_hosts=self.attack_operation.get_adversary().get_compromised_hosts()):
            return

        # execute mtd
        mtd.mtd_operation(self.attack_operation.get_adversary())

        finish_time = env.now + self._proceed_time
        duration = finish_time - start_time
        
        if self.logging:
            logging.info('MTD: %s finished in %.1fs at %.1fs.' % (mtd.get_name(), duration, finish_time))

        self.network.last_mtd_triggered_time = self.env.now


        # release resource
        self._get_mtd_resource(mtd).release(request)
        # append execution records
        self.network.get_mtd_stats().append_mtd_operation_record(mtd, start_time, finish_time, duration)
        # interrupt adversary
        self._interrupt_adversary(env, mtd)

        # Update time since last MTD operation
        self.network.last_mtd_triggered_time = self.env.now



    def _get_mtd_resource(self, mtd):
        """Get the resource to be occupied by the mtd"""
        if mtd.get_resource_type() == 'network':
            return self.network_layer_resource
        elif mtd.get_resource_type() == 'application':
            return self.application_layer_resource
        return self.reserve_resource

    def _interrupt_adversary(self, env, mtd):
        """
        interrupt the attack process of the adversary
        """
        attack_process = self.attack_operation.get_attack_process()
        if attack_process is not None and attack_process.is_alive:
            if mtd.get_resource_type() == 'network':
                self.attack_operation.set_interrupted_mtd(mtd)
                self.attack_operation.get_attack_process().interrupt()
                
                if self.logging:
                    logging.info(
                    'MTD: Interrupted %s at %.1fs!' % (self.attack_operation.get_adversary().get_curr_process(),
                                                       env.now + self._proceed_time))
                self.network.get_mtd_stats().add_total_attack_interrupted()
            elif mtd.get_resource_type() == 'application' and \
                    self.attack_operation.get_adversary().get_curr_process() not in [
                'SCAN_HOST',
                'ENUM_HOST',
                'SCAN_NEIGHBOR']:
                self.attack_operation.set_interrupted_mtd(mtd)
                self.attack_operation.get_attack_process().interrupt()

                if self.logging:
                    logging.info(
                    'MTD: Interrupted %s at %.1fs!' % (self.attack_operation.get_adversary().get_curr_process(),
                                                       env.now + self._proceed_time))
                    
                self.network.get_mtd_stats().add_total_attack_interrupted()

    def get_proceed_time(self):
        return self._proceed_time

    def get_application_resource(self):
        return self.application_layer_resource

    def get_network_resource(self):
        return self.network_layer_resource

    def get_reserve_resource(self):
        return self.reserve_resource

    def get_mtd_scheme(self):
        return self._mtd_scheme
    
    def get_state_and_time_series(self):
        # State metrics

        compromised_num = self.evaluation.compromised_num()
        host_compromise_ratio = compromised_num/len(self.network.get_hosts()) \

        exposed_endpoints = len(self.network.get_exposed_endpoints())

        attack_path_exposure = self.network.attack_path_exposure()

        attack_stats = self.adversary.get_network().get_scorer().get_statistics()
        risk = attack_stats['Vulnerabilities Exploited']['risk'][-1] if attack_stats['Vulnerabilities Exploited']['risk'] else 0
        roa = attack_stats['Vulnerabilities Exploited']['roa'][-1] if attack_stats['Vulnerabilities Exploited']['roa'] else 0

        shortest_paths = self.network.scorer.shortest_path_record 
        shortest_path_variability = (len(shortest_paths[-1]) - len(shortest_paths[-2]))/len(shortest_paths) if len(shortest_paths) > 1 else 0

        evaluation_results = self.evaluation.evaluation_result_by_compromise_checkpoint(np.arange(0.01, 1.01, 0.01))
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
        time_since_last_mtd = self.env.now - self.network.last_mtd_triggered_time
        # time_since_last_mtd = 1
        mtd_freq = self.evaluation.mtd_execution_frequency()

        state_array = np.array([host_compromise_ratio, exposed_endpoints, attack_path_exposure, overall_asr_avg, roa, shortest_path_variability, risk])
 

        time_series_array = np.array([mtd_freq, overall_mttc_avg, time_since_last_mtd])

        # self.security_metrics_record.append_security_metric_record(state_array,time_series_array, env.now)
 
        return state_array, time_series_array
    
    # def get_state_and_time_series(self):

   
    #     evaluation = Evaluation(self.network, self.adversary, self.features, self.security_metrics_record)
    #     exposed_endpoints = len(self.network.get_exposed_endpoints())
    #     attack_path_exposure = self.network.attack_path_exposure()
        
    #     shortest_paths = self.network.scorer.shortest_path_record 
    #     shortest_path_variability = (len(shortest_paths[-1]) - len(shortest_paths[-2]))/len(shortest_paths) if len(shortest_paths) > 1 else 0
        

    #     compromised_num = evaluation.compromised_num()
    #     host_count = len(self.network.get_hosts())
    #     host_compromise_ratio = compromised_num / host_count
  

    #     sensitivity_factor = random.random()
    #     if sensitivity_factor <= self.attacker_sensitivity:
    #         current_attack = self.adversary.get_curr_process()
    #         current_attack_value = self.attack_dict.get(current_attack, 7)
    #     else:
    #         current_attack_value = 7

    #     evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint(np.arange(0.01, 1.01, 0.01))
    #     if evaluation_results:
    #         total_asr, total_time_to_compromise, total_compromises = 0, 0, 0

    #         for result in evaluation_results:
    #             if result['host_compromise_ratio'] != 0:  
    #                 total_time_to_compromise += result['time_to_compromise']
    #                 total_compromises += 1
    #             total_asr += result['attack_success_rate']

    #         overall_asr_avg = total_asr / len(evaluation_results) if evaluation_results else 0
    #         overall_mttc_avg = total_time_to_compromise / total_compromises if total_compromises else 0
    #     else:
    #         overall_asr_avg = 0
    #         overall_mttc_avg = 0

    #     time_since_last_mtd = self.env.now - self.network.get_last_mtd_triggered_time()
    #     mtd_freq = evaluation.mtd_execution_frequency()

    #     attack_stats = self.adversary.get_network().get_scorer().get_statistics()
    #     risk = attack_stats['Vulnerabilities Exploited']['risk'][-1] if attack_stats['Vulnerabilities Exploited']['risk'] else 0
    #     roa = attack_stats['Vulnerabilities Exploited']['roa'][-1] if attack_stats['Vulnerabilities Exploited']['roa'] else 0

    #     # state_dict = {
    #     #     "host_compromise_ratio": host_compromise_ratio,
    #     #     "exposed_endpoints": exposed_endpoints,
    #     #     "attack_path_exposure": attack_path_exposure,
    #     #     "overall_asr_avg": overall_asr_avg,
    #     #     "roa": roa,
    #     #     "shortest_path_variability": shortest_path_variability,
    #     #     "risk": risk,
    #     #     "current_attack_value": current_attack_value
    #     # }
         
    #     state_array = np.array([host_compromise_ratio, exposed_endpoints, attack_path_exposure, overall_asr_avg, roa, shortest_path_variability, risk, current_attack_value])
 

    #     time_series_array = np.array([mtd_freq, overall_mttc_avg, time_since_last_mtd])

    #     self.security_metrics_record.append_security_metric_record(state_array, time_series_array, self.env.now)
 
    #     return state_array, time_series_array
    