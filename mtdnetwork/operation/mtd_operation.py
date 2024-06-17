from mtdnetwork.component.time_generator import exponential_variates
import logging
import simpy
from mtdnetwork.component.mtd_scheme import MTDScheme
from mtdnetwork.statistic.evaluation import Evaluation
import numpy as np


class MTDOperation:

    def __init__(self, env, end_event, network, attack_operation, scheme, adversary, proceed_time=0,
                 mtd_trigger_interval=None, custom_strategies=None):
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
        self.logging = False

        self._mtd_scheme = MTDScheme(network=network, scheme=scheme, mtd_trigger_interval=mtd_trigger_interval,
                                     custom_strategies=custom_strategies)
        self._proceed_time = proceed_time

        self.application_layer_resource = simpy.Resource(self.env, 1)
        self.network_layer_resource = simpy.Resource(self.env, 1)
        self.reserve_resource = simpy.Resource(self.env, 1)

    def proceed_mtd(self):
        if self.network.get_unfinished_mtd():
            for k, v in self.network.get_unfinished_mtd().items():
                self._mtd_scheme.suspend_mtd(v)
        if self._mtd_scheme.get_scheme() == 'simultaneous':
            self.env.process(self._mtd_batch_trigger_action())
        else:
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
                if not self.end_event.triggered:  # Check if the event has not been triggered yet
                    self.end_event.succeed()
                return
            
            # test - getting some stats out of the network everytime an MTD is triggered
            # timenetwork.py
            # mtd_stats = self.network.get_mtd_stats().dict()
            exposed_endpoints = len(self.network.get_exposed_endpoints())
            attack_path_exposure = self.network.attack_path_exposure()
            # advesary.py
            num_vulns = len(self.adversary.get_curr_vulns())
            # attack_stats = self.adversary.get_statistics()
            # compromised_hosts = len(self.adversary.get_compromised_hosts())
            # current_attack = self.adversary.get_curr_process()
            # evaluation.py
            evaluation = Evaluation(self.network, self.adversary)
            mtd_freq = evaluation.mtd_execution_frequency()
            compromised_num = evaluation.compromised_num()
            host_compromise_ratio = compromised_num/len(self.network.get_hosts())

            # scan_port = len(evaluation.compromise_record_by_attack_action('SCAN_PORT'))
            # exploit_vlun = len(evaluation.compromise_record_by_attack_action('EXPLOIT_VULN'))
            # brute_force = len(evaluation.compromise_record_by_attack_action('BRUTE_FORCE'))

            evaluation_results = evaluation.evaluation_result_by_compromise_checkpoint(np.arange(0.01, 1.01, 0.01))
            if evaluation_results:
                total_asr, total_time_to_compromise, total_compromises = 0, 0, 0

                for result in evaluation_results:
                    if result['host_compromise_ratio'] != 0:  
                        total_time_to_compromise += result['time_to_compromise']
                        total_compromises += 1
                    total_asr += result['attack_success_rate']

                # Calculate overall averages
                overall_asr_avg = total_asr / len(evaluation_results) if evaluation_results else 0
                overall_mttc_avg = total_time_to_compromise / total_compromises if total_compromises else 0
            else:
                overall_asr_avg = 0
                overall_mttc_avg = 0

            time_since_last_mtd = self.env.now - self.network.last_mtd_triggered_time
            self.network.last_mtd_triggered_time = self.env.now

            # logging.info(f"MTD Stats: {mtd_stats}")
            #logging.info(f"Current Attack: {current_attack}")
            # logging.info(f"Attack Stats: {attack_stats}")
            # logging.info(f"Number of Compromised Hosts: {compromised_hosts}")
            # logging.info(f"Evaluation Results: {evaluation_results}")
            # logging.info(f"Scan Port Compromise Ratio: {scan_port/compromised_num}")
            # logging.info(f"Exploit Vuln Compromise Ratio: {exploit_vlun/compromised_num}")
            # logging.info(f"Brute Force Compromise Ratio: {brute_force/compromised_num}")

            logging.info(f"STATS BEFORE MTD OPERATION")
            logging.info(f"Host Compromise Ratio: {host_compromise_ratio}")
            logging.info(f"Num Exposed Endpoints: {exposed_endpoints}")
            logging.info(f"Attack Path Exposure Score: {attack_path_exposure}")
            logging.info(f"Attack Success Rate: {overall_asr_avg}")

            logging.info(f"MTD Frequency: {mtd_freq}")
            logging.info(f"Time to Compromise: {overall_mttc_avg}")
            logging.info(f"Time Since Last MTD: {time_since_last_mtd}")


            # register an MTD
            if not self.network.get_mtd_queue():
                self._mtd_scheme.register_mtd()
            # trigger an MTD
            if self.network.get_suspended_mtd():
                mtd = self._mtd_scheme.trigger_suspended_mtd()
            else:
                mtd = self._mtd_scheme.trigger_mtd()
           
            if self.logging:
                logging.info('MTD: %s triggered %.1fs' % (mtd.get_name(), self.env.now + self._proceed_time))
            logging.info('MTD: %s triggered %.1fs' % (mtd.get_name(), self.env.now + self._proceed_time))

            resource = self._get_mtd_resource(mtd)
            if len(resource.users) == 0:
                self.env.process(self._mtd_execute_action(env=self.env, mtd=mtd))
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

    def _mtd_batch_trigger_action(self):
        """
        trigger all MTDs at a time with a fixed priority
        """
        while True:
            # terminate the simulation if the network is compromised
            if self.network.is_compromised(
                    compromised_hosts=self.attack_operation.get_adversary().get_compromised_hosts()):
                return
            
            suspension_queue = self.network.get_suspended_mtd()
            if suspension_queue:
                # triggering the suspended MTDs by priority
                suspended_mtd_priorities = sorted(suspension_queue.keys())
                for priority in suspended_mtd_priorities:
                    resource = self._get_mtd_resource(mtd=suspension_queue[priority])
                    if len(resource.users) == 0:
                        mtd = self._mtd_scheme.trigger_suspended_mtd()
                        logging.info('MTD: %s triggered %.1fs' % (mtd.get_name(), self.env.now + self._proceed_time))
                        yield self.env.process(self._mtd_execute_action(env=self.env, mtd=mtd))
            else:
                # register and trigger all MTDs
                if not self.network.get_suspended_mtd() and not self.network.get_mtd_queue():
                    self._mtd_scheme.register_mtd()
                while self.network.get_mtd_queue():
                    mtd = self._mtd_scheme.trigger_mtd()
                    logging.info('MTD: %s triggered %.1fs' % (mtd.get_name(), self.env.now + self._proceed_time))
                    resource = self._get_mtd_resource(mtd=mtd)
                    if len(resource.users) == 0:
                        # execute MTD
                        self.env.process(self._mtd_execute_action(env=self.env, mtd=mtd))
                    else:
                        # suspend MTD if resource is occupied
                        self._mtd_scheme.suspend_mtd(mtd_strategy=mtd)
                        logging.info('MTD: %s suspended at %.1fs due to resource occupation' %
                                     (mtd.get_name(), self.env.now + self._proceed_time))

            # exponential distribution for triggering MTD operations
            yield self.env.timeout(exponential_variates(self._mtd_scheme.get_mtd_trigger_interval(),
                                                        self._mtd_scheme.get_mtd_trigger_std()))

    def _mtd_execute_action(self, env, mtd):
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

        # release resource
        self._get_mtd_resource(mtd).release(request)
        # append execution records
        self.network.get_mtd_stats().append_mtd_operation_record(mtd, start_time, finish_time, duration)
        # interrupt adversary
        self._interrupt_adversary(env, mtd)

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
                
                if logging:
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
