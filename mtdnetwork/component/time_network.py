from mtdnetwork.component.network import Network
from mtdnetwork.statistic.mtd_statistics import MTDStatistics
from mtdnetwork.component.host import Host
import random


class TimeNetwork(Network):

    def __init__(self, total_nodes=50, total_endpoints=5, total_subnets=8, total_layers=4,
                 target_layer=4, total_database=5, terminate_compromise_ratio=0.8):
        # default parameters
        self._mtd_stats = MTDStatistics()
        self._mtd_queue = []
        self._suspension_queue = dict()
        self._unfinished_mtd = dict()
        if total_nodes < 2 * total_subnets:
            total_nodes = 2 * total_subnets
        super().__init__(total_nodes=total_nodes, total_endpoints=total_endpoints, total_subnets=total_subnets,
                         total_layers=total_layers, target_layer=target_layer, total_database=total_database)
        self.init_network()
        self.last_mtd_triggered_time = 0

    def setup_network(self):
        """
        Using the generated graph, generates a host for each node on the graph.
        """
        ip_addresses = []

        for host_id in self.nodes:
            node_os = Host.get_random_os()
            node_os_version = Host.get_random_os_version(node_os)
            node_ip = Host.get_random_address(existing_addresses=ip_addresses)
            ip_addresses.append(node_ip)
            self.graph.nodes[host_id]["host"] = Host(
                node_os,
                node_os_version,
                host_id,
                node_ip,
                random.choices(self.users_list, k=self.users_per_host),
                self,
                self.service_generator
            )

    def is_compromised(self, compromised_hosts):
        # 80% compromise ratio
        return len(compromised_hosts) / self.total_nodes > 0.8

    def get_mtd_stats(self):
        return self._mtd_stats

    def get_mtd_queue(self):
        return self._mtd_queue

    def get_suspended_mtd(self):
        return self._suspension_queue

    def get_unfinished_mtd(self):
        return self._unfinished_mtd

    def set_unfinished_mtd(self, mtd):
        self._unfinished_mtd[mtd.get_resource_type()] = mtd

    def set_last_mtd_triggered_time(self, time):
        self.last_mtd_triggered_time = time





