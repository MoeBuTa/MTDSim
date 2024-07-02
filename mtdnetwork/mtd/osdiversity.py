from mtdnetwork.mtd import MTD
import random
from mtdnetwork.data import constants


class OSDiversity(MTD):
    def __init__(self, network=None):
        super().__init__(name="OSDiversity",
                         mtd_type='diversity',
                         resource_type='application',
                         network=network)

    def mtd_operation(self, adversary=None):
        """
        todo: dynamic MTDs based on the network state
        """
        service_generator = self.network.get_service_generator()
        hosts = self.network.get_hosts()
        for host_id, host_instance in hosts.items():
            if host_id in self.network.exposed_endpoints:
                continue
            prev_os = host_instance.os_type
            prev_os_version = host_instance.os_version
            prev_os_version_index = constants.OS_VERSION_DICT[prev_os].index(prev_os_version)
            new_os = random.choice(constants.OS_TYPES)
            new_os_version = constants.OS_VERSION_DICT[new_os][prev_os_version_index]

            host_instance.os_type = new_os
            host_instance.os_version = new_os_version

            for node_id in range(host_instance.total_nodes):
                if node_id == host_instance.target_node:
                    continue

                curr_service = host_instance.graph.nodes[node_id]["service"]
                if not service_generator.service_is_compatible_with_os(new_os, new_os_version, curr_service):
                    host_instance.graph.nodes[node_id]["service"] = service_generator.get_random_service_latest_version(
                        host_instance.os_type,
                        host_instance.os_version
                    )
        # Update Attack Path Exposure for target networks
        if self.network.get_network_type() == 0:
            self.network.add_attack_path_exposure()
        self.network.add_shortest_path()