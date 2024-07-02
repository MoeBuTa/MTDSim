from mtdnetwork.mtd import MTD


class CompleteTopologyShuffle(MTD):
    """
    Completely regenerates the network, preserving the hosts from previously.
    """

    def __init__(self, network=None):
        super().__init__(name="CompleteTopologyShuffle",
                         mtd_type='shuffle',
                         resource_type='network',
                         network=network)

    def mtd_operation(self, adversary=None):
        hosts = self.network.get_hosts()

        # Regenerate the network graph
        self.network.gen_graph()
        for host_id, host_instance in hosts.items():
            self.network.graph.nodes[host_id]["host"] = host_instance
        self.network.update_reachable_mtd()

        # Update Attack Path Exposure for target networks
        if self.network.get_network_type() == 0:
            self.network.add_attack_path_exposure()
        self.network.add_shortest_path()

        # Set nHosts per layer
        # Generate new graph with those number of hosts per layer
        # Replace the new nodes with the old host instances
