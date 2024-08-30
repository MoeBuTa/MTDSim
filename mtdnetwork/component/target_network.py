from mtdnetwork.component.network import Network


class TargetNetwork(Network):

    def __init__(self, total_nodes=200, total_endpoints=20, total_subnets=20, total_layers=5, target_layer=2,
                 total_database=5):
        super().__init__(total_nodes=total_nodes, total_endpoints=total_endpoints, total_subnets=total_subnets,
                         total_layers=total_layers, target_layer=target_layer, total_database=total_database)

    def copy_network(self, network):
        self.total_nodes = network.total_nodes
        self.total_endpoints = network.total_endpoints
        self.total_subnets = network.total_subnets
        self.layers = network.total_layers
        self.target_layer = network.target_layer
        self.network_type = 0

        self.graph = network.get_graph_copy().copy()
        self.colour_map = network.get_colourmap()
        self.pos = network.get_pos()
        self.node_per_layer = network.get_node_per_layer()
        self.users_list = network.get_users_list()
        self.users_per_host = network.get_users_per_host()
        self.scorer.set_initial_statistics(self)
        self.update_host_information()
        self.add_attack_path_exposure()
        ##
        # self.add_shortest_path()
        


