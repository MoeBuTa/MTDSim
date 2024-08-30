import networkx as nx
import pkg_resources
import matplotlib.pyplot as plt
import numpy as np
import random
import mtdnetwork.data.constants as constants
import mtdnetwork.component.services as services
from mtdnetwork.component.host import Host
from mtdnetwork.statistic.scorer import Scorer
import os


class Network:

    def __init__(self, total_nodes, total_endpoints, total_subnets, total_layers, total_database, target_layer=None,
                 users_to_nodes_ratio=constants.USER_TO_NODES_RATIO,
                 prob_user_reuse_pass=constants.USER_PROB_TO_REUSE_PASS, seed=None):
        """
        Initialises the state of the network for the simulation.

        Parameters:
            total_nodes:
                the number of the nodes in the network.
            total_endpoints:
                the number of nodes exposed on the internet (hacker can interact directly with them).
                total_endpoints must be less than total_nodes.
            total_subnets:
                how many subnets in the network.
            total_layers:
                how many layers deep from the exposed endpoints the network is.
            user_to_nodes_ratio:
                the percent of users in comparison to host machines.
                each node will then be given `int(1/user_to_nodes_ratio)` users each (less users more users on each computer).
            prob_user_reuse_pass:
                the probability that a user has reused their password.
            seed:
                the seed for the random number generator if one needs to be set
        """
        self.graph = None
        self.colour_map = None
        self.users_per_host = None
        self.total_users = None
        self.users_list = None
        self.pos = None
        if seed is not None:
            random.seed(seed)
        self.total_nodes = total_nodes
        self.total_endpoints = total_endpoints
        self.total_subnets = total_subnets
        self.layers = total_layers
        self.exposed_endpoints = [n for n in range(total_endpoints)]

        self.total_database = total_database
        self._database = [n for n in range(total_nodes - total_database, total_nodes)]

        self.tags = []
        self.tag_priority = []
        self.service_generator = services.ServicesGenerator()
        self.nodes = [n for n in range(total_nodes)]
        self.mtd_strategies = []

        self.reachable = []
        self.compromised_hosts = []
        self.node_per_layer = []
        # Network type 0 is a targetted attack, Network type 1 is a general attack (no target node)
        self.network_type = 1
        self.vuln_dict = {}
        self.vuln_count = {}
        self.service_dict = {}
        self.service_count = {}
        self.total_vulns = 0
        # self.total_vulns_lst = []
        # self.vuln_dict_lst = []
        self.total_services = 0
        self.scorer = Scorer()
        self.users_to_nodes_ratio = users_to_nodes_ratio
        self.prob_user_reuse_pass = prob_user_reuse_pass
   
        self.target_node = None
        self.target_layer = target_layer


    def init_network(self):
        self.assign_tags()
        self.assign_tag_priority()
        self.setup_users(self.users_to_nodes_ratio, self.prob_user_reuse_pass, constants.USER_TOTAL_FOR_EACH_HOST)
        self.gen_graph()
        self.setup_network()
        self.scorer.set_initial_statistics(self)

    def update_host_information(self):
        """"
        Updates the host
        """
        for host_id in self.nodes:
            host = self.get_host(host_id)
            host.swap_network(self)



    def gen_graph(self, min_nodes_per_subnet=2, max_subnets_per_layer=5, subnet_m_ratio=0.2,
                  prob_inter_layer_edge=0.4):
        """
        Generates a network of subnets using the Barabasi-Albert Random Graph model.

        Parameters:
            min_nodes_per_subnet:
                minimum number of computer nodes for each subnet
            max_subnets_per_layer:
                the maximum number of subnets per layer
            subnet_m_ratio:
                a ratio that is used to determine the parameter m for the barabasi albert graph.
                m is the number of edges to attach from a new node to existing nodes
            prob_inter_layer_edge:
                probability that a node connects to a different layer in the network.
        """
        # Decide the number of subnets for each layer of the network
        subnets_per_layer = []
        while len(subnets_per_layer) < self.layers:
            # Adds 1 to start of array if array is empty
            if len(subnets_per_layer) == 0: subnets_per_layer.append(1)
            l_subnets = random.randint(1, max_subnets_per_layer)
            # Only appends value if it doesn't exceed maximum number of subnets possible
            if self.total_subnets - (sum(subnets_per_layer) + l_subnets) > self.layers - len(subnets_per_layer):
                subnets_per_layer.append(l_subnets)

        # Randomly adds one to random subnets until there is the correct amount of subnets (Could be Optimised in
        # future)
        while sum(subnets_per_layer) < self.total_subnets:
            s_index = random.randint(1, self.layers - 1)
            if subnets_per_layer[s_index] <= max_subnets_per_layer:
                subnets_per_layer[s_index] = subnets_per_layer[s_index] + 1

        max_subnet_in_layer = max(subnets_per_layer)

        # Assign nodes to each layer
        nodes_per_layer = [self.total_endpoints]
        # Appends the minimum number of nodes that should be in the layer
        for subs in subnets_per_layer[1:]:
            nodes_per_layer.append(min_nodes_per_subnet * subs)

        # Randomly adds one to random subnets until there is the correct number of nodes (Could be Opitmised in future)
        while sum(nodes_per_layer) < self.total_nodes:
            n_index = random.randint(1, self.layers - 1)
            nodes_per_layer[n_index] = nodes_per_layer[n_index] + 1

        # Assign number of nodes to each subnet
        subnet_nodes = []
        for i, subnets in enumerate(subnets_per_layer):
            # List containing the minimum number of nodes for every subnet in layer
            temp_subnet_nodes = [min_nodes_per_subnet for _i in range(subnets)]
            # Randomly adds one to random subnets until correct number of nodes for the layer
            while sum(temp_subnet_nodes) < nodes_per_layer[i]:
                n_index = random.randint(0, subnets - 1)
                temp_subnet_nodes[n_index] = temp_subnet_nodes[n_index] + 1
            subnet_nodes.append(temp_subnet_nodes)
            
        # self.graphenerate the graph
        self.graph = nx.Graph()
        # Node offset
        node_id = 0
        self.colour_map = []
        self.pos = {}
        attr = {}
        min_y_pos = 200000
        max_y_pos = -200000
        # Layer = i, subnet = j, s_nodes = # of nodes in subnet
        for i, subnet_node_list in enumerate(subnet_nodes):
            for j, s_nodes in enumerate(subnet_node_list):
                m = int(s_nodes * subnet_m_ratio)
                if m < 1:
                    m = 1
                elif m >= s_nodes:
                    m = s_nodes - 1
                subgraph = nx.barabasi_albert_graph(s_nodes, m)
                new_node_mapping = {k: k + node_id for k in range(s_nodes)}
                subgraph = nx.relabel_nodes(subgraph, new_node_mapping)
                new_attr = {k + node_id: {"subnet": j, "layer": i} for k in range(s_nodes)}
                attr = {**attr, **new_attr}

                # Setting offset to next empty node
                node_id += s_nodes

                subgraph_pos = nx.spring_layout(subgraph)
                if i != 0:
                    subgraph_pos = {
                        k: np.array(
                            [v[0] + i * 2.25, v[1] + j * 3 + 1.5 * (max_subnet_in_layer - len(subnet_node_list))])
                        for k, v in subgraph_pos.items()
                    }

                    for k, v in subgraph_pos.items():
                        y = v[1] + j * 3 + 1.5 * (max_subnet_in_layer - len(subnet_node_list))
                        subgraph_pos[k] = np.array(
                            [v[0] + i * 2.25, y]
                        )
                        if y < min_y_pos:
                            min_y_pos = y
                        if y > max_y_pos:
                            max_y_pos = y
                else:
                    subgraph_pos = {
                        k: np.array([0, k])
                        for k, _v in subgraph_pos.items()
                    }
                # Stores all the positions of items from subgraphs
                self.pos = {**self.pos, **subgraph_pos}
               
                # Selects Target Host
                if i == self.target_layer and j == 1 and self.network_type == 0 and self.target_node != -1:
                    self.target_node = node_id - random.randrange(0, s_nodes)
                    print("Target Node is: ", self.target_node)

                # Assigns Colour of nodes based on constant key
                for k in range(s_nodes):
                    self.colour_map.append(constants.NODE_COLOURS[i])

                # Adds Subgraph to final graph
                self.graph = nx.compose(self.graph, subgraph)

        # Defines Nodes for whole graph
        nx.set_node_attributes(self.graph, attr)

        # Connect the graph
        def get_other_node(node_list, node_degrees, other_node):
            n = random.choices(node_list, weights=node_degrees, k=1)[0]
            if n == other_node:
                return get_other_node(node_list, node_degrees, other_node)
            return n

        while not nx.is_connected(self.graph):
            node_layers = nx.get_node_attributes(self.graph, "layer")
            for i in range(self.layers - 1):
                node_a = [n for n in node_layers if node_layers[n] == i]
                degree_node_a = [self.graph.degree(n) for n in node_a]
                node_b = [n for n in node_layers if node_layers[n] == i + 1]
                degree_node_b = [self.graph.degree(n) for n in node_b]

                n_a1 = random.choices(node_a, weights=degree_node_a, k=1)[0]
                if not nx.is_connected(self.graph.subgraph(node_a + node_b)):
                    n_b = random.choices(node_b, weights=degree_node_b, k=1)[0]
                    self.graph.add_edge(n_a1, n_b)
                if random.random() < prob_inter_layer_edge and subnets_per_layer[i] > 1 and not nx.is_connected(
                        self.graph.subgraph(node_a)):
                    n_a2 = get_other_node(node_a, degree_node_a, n_a1)
                    self.graph.add_edge(n_a1, n_a2)

        endpoint_nodes_list = [n for n in range(self.total_endpoints)]
        blank_endpoints = []

        # Remove edges between endpoint nodes (not needed since adversary can reach them all anyway)
        # Store all external nodes with no internal nodes into blank_endpoints
        for n in endpoint_nodes_list:
            neighbors = list(self.graph.neighbors(n))
            for neighbor in neighbors:
                if neighbor in endpoint_nodes_list:
                    self.graph.remove_edge(n, neighbor)
            internal_connection = list(self.graph.neighbors(n))
            if not internal_connection:
                blank_endpoints.append(n)

        # Pulls from blank_endpoints until all exposed endpoints are connected
        node_layers = nx.get_node_attributes(self.graph, "layer")
        layer1_nodes = [n for n in node_layers if node_layers[n] == 1]
        layer1_weights = [self.graph.degree(n) for n in layer1_nodes]
        while len(blank_endpoints):
            endpoint = blank_endpoints.pop(0)
            other_node = random.choices(layer1_nodes, weights=layer1_weights, k=1)[0]
            self.graph.add_edge(endpoint, other_node)

        # Updates Colour of target node to red
        if self.network_type == 0:
            self.colour_map[self.target_node] = "red"

        # Fix positions for endpoints
        for n in range(self.total_endpoints):
            position = (n + 1) / self.total_endpoints * (max_y_pos - min_y_pos) + min_y_pos
            new_pos = {n: np.array([0, position])}
            self.pos.update(new_pos)

        # Update Nodes Per Layer for Complete topology shuffling
        self.node_per_layer = nodes_per_layer.copy()
        self.node_per_layer[0] = self.total_endpoints

        # print("Endpoint list:", self.total_endpoints)
        # print("Node list:", self.nodes)
        # print("Exposed Endpoint list: ", self.exposed_endpoints)
        # print("nodes per layer: ", self.node_per_layer)

    # def gen_graph(self, min_nodes_per_subnet=3, max_subnets_per_layer=5, subnet_m_ratio=0.2,
    #               prob_inter_layer_edge=0.4):
    #     """
    #     Generates a network of subnets using the Barabasi-Albert Random Graph model.
    #
    #     Parameters:
    #         min_nodes_per_subnet:
    #             minimum number of computer nodes for each subnet
    #         max_subnets_per_layer:
    #             the maximum number of subnets per layer
    #         subnet_m_ratio:
    #             a ratio that is used to determine the parameter m for the barabasi albert graph.
    #             m is the number of edges to attach from a new node to existing nodes
    #         prob_inter_layer_edge:
    #             probability that a node connects to a different layer in the network.
    #     """
    #     # Decide the number of subnets for each layer of the network
    #     # Adds 1 to start of array
    #     subnets_per_layer = [1]
    #     while 0 < len(subnets_per_layer) < self.layers - 1:
    #         l_subnets = random.randint(1, max_subnets_per_layer)
    #         # Only appends value if it doesn't exceed maximum number of subnets possible
    #         if self.total_subnets - (sum(subnets_per_layer) + l_subnets) > self.layers - len(subnets_per_layer):
    #             subnets_per_layer.append(l_subnets)
    #     # Adds 1 to end of array
    #     subnets_per_layer.append(1)
    #
    #     # Randomly adds one to random subnets until there is the correct amount of subnets (Could be Optimised in
    #     # future)
    #     while sum(subnets_per_layer) < self.total_subnets:
    #         s_index = random.randint(1, self.layers - 2)
    #         if subnets_per_layer[s_index] <= max_subnets_per_layer:
    #             subnets_per_layer[s_index] = subnets_per_layer[s_index] + 1
    #
    #     max_subnet_in_layer = max(subnets_per_layer)
    #
    #     # Assign nodes to each layer
    #     nodes_per_layer = [self.total_endpoints]
    #     # Appends the minimum number of nodes that should be in the layer
    #     for subs in subnets_per_layer[1:-1]:
    #         nodes_per_layer.append(min_nodes_per_subnet * subs)
    #
    #     nodes_per_layer.append(self.total_database)
    #
    #     # Randomly adds one to random subnets until there is the correct number of nodes (Could be Opitmised in future)
    #     while sum(nodes_per_layer) < self.total_nodes:
    #         n_index = random.randint(1, self.layers - 2)
    #         nodes_per_layer[n_index] = nodes_per_layer[n_index] + 1
    #
    #     # Assign number of nodes to each subnet
    #     subnet_nodes = []
    #     for i, subnets in enumerate(subnets_per_layer):
    #         # List containing the minimum number of nodes for every subnet in layer
    #         temp_subnet_nodes = [min_nodes_per_subnet for _i in range(subnets)]
    #         # Randomly adds one to random subnets until correct number of nodes for the layer
    #         while sum(temp_subnet_nodes) < nodes_per_layer[i]:
    #             n_index = random.randint(0, subnets - 1)
    #             temp_subnet_nodes[n_index] = temp_subnet_nodes[n_index] + 1
    #         subnet_nodes.append(temp_subnet_nodes)
    #
    #     # generate the graph
    #     self.graph = nx.Graph()
    #     # Node offset
    #     node_id = 0
    #     self.colour_map = []
    #     self.pos = {}
    #     attr = {}
    #     min_y_pos = 200000
    #     max_y_pos = -200000
    #     # Layer = i, subnet = j, s_nodes = # of nodes in subnet
    #     for i, subnet_node_list in enumerate(subnet_nodes):
    #         for j, s_nodes in enumerate(subnet_node_list):
    #             m = int(s_nodes * subnet_m_ratio)
    #             if m < 1:
    #                 m = 1
    #             elif m >= s_nodes:
    #                 m = s_nodes - 1
    #             subgraph = nx.barabasi_albert_graph(s_nodes, m)
    #             new_node_mapping = {k: k + node_id for k in range(s_nodes)}
    #             subgraph = nx.relabel_nodes(subgraph, new_node_mapping)
    #             new_attr = {k + node_id: {"subnet": j, "layer": i} for k in range(s_nodes)}
    #             attr = {**attr, **new_attr}
    #
    #             # Setting offset to next empty node
    #             node_id += s_nodes
    #
    #             subgraph_pos = nx.spring_layout(subgraph)
    #             if i != 0:
    #                 subgraph_pos = {
    #                     k: np.array(
    #                         [v[0] + i * 2.25, v[1] + j * 3 + 1.5 * (max_subnet_in_layer - len(subnet_node_list))])
    #                     for k, v in subgraph_pos.items()
    #                 }
    #
    #                 for k, v in subgraph_pos.items():
    #                     y = v[1] + j * 3 + 1.5 * (max_subnet_in_layer - len(subnet_node_list))
    #                     subgraph_pos[k] = np.array(
    #                         [v[0] + i * 2.25, y]
    #                     )
    #                     if y < min_y_pos:
    #                         min_y_pos = y
    #                     if y > max_y_pos:
    #                         max_y_pos = y
    #             else:
    #                 subgraph_pos = {
    #                     k: np.array([0, k])
    #                     for k, _v in subgraph_pos.items()
    #                 }
    #             # Stores all the positions of items from subgraphs
    #             self.pos = {**self.pos, **subgraph_pos}
    #
    #             # Selects Target Host
    #             if self.network_type == 0 and i == self.target_layer and j == 1:
    #                 self.target_node = node_id - random.randrange(0, s_nodes)
    #                 print("Target Node is: ", self.target_node)
    #
    #             # Assigns Colour of nodes based on constant key
    #             for k in range(s_nodes):
    #                 self.colour_map.append(constants.NODE_COLOURS[i])
    #
    #             # Adds Subgraph to final graph
    #             self.graph = nx.compose(self.graph, subgraph)
    #
    #     # fix redundant nodes
    #     self.colour_map = self.colour_map[:self.total_nodes]
    #     i = len(self.graph.nodes())
    #     while i > self.total_nodes:
    #         self.graph.remove_node(i-1)
    #         del self.pos[i-1]
    #         i -= 1
    #
    #     # Defines Nodes for whole graph
    #     nx.set_node_attributes(self.graph, attr)
    #
    #     # Connect the graph
    #     def get_other_node(node_list, node_degrees, other_node):
    #         n = random.choices(node_list, weights=node_degrees, k=1)[0]
    #         if n == other_node:
    #             return get_other_node(node_list, node_degrees, other_node)
    #         return n
    #
    #     while not nx.is_connected(self.graph):
    #         node_layers = nx.get_node_attributes(self.graph, "layer")
    #         for i in range(self.layers - 1):
    #             node_a = [n for n in node_layers if node_layers[n] == i]
    #             degree_node_a = [self.graph.degree(n) for n in node_a]
    #             node_b = [n for n in node_layers if node_layers[n] == i + 1]
    #             degree_node_b = [self.graph.degree(n) for n in node_b]
    #
    #             n_a1 = random.choices(node_a, weights=degree_node_a, k=1)[0]
    #             if not nx.is_connected(self.graph.subgraph(node_a + node_b)):
    #                 print(node_b)
    #                 n_b = random.choices(node_b, weights=degree_node_b, k=1)[0]
    #                 self.graph.add_edge(n_a1, n_b)
    #             if random.random() < prob_inter_layer_edge and subnets_per_layer[i] > 1 and not nx.is_connected(
    #                     self.graph.subgraph(node_a)):
    #                 n_a2 = get_other_node(node_a, degree_node_a, n_a1)
    #                 self.graph.add_edge(n_a1, n_a2)
    #
    #     # Updates Colour of target node to red
    #     if self.target_node:
    #         self.colour_map[self.target_node] = "red"
    #
    #     endpoint_nodes_list = self.exposed_endpoints
    #     blank_endpoints = []
    #
    #     # Remove edges between endpoint nodes (not needed since adversary can reach them all anyway)
    #     # Store all external nodes with no internal nodes into blank_endpoints
    #     for n in endpoint_nodes_list:
    #         neighbors = list(self.graph.neighbors(n))
    #         for neighbor in neighbors:
    #             if neighbor in endpoint_nodes_list:
    #                 self.graph.remove_edge(n, neighbor)
    #         internal_connection = list(self.graph.neighbors(n))
    #         if not internal_connection:
    #             blank_endpoints.append(n)
    #
    #     # Pulls from blank_endpoints until all exposed endpoints are connected
    #     node_layers = nx.get_node_attributes(self.graph, "layer")
    #     layer1_nodes = [n for n in node_layers if node_layers[n] == 1]
    #     layer1_weights = [self.graph.degree(n) for n in layer1_nodes]
    #     while len(blank_endpoints):
    #         endpoint = blank_endpoints.pop(0)
    #         other_node = random.choices(layer1_nodes, weights=layer1_weights, k=1)[0]
    #         self.graph.add_edge(endpoint, other_node)
    #
    #     # Fix positions for endpoints
    #     for n in range(self.total_endpoints):
    #         position = (n + 1) / self.total_endpoints * (max_y_pos - min_y_pos) + min_y_pos
    #         new_pos = {n: np.array([0, position])}
    #         self.pos.update(new_pos)
    #
    #     # handling database nodes
    #     database_nodes_list = self._database
    #     blank_database = []
    #     for n in database_nodes_list:
    #         neighbors = list(self.graph.neighbors(n))
    #         for neighbor in neighbors:
    #             if neighbor in database_nodes_list:
    #                 self.graph.remove_edge(n, neighbor)
    #         internal_connection = list(self.graph.neighbors(n))
    #         if not internal_connection:
    #             blank_database.append(n)
    #
    #     # Pulls from blank_database until all database nodes are connected
    #     second_last_layer_nodes = [n for n in node_layers if node_layers[n] == self.layers - 2]
    #     second_last_layer_weights = [self.graph.degree(n) for n in second_last_layer_nodes]
    #     while len(blank_database):
    #         endpoint = blank_database.pop(0)
    #         other_nodes = random.choices(second_last_layer_nodes, weights=second_last_layer_weights, k=5)
    #         for node in other_nodes:
    #             self.graph.add_edge(endpoint, node)
    #
    #     # Fix positions for endpoints
    #     for n in range(self.total_nodes - self.total_database, self.total_nodes):
    #         position = (self.total_nodes - n) * 1 / self.total_database * 10
    #         new_pos = {n: np.array([(self.layers + 2) * 2.25, position])}
    #         self.pos.update(new_pos)
    #
    #     # Update Nodes Per Layer for Complete topology shuffling
    #     self.node_per_layer = nodes_per_layer.copy()
    #     self.node_per_layer[0] = self.total_endpoints
    #     self.node_per_layer[-1] = self.total_database
    #
    #
    #     # print("Endpoint list:", self.total_endpoints)
    #     # print("Node list:", self.graph.nodes)
    #     # print("Exposed Endpoint list: ", self.exposed_endpoints)
    #     # print("nodes per layer: ", self.node_per_layer)

    def get_total_endpoints(self):
        return self.total_endpoints

    def get_exposed_endpoints(self):
        return self.exposed_endpoints

    def get_database(self):
        return self._database

    def get_total_database(self):
        return self.total_database

    def get_scorer(self):
        return self.scorer

    def get_statistics(self):
        return self.scorer.get_statistics()

    def get_service_generator(self):
        return self.service_generator

    def get_hosts(self):
        return dict(nx.get_node_attributes(self.graph, "host"))

    def get_subnets(self):
        return dict(nx.get_node_attributes(self.graph, "subnet"))

    def get_layers(self):
        return dict(nx.get_node_attributes(self.graph, "layer"))

    def get_graph(self):
        return self.graph

    def get_graph_copy(self):
        return self.graph.copy()

    def get_pos(self):
        return self.pos

    def get_colourmap(self):
        return self.colour_map

    def get_total_nodes(self):
        return self.total_nodes

    def get_network_type(self):
        return self.network_type

    def get_unique_subnets(self):
        subnets = self.get_subnets()
        layers = self.get_layers()

        layer_subnets = {}

        for host_id, subnet_id in subnets.items():
            layer_id = layers[host_id]

            if not layer_id in layer_subnets:
                layer_subnets[layer_id] = {}

            layer_subnets[layer_id][subnet_id] = layer_subnets[layer_id].get(subnet_id, []) + [host_id]

        return layer_subnets

    def get_reachable(self):
        """
        Returns:
            The reachable array
        """
        return self.reachable

    def get_node_per_layer(self):
        """
        Returns:
            Number of nodes per layer
        """
        return self.node_per_layer

    def get_users_list(self):
        return self.users_list

    def get_users_per_host(self):
        return self.users_per_host

    def get_target_node(self):
        return self.target_node

    def add_attack_path_exposure(self):
        """
        Adds the Attack Path Exposure Score to statistics
        """
        self.scorer.add_attack_path_exposure(self.attack_path_exposure())

    def add_shortest_path(self):
        """
            Shortest Attack Path Variability (SAPV): changes on shortest attack paths over time
        """
        shortest_path = self.get_path_from_exposed(self.target_node, self.graph)[0]
       
        self.scorer.add_shortest_path(shortest_path)



    def attack_path_exposure(self):
        """
        Gets the total attack path exposure, scoring each node based on the % of new vulnerabilities found in each node on the shortest path to the target_node out of 1

        Returns:
            ave_score: Score of each host added up, divided by the number of hosts
        """
        shortest_path = self.get_path_from_exposed(self.target_node, self.graph)[0]
        vuln_list = []
        total_score = 0
        for host_id in shortest_path:
            host = self.get_host(host_id)
            service_id_list = host.get_path_from_exposed()
            services = host.get_services_from_list(service_id_list)
            not_unique_host_vulns = 0
            total_host_vulns = 0

            for service in services:
                vulns = service.get_vulns()
                total_host_vulns = len(vulns) + total_host_vulns

                for vuln in vulns:
                    if vuln not in vuln_list:
                        vuln_list.append(vuln)
                    else:
                        not_unique_host_vulns = not_unique_host_vulns + 1
            if total_host_vulns - not_unique_host_vulns == 0:
                new_vuln_percent = 0
            else:
                new_vuln_percent = (total_host_vulns - not_unique_host_vulns) / total_host_vulns
            total_score = total_score + new_vuln_percent
        if len(shortest_path) > 0:
            return total_score / len(shortest_path)
        else:
            return total_score

    def setup_users(self, user_to_nodes_ratio, prob_user_reuse_pass, users_per_host):
        """
        Randomly generates users that use the network

        Parameters:
            user_to_nodes_ratio:
                the percent of users in comparison to host machines.
                each node will then be given `int(1/user_to_nodes_ratio)` users each (less users more users on each computer).
            prob_user_reuse_pass:
                the probability that a user has reused their password.
            users_per_host:
                how many users are allocated to each host on the network.
        """
        self.total_users = int(self.total_nodes * user_to_nodes_ratio)
        if self.total_users < 1:
            self.total_users = 1

        names = [x.decode() for x in pkg_resources.resource_string('mtdnetwork', "data/first-names.txt").splitlines()]

        random_users = random.choices(names, k=self.total_users)
        self.users_list = [
            (user, random.random() < prob_user_reuse_pass)
            for user in random_users
        ]

        self.users_per_host = users_per_host

    def update_reachable_mtd(self):
        """
        Updates the Reachable array with only compromised nodes that are reachable after MTD
        NOTE: Probably can be optimised for speed
        """
        self.reachable = self.exposed_endpoints.copy()
        compromised_neighbour_nodes = []

        # Appends all neighbouring hosts from endpoints
        for endpoint in self.exposed_endpoints:
            visible_hosts = list(self.graph.neighbors(endpoint))
            for host in visible_hosts:
                for c_host in self.compromised_hosts:
                    if host == c_host:
                        compromised_neighbour_nodes.append(host)
                        self.reachable.append(host)

        # Checks if neighbouring hosts of compromised node are also compromised, if so add them to the list
        while len(compromised_neighbour_nodes) != 0:
            appended_host = compromised_neighbour_nodes.pop(0)
            visible_hosts = list(self.graph.neighbors(appended_host))
            for host in visible_hosts:
                for c_host in self.compromised_hosts:
                    if host == c_host:
                        if host not in self.reachable:
                            compromised_neighbour_nodes.append(host)
                            self.reachable.append(host)
                        # repeated = False
                        # for reachable in self.reachable:
                        #     if reachable == host:
                        #         repeated = True
                        # if repeated == False:
                        #     compromised_neighbour_nodes.append(host)
                        #     self.reachable.append(host)

    def update_reachable_compromise(self, compromised_node_id, compromised_hosts):
        """
        Updates the Reachable with the node_id of the compromised node
        """
        self.reachable.append(compromised_node_id)
        appended_host = compromised_node_id
        self.compromised_hosts = compromised_hosts
        all_reachable_hosts_added = False
        compromised_neighbour_nodes = []

        # Checks if neighbouring hosts of compromised node are also compromised, if so add them to the list
        while all_reachable_hosts_added == False:
            visible_hosts = list(self.graph.neighbors(appended_host))
            for host in visible_hosts:
                for c_host in compromised_hosts:
                    if host == c_host:
                        # repeated = False
                        # for reachable in self.reachable:
                        #     if reachable == host:
                        #         repeated = True
                        # if repeated == False:
                        if host not in self.reachable:
                            compromised_neighbour_nodes.append(host)
                            self.reachable.append(host)

            if len(compromised_neighbour_nodes) == 0:
                all_reachable_hosts_added = True
            else:
                appended_host = compromised_neighbour_nodes.pop(0)

    def get_host_id_priority(self, host_id):
        """
        Assign priority of host_id based on layer

        Parameters:
            host_id: node id of the desired node

        Returns:
            Priority: An integer based on tag_priority array, with target node scoring 0, top priority node scoring 1, and subsequent nodes scoring 1 higher
        """
        if host_id == self.target_node:
            return 0
        layers = self.get_layers()
        host_layer = layers.get(host_id)
        priority = -1
        i = 0
        for tag in self.tag_priority:
            if self.tags[host_layer] == tag:
                priority = i
            i += 1
        return priority + 1

    def assign_tags(self):
        """
        Assigns the tags to layers from constants.py
        """
        i = 0
        while i < self.layers:
            self.tags.append(constants.HOST_TAGS[i])
            i += 1

    def assign_tag_priority(self):
        """
        Orders tags based on priority
        """
        i = 0
        order = []
        while i < self.layers:
            dist = abs(self.target_layer - i)
            order.append(dist)
            i += 1

        layer_index = 0
        priority = 0
        order_index = 0
        while layer_index < self.layers:
            for order_prio in order:
                if order_prio == priority:
                    self.tag_priority.append(self.tags[order_index])
                    layer_index += 1
                order_index += 1
            priority += 1
            order_index = 0

    def get_path_from_exposed(self, target_node, graph=None):
        """
        Gets the shortest path and distance from the exposed endpoints.

        Can also specify a subgraph to use for finding

        Parameters:
            target_node:
                the target node to reach to

        Returns:
            a tuple where the first element is the shortest path and the second element is the distance
        """
        if graph is None:
            graph = self.graph

        shortest_distance = constants.LARGE_INT
        shortest_path = []

        for ex_node in self.exposed_endpoints:
            try:
                path = nx.shortest_path(graph, ex_node, target_node)
                path_len = len(path)

                if path_len < shortest_distance:
                    shortest_distance = path_len
                    shortest_path = path
            except:
                pass

        # This function is used when the attacker can't find a path to host

        # if shortest_distance == constants.LARGE_INT:
        #     raise exceptions.ActionBlockedError

        return shortest_path, shortest_distance

    def get_shortest_distance_from_exposed_or_pivot(self, host_id, pivot_host_id=-1, graph=None):
        if host_id in self.exposed_endpoints:
            return 0
        if graph is None:
            graph = self.graph
        shortest_distance = self.get_path_from_exposed(host_id, graph=graph)[1]
        if pivot_host_id >= 0:
            try:
                path = nx.shortest_path(graph, host_id, pivot_host_id)
                path_len = len(path)

                if path_len < shortest_distance:
                    shortest_distance = path_len
            except:
                pass

        return shortest_distance

    def sort_by_distance_from_exposed_and_pivot_host(self, host_stack, compromised_hosts, pivot_host_id=-1):
        """
        Sorts the Host Stack by the shortest number of hops to reach the target hosts.

        Parameters:
            host_stack:
                a list of host IDs the attacker wants to attack
            compromised_hosts:
                a list of host IDs the hacker has compromised
            pivot_host_id:
                the ID of the host that is compromised that the hacker is using to pivot from.
                if None then it only sorts by the exposed endpoints
        """

        visible_network = self.get_hacker_visible_graph()

        non_exposed_endpoints = [
            host_id
            for host_id in host_stack
            if not host_id in self.exposed_endpoints
        ]

        return sorted(
            non_exposed_endpoints,
            key=lambda host_id: self.get_shortest_distance_from_exposed_or_pivot(
                host_id,
                pivot_host_id=pivot_host_id,
                graph=visible_network
            ) + random.random()
        ) + [
                   host_id
                   for host_id in self.exposed_endpoints
                   if host_id in host_stack
               ]

    def get_neighbors(self, host_id):
        """
        Returns the neighbours for a host.

        Parameters:
            host_id:
                the host ID to get the neighbors from

        Returns:
            a list of the neighbors for the host.
        """
        return list(self.graph.neighbors(host_id))

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

    def get_hacker_visible_graph(self):
        """
        Returns the Network graph that is visible to the hacker depending on the hosts that have already been compromised

        """
        visible_hosts = []
        for c_host in self.reachable:
            visible_hosts = visible_hosts + list(self.graph.neighbors(c_host))

        visible_hosts = visible_hosts + self.reachable
        visible_hosts = visible_hosts + self.exposed_endpoints

        return self.graph.subgraph(
            list(set(visible_hosts))
        )

    def get_host(self, host_id):
        """
        Gets the Host instance based on the host_id

        Parameters:
            the ID of the Host Instance

        Returns:
            the corresponding Host instance
        """

        return self.graph.nodes.get(host_id, {}).get("host", None)

    def get_total_vulns(self):
        return self.total_vulns

    def record_current_total_vulns(self):
        for host_id in self.nodes:
            host = self.get_host(host_id)
            vulns = host.get_all_vulns()
            self.total_vulns += len(vulns)
        self.total_vulns_lst.append(self.total_vulns)
    
    def record_current_vuln_dict(self):
        self.vuln_dict_lst.append(self.get_vuln_dict())

    def get_vuln_dict(self):
        """
        Gets all the vulnerabilities for every hosts and puts them in vuln_dict

        Returns:
            the freuqency of every vuln
        """
        for host_id in self.nodes:
            host = self.get_host(host_id)
            vulns = host.get_all_vulns()
            self.total_vulns += len(vulns)
            self.vuln_dict[host_id] = vulns
            for v in vulns:
                v_id = v.get_id()
                if v_id in self.vuln_count:
                    self.vuln_count[v.get_id()] += 1
                else:
                    self.vuln_count[v.get_id()] = 1
        
        return self.vuln_count

    def get_total_services(self):
        return self.total_services

    def get_service_dict(self):
        """
        Gets all the services for every hosts and puts them in service_dict

        Returns:
            the freuqency of every service
        """
        for host_id in self.nodes:
            host = self.get_host(host_id)
            services = host.get_all_services()
            self.total_services += len(services)
            self.service_dict[host_id] = services
            for s in services:
                s_id = s.get_id()
                if s_id in self.service_count:
                    self.service_count[s.get_id()] += 1
                else:
                    self.service_count[s.get_id()] = 1
        return self.service_count

    def is_target_compromised(self):
        if self.get_host(self.target_node).is_compromised():
            return True
        else:
            return False

    def is_compromised(self, compromised_hosts):
        """
        Checks if the Network has been completely compromised.

        Parameters:
            compromised_hosts:
                the list of host IDs that have been compromised by the hacker

        Returns:
            boolean
        """
        return len(compromised_hosts) == self.total_nodes

    def draw(self):
        plt.figure(1, figsize=(15, 12))
        nx.draw(self.graph, pos=self.pos, node_color=self.colour_map, with_labels=True)
        directory = os.getcwd()
        plt.savefig(directory + '/experimental_data/plots/network.png')
