import random, uuid
import networkx as nx
import mtdnetwork.data.constants as constants


class Host:
    def __init__(self, operating_system, os_version, host_id, host_ip, users_list,
                 network, service_generator, k_nearest_neighbors_percent=0.5,
                 prob_strogatz_rewire=0.5):
        """
        Initialises the host with the specified Operating System with a random network of internal
        services that the host runs.

        The internal service network for the Host maps what services the hacker needs to compromise
        in order to compromise the entire Host

        A host is said to be compromised if a service adjacent to the target node is compromised or
        the hacker has been able to obtain the password for a user on the Host

        Parameters:
            operating_system:
                the operating system of the host
            os_version:
                the OS version
            host_id:
                the ID of the host on the graph of hosts
            host_ip:
                the IP address of the host
            users_list:
                a tuple list of Users on the Host that also specify if the user reuses their password
                eg. if Josh is added to this host and they reused their pass then users_list would be
                [("Josh", True)]
            network:
                the Network instance of the simulation
            service_generator:
                the ServiceGenerator instance for generating new services
            k_nearest_neighbors_percent:
                the ratio for the k nearest neighbors when generating the internal graph using Watts-Strogatz random graph
            prob_strogatz_rewire:
                the probability that an edge is rewired for the Watts-Strogatz random graph
        """
        self.graph = None
        self.os_type = operating_system
        self.os_version = os_version
        self.ip = host_ip
        self.host_id = host_id
        self.network = network
        self.p_u_compromise = False
        self.total_users = 0
        self.uuid = str(uuid.uuid4())

        self.total_services = random.randint(constants.HOST_SERVICES_MIN, constants.HOST_SERVICES_MAX)
        # +1 for the target service node the adversary needs to be adjacent inorder to compromise the host
        self.total_nodes = self.total_services + 1
        self.compromised = False
        self.compromised_services = []
        self.gen_internal_network(
            k_nearest_neighbors_percent,
            prob_strogatz_rewire
        )
        self.setup_network(service_generator)
        self.set_host_users(users_list)
       

    def is_exposed_endpoint(self):
        return self.host_id in self.network.exposed_endpoints

    def __eq__(self, other):
        if not isinstance(other, Host):
            return False
        return other.uuid == self.uuid

    def get_all_services(self):
        return list(dict(nx.get_node_attributes(self.graph, "service")).values())

    def get_test_values(self):
        return [(nx.get_node_attributes(self.graph, "port")), (nx.get_node_attributes(self.graph, "service"))]

    def get_total_nodes(self):
        return self.total_nodes

    def get_all_vulns(self):
        all_services = self.get_all_services()

        total_vulns = 0

        all_vulns = []
        for service in all_services:
            service_vulns = service.get_all_vulns()
            total_vulns += len(service_vulns)

            for v in service_vulns:
                if not v in all_vulns:
                    all_vulns.append(v)
        # print("Services on this host has average of this many vulns: ", total_vulns/len(all_services))

        return all_vulns

    def get_vulns_for_list(self, list):

        vulns = []
        for service in list:
            service_vulns = service.get_all_vulns()

            for v in service_vulns:
                if not v in vulns:
                    vulns.append(v)

        return vulns

    def swap_network(self, network):
        self.network = network

    def setup_network(self, service_generator, keep_ports=False):
        port_addresses = []
        for node_id in range(self.total_nodes):
            if node_id == self.target_node:
                continue
            if not keep_ports:
                node_port = Host.get_random_port(existing_ports=port_addresses)
                self.graph.nodes[node_id]["port"] = node_port
            self.graph.nodes[node_id]["service"] = service_generator.get_random_service(
                self.os_type,
                self.os_version
            )

    def set_compromised(self):
        """
        Sets the host as compromised
        """
        self.compromised = True
        # self.change_node_color(color="red")

    def possible_user_compromise(self):
        """
        Returns:
            true if there is a user on the host that has reused their password.
        """
        return self.p_u_compromise

    def can_auto_compromise_with_users(self, compromised_users):
        """
        Creates an action where the adversary tries to compromise the host using a reused password

        Parameters:
            compromised_users:
                a list of users the adversary has compromised

        Returns:
            an action that will return if the adversary was succesfully in finding a user with a reused password
        """
        if self.possible_user_compromise():
            c_reused_comp = True in [reused_pass for (username, reused_pass) in self.users.items() if
                                     username in compromised_users]
            if c_reused_comp:
                self.set_compromised()

            return c_reused_comp
        return False

    def is_compromised(self):
        return self.compromised

    def compromise_with_users(self, compromised_users):
        """
        Tries to brute force a login using the list of users that the adversary has compromised

        Parameters:
            compromised_users:
                the list of users the hacker has compromised

        Returns:
            an action that returns if the brute force worked
        """
        attempt_users = [username for username in self.users.keys() if username in compromised_users]

        if random.random() < constants.HOST_MAX_PROB_FOR_USER_COMPROMISE * len(attempt_users) / self.total_users:
            self.set_compromised()
            return True
        return False

    def get_service_and_vulns(self):
        """
        Returns a dictionary of all non-target nodes and their vulnerabilities
        """
        services = nx.get_node_attributes(self.graph, "service")
        return {
            service_id: service.get_vulns()
            for (service_id, service) in services.items()
        }

    def get_target_node(self):
        return self.target_node

    def get_exposed_nodes(self):
        return self.exposed_endpoints

    def remove_no_vuln_nodes(self):
        """
        Removes all service nodes with no vulnerabilities
        EDIT: Removed due to assigning a vulnerability to every service (Random element of adding vulnerabilites caused many graphs to break)
        """
        vulns = self.get_service_and_vulns()
        blank_endpoints = []
        blank_endpoints_index = 0

        for key, value in vulns.items():
            if not value:
                blank_endpoints.append(key)
        self.graph.remove_nodes_from(blank_endpoints)

        if blank_endpoints:
            for n in range(self.total_nodes):
                if n == blank_endpoints[blank_endpoints_index]:
                    self.colour_map.pop(n - blank_endpoints_index)
                    blank_endpoints_index += 1
                    if blank_endpoints_index == len(blank_endpoints):
                        blank_endpoints_index = 0
                    if n in self.exposed_endpoints:
                        self.exposed_endpoints.remove(n)

        print("Blank endpoints: ", blank_endpoints, "target node: ", self.target_node)
        self.total_nodes = self.total_nodes - len(blank_endpoints)

    def get_services(self, just_exploited=False):
        """
        Gets the services on the host

        Parameters:
            just_exploited:
                only return the services that have been exploited

        Returns:
            a dict where the key is the service ID and the value is the Service instance
        """
        services = nx.get_node_attributes(self.graph, "service")
        if just_exploited:
            return {
                service_id: service
                for (service_id, service) in services.items()
                if service.is_exploited()
            }
        exposed_services = [
            host_id
            for (host_id, service) in services.items()
            if host_id in self.exposed_endpoints or service.is_exploited()
        ]

        adjacent_services = []
        for ec_service_id in exposed_services:
            if not services[ec_service_id].is_exploited():
                continue
            for n_id in self.graph.neighbors(ec_service_id):
                if n_id == self.target_node: continue
                if not n_id in exposed_services and not n_id in adjacent_services:
                    adjacent_services.append(n_id)

        return {
            service_id: services[service_id]
            for service_id in exposed_services + adjacent_services
        }

    def get_services_from_list(self, list):
        """
        Gets Service instance from a list of service numbers of host

        Parameters:
            list:
                list of shortest path to target node (including target node)

        """
        target = list.pop(len(list) - 1)
        service_list = []
        for service_id in list:
            service = self.graph.nodes.get(service_id, {}).get("service", None)
            service_list.append(service)
        return service_list

    def get_services_from_ports(self, discovered_service_ports, ignore_services=[]):
        """
        Gets the services from the discovered ports

        Returns:
            discovered_service_ports:
                a list of ports that have been discovered
            ignore_services:
                a list of service IDs that will be ignored
        """
        discovered_service_ports = discovered_service_ports + self.exposed_endpoints
        exposed_services = self.get_services()
        port_numbers = nx.get_node_attributes(self.graph, "port")
        shortest_path_to_target = dict(nx.single_target_shortest_path_length(
            self.graph,
            self.target_node
        ))

        result = [
            {
                "service_id": host_id,
                "port": port_numbers[host_id],
                "service": exposed_services[host_id]
            }
            for host_id in exposed_services
            if port_numbers[host_id] in discovered_service_ports and host_id not in ignore_services
        ]

        return sorted(
            result,
            key=lambda host_dict: (
                shortest_path_to_target[host_dict["service_id"]],
                host_dict["service"].get_highest_roa_vuln()
            ),
            reverse=True
        )

    def port_scan(self):
        port_numbers_dict = nx.get_node_attributes(self.graph, "port")
        services = nx.get_node_attributes(self.graph, "service")
        port_numbers = []

        service_q = []
        seen = []
        for ec_service_id in self.exposed_endpoints:
            service_q.append(ec_service_id)
            seen.append(ec_service_id)
            port_numbers.append(port_numbers_dict[ec_service_id])

        while len(service_q) > 0:
            service_id = service_q.pop(0)
            service = services[service_id]

            if not service_id in self.exposed_endpoints:
                port_numbers.append(port_numbers_dict[service_id])

            if service.is_exploited():
                for n in self.graph.neighbors(service_id):
                    if n == self.target_node: continue
                    if n in seen: continue
                    service_q.append(n)
                    seen.append(n)
        return port_numbers

    def get_vulns(self, discovered_service_ports, ignore_services=[], roa_threshold=0):
        """
        Gets the possible vulnerabilities on the Host based on the discovered port numbers

        Parameters:
            discovered_service_ports:
                a list of ports found open doing a port scan
            ignore_services:
                the list of service IDs to ignore when searching for vulnerabilities
            roa_threshold:
                the return of attack (RoA) threshold that the adversary users for sorting the exploits they want to use

        Returns:
            a hacker action for discovering if the vulnerabilities are present on the host and a possible method of exploitation
        """
        services_dict = self.get_services_from_ports(discovered_service_ports, ignore_services=ignore_services)
        vulns = []
        discovery_time = 0
        for service_dict in services_dict:
            service = service_dict["service"]
            vulns = vulns + service.get_vulns(roa_threshold=roa_threshold)
            discovery_time += service.discover_vuln_time(roa_threshold=roa_threshold)

        new_vulns = []
        for vuln in vulns:
            if vuln.has_dependent_vulns:
                if vuln.can_exploit_with_dependent_vuln(vulns):
                    new_vulns.append(vuln)
            else:
                new_vulns.append(vuln)
        return new_vulns

    # def total_exploit_time(self, vulns):
    #     exploit_time = 0
    #     for vuln in vulns:
    #         exploit_time += vuln.exploit_time()
    #     return exploit_time

    # def exploit_vulns(self, vulns):
    #     """
    #     Tries exploiting the list of vulnerabilities that the hacker is trying to exploit on the Host
    #
    #     Parameters:
    #         vulns:
    #             the list of vulnerabilities that are trying to be exploited
    #
    #     Returns:
    #         an action if that returns a dictionary of if a new service was exploited and if the host is compromised
    #     """
    #     exploit_time = 0
    #     for vuln in vulns:
    #         vuln.network(host=self)
    #         exploit_time += vuln.exploit_time()
    #
    #     return self.check_compromised()

    def check_compromised(self):
        """
        check if the current host is compromised
        """
        services = self.get_services(just_exploited=True)
        for service_id in services:
            if service_id not in self.compromised_services:
                self.compromised_services.append(service_id)
                self.colour_map[service_id] = "red"
            if self.target_node in list(self.graph.neighbors(service_id)):
                self.set_compromised()
        return self.compromised

    def discover_neighbors(self):
        """
        Scans for neighbors of this Host

        Returns:
            an action that returns a list a neighboring host ids
        """
        neighbors = list(self.network.graph.neighbors(self.host_id))
        return neighbors

    def get_ports(self):
        """
        Returns all of the ports on the host
        """
        port_map = nx.get_node_attributes(self.graph, "port")
        return sorted([
            port_map[s]
            for s in port_map
        ], reverse=True)

    def get_ports_for_services(self, services):
        port_map = nx.get_node_attributes(self.graph, "port")
        return sorted([
            port_map[s]
            for s in services
        ], reverse=True)

    def get_os_type_and_version(self):
        return self.os_type, self.os_version

    def get_compromised_users(self):
        if not self.compromised: return []
        return self.get_users()

    def get_users(self):
        return list(self.users.keys())

    def change_node_color(self, color="red"):
        """
        Changes the color of the host on the overall network
        """
        i = 0
        host_index = -1
        while i < len(self.network.colour_map):
            if self.network.colour_map[i] == self.host_id:
                host_index = i
            i += 1
        self.network.colour_map[host_index] = color
        pass

    def set_host_users(self, users_list):
        """
        Sets the users on the Host

        Parameters:
            users_list:
                the tuple users list specifying the username and if the user reuses their password
        """
        self.users = {
            user_tuple[0]: user_tuple[1]
            for user_tuple in users_list
        }
        for user_reuse in self.users.values():
            self.total_users += 1
            if user_reuse:
                self.p_u_compromise = True
                break

    def graph_choose_target_and_exposed(self, graph, target_distance):
        """
        Chooses the target and exposed services on the internal network.

        Exposed services means that an adversary can see the ports open externally

        Parameters:
            graph:
                the graph that is being generated
            target_distance:
                the target distance from the target node for exposed services
        """
        shortest_path_length = dict(nx.all_pairs_shortest_path_length(graph))
        nodes_list = list(graph.nodes)

        exposed_endpoints, adjacent_to_target = [], []
        target_node = -1
        exposed_endpoints = []
        target_count = -1

        for x in nodes_list:
            x_count = 0
            e_endpoints = []
            for y in nodes_list:
                if shortest_path_length[x][y] >= target_distance:
                    x_count += 1
                    e_endpoints.append(y)

            if x_count > target_count:
                target_node = x
                target_count = x_count
                exposed_endpoints = e_endpoints

        if target_node in exposed_endpoints:
            exposed_endpoints.remove(target_node)

        return shortest_path_length, target_node, exposed_endpoints, adjacent_to_target

    def get_exposed_endpoints(self):
        return self.exposed_endpoints

    def gen_internal_network(self, k_nearest_neighbors_percent, p):
        k = int(self.total_services * k_nearest_neighbors_percent)
        if k < 2:
            k = 2

        self.graph = nx.connected_watts_strogatz_graph(self.total_nodes, k, p)
        results = self.graph_choose_target_and_exposed(self.graph, nx.diameter(self.graph) - 1)

        self.shortest_path_length = results[0]
        self.target_node = results[1]
        self.exposed_endpoints = results[2]
        self.adjacent_to_target = results[3]

        other_nodes = [
            node_id for node_id in range(self.total_nodes)
            if node_id != self.target_node and not node_id in self.exposed_endpoints
        ]

        for o1_node in other_nodes:
            exposed_neighbors = [n for n in self.graph.neighbors(o1_node) if n in self.exposed_endpoints]
            if len(exposed_neighbors) == 0:
                self.graph.add_edge(o1_node, random.choice(self.exposed_endpoints))

            for o2_node in other_nodes:
                if o1_node == o2_node: continue
                if self.graph.has_edge(o1_node, o2_node):
                    self.graph.remove_edge(o1_node, o2_node)
                self.graph.add_edge(o1_node, self.target_node)
                self.graph.add_edge(o2_node, self.target_node)

        for e1_node in self.exposed_endpoints:
            for e2_node in self.exposed_endpoints:
                if e1_node == e2_node: continue
                if self.graph.has_edge(e1_node, e2_node):
                    self.graph.remove_edge(e1_node, e2_node)
                    if len(other_nodes) > 0:
                        o1_node = random.choice(other_nodes)
                        o2_node = random.choice(other_nodes)
                        if not self.graph.has_edge(e1_node, o1_node):
                            self.graph.add_edge(e1_node, o1_node)

                        if not self.graph.has_edge(e2_node, o2_node):
                            self.graph.add_edge(e2_node, o2_node)
                    else:
                        if nx.is_isolate(self.graph, e1_node):
                            self.graph.add_edge(e1_node, self.target_node)
                        if nx.is_isolate(self.graph, e2_node):
                            self.graph.add_edge(e2_node, self.target_node)

        self.colour_map = []
        for node in list(self.graph.nodes):
            if node == self.target_node:
                self.colour_map.append("yellow")
            elif node in self.exposed_endpoints:
                self.colour_map.append("green")
            else:
                self.colour_map.append("blue")

    def draw(self):
        """
        Draws the internal network for the Host
        """
        return nx.draw_networkx(self.graph, node_color=self.colour_map, with_labels=True)

    @staticmethod
    def get_random_os_version(operating_system):
        """
        Gets a random Operating System version.

        The latest versions are more likely to be chosen over older and outdated OS versions.

        Parameters:
            operating_system:
                the name of the operating system to pick a random version for

        Returns:
            a random version where the latest versions of the OS are more likely to be chose over older ones
        """
        versions = constants.OS_VERSION_DICT[operating_system]
        total_versions = len(versions)
        return random.choices(versions, weights=[total_versions - index for index in range(total_versions)], k=1)[0]

    @staticmethod
    def get_random_os():
        """
        Returns:
            A random OS
        """
        return random.choice(constants.OS_TYPES)

    @staticmethod
    def get_random_address(existing_addresses=None):
        """
        Gets a Random IP address and makes sure that one does not already exist.

        Parameters:
            existing_addresses:
                a list of already allocated IP addresses on the network
            
        Returns:    
            a IPv4 address
        """
        if existing_addresses is None:
            existing_addresses = []
        new_ip = "{}.{}.{}.{}".format(*[random.randint(1, 256) for _i in range(4)])
        if new_ip in existing_addresses:
            return Host.get_random_address(existing_addresses=existing_addresses)
        return new_ip

    @staticmethod
    def get_random_port(existing_ports=None):
        """
        Gets a random port while checking a port has already been allocated for a host

        Parameters:
            existing_ports:
                the port numbers that have already been allocated to the host

        Returns:
            a new port number for the new service
        """
        if existing_ports is None:
            existing_ports = []
        new_port = random.choice(constants.HOST_PORT_RANGE)
        if new_port in existing_ports:
            return Host.get_random_port(existing_ports=existing_ports)
        return new_port

    def get_path_from_exposed(self):
        """
        Gets the shortest path and distance from the exposed endpoints.

        Can also specify a subgraph to use for finding

        Returns:
            a tuple where the first element is the shortest path and the second element is the distance
        """
        graph = self.graph

        shortest_distance = constants.LARGE_INT
        shortest_path = []

        for ex_service in self.exposed_endpoints:
            try:
                path = nx.shortest_path(graph, ex_service, self.target_node)
                path_len = len(path)

                if path_len < shortest_distance:
                    shortest_distance = path_len
                    shortest_path = path
            except:
                pass

        # This function is used for sorting so shouldn't raise an exception
        # some MTD cause this exception to be raised.
        #

        return shortest_path
