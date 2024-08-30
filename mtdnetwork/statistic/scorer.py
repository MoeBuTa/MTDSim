# import mtdnetwork.irrelevant_stuff.exceptions as exceptions

import numpy as np
class Statistics:

    def __init__(self, record_type):
        """
        Statistics when events have occurred for a particular type of event.

        Parameters:
            record_type:
                the name of the type of record
        """
        self.record_type = record_type
        self.x_list = []
        self.y_list = []

    def add_event(self, x, y):
        self.x_list.append(x)
        self.y_list.append(y)

    def get_dict(self):
        cumul_y = [i + 1 for i in range(len(self.y_list))]
        return {
            "record name": self.record_type,
            "x": self.x_list,
            "y": self.y_list,
            "cumulative y": cumul_y,
            "total events": len(self.y_list)
        }

    def __str__(self):
        return self.record_type


class CompromiseStatistics(Statistics):

    def __init__(self, record_type):
        self.non_exposed_x = []
        self.non_exposed_y = []
        super().__init__(record_type)

    def add_event(self, x, host_instance):
        host_id = host_instance.host_id
        if not host_instance.is_exposed_endpoint():
            self.non_exposed_x.append(x)
            self.non_exposed_y.append(host_id)

        self.x_list.append(x)
        self.y_list.append(host_id)

    def get_dict(self):
        cumul_y = [i + 1 for i in range(len(self.y_list))]
        cumul_non_exposed_y = [i + 1 for i in range(len(self.non_exposed_y))]

        return {
            "record name": self.record_type,
            "x": self.x_list,
            "y": self.y_list,
            "cumulative y": cumul_y,
            "total events": len(self.y_list),
            "not exposed x": self.non_exposed_x,
            "not exposed y": self.non_exposed_y,
            "cumulative not exposed y": cumul_non_exposed_y,
            "total not exposed events": len(self.non_exposed_y)
        }


class VulnStatistics(Statistics):
    def __init__(self, record_type):
        self.roa_list = []
        self.impact_list = []
        self.complexity_list = []
        self.risk_list = []
        self.has_os_dependency = 0
        self.has_dependent_vulns = 0
        super().__init__(record_type)

    def add_event(self, curr_time, vuln):
        self.x_list.append(curr_time)
        self.roa_list.append(vuln.roa())
        self.impact_list.append(vuln.impact)
        self.complexity_list.append(vuln.complexity)
        self.risk_list.append(vuln.risk())
        self.has_os_dependency += 1 if vuln.has_os_dependency else 0
        self.has_dependent_vulns += 1 if vuln.has_dependent_vulns else 0

    def get_dict(self):
        cumulative_exploited_vulns = [i + 1 for i in range(len(self.x_list))]
        return {
            "record name": self.record_type,
            "x": self.x_list,
            "cumulative exploited vulns": cumulative_exploited_vulns,
            "roa": self.roa_list,
            "impact": self.impact_list,
            "risk": self.risk_list,
            "complexity": self.complexity_list,
            "total had os dependency": self.has_os_dependency,
            "total were dependent on another vuln": self.has_dependent_vulns
        }


class MTDStatistics(Statistics):

    def __init__(self, record_type):
        self.blocked_x_list = []
        self.blocked_y_list = []
        super().__init__(record_type)

    def add_blocked_event(self, x, y):
        self.blocked_x_list.append(x)
        self.blocked_y_list.append(y)

    def get_dict(self):
        cumul_y = [i + 1 for i in range(len(self.y_list))]
        cumul_blocked_y = [i + 1 for i in range(len(self.blocked_y_list))]
        return {
            "record name": self.record_type,
            "x": self.x_list,
            "y": self.y_list,
            "cumulative y": cumul_y,
            "total events": len(self.y_list),
            "blocked times": self.blocked_x_list,
            "total blocks": len(self.blocked_x_list),
            "blocked values": self.blocked_y_list,
            "cumulative blocked": cumul_blocked_y
        }


class Scorer:

    def __init__(self):
        self.host_compromises = CompromiseStatistics("Host Compromises")
        self.host_vuln_compromises = CompromiseStatistics("Vuln Compromises")
        self.host_reuse_pass_compromises = CompromiseStatistics("Reuse Password Compromises")
        self.host_pass_spray_compromises = CompromiseStatistics("Password Spray Compromises")
        self.user_account_leaks = Statistics("User Account Has Been Leaked By Compromise")
        self.attack_path_exposure = []
        self.shortest_path_record = []

        # Gather statistics on the types of vulnerabilities that were exploited
        # eg. RoA score, impact and complexity
        self.vuln_compromises = VulnStatistics("Vulnerabilities Exploited")

        self.last_mtd = None
        self.mtd_statistics = []

  
    def register_mtd(self, mtd_strategy):
        self.mtd_statistics.append(MTDStatistics(str(mtd_strategy)))

    def set_last_mtd(self, mtd_strategy):
        """
        Sets the last triggered MTD strategy for logging which MTD strategies blocked changes
        """
        mtd_name = str(mtd_strategy)
        for mtd_stat in self.mtd_statistics:
            if mtd_name == str(mtd_stat):
                self.last_mtd = mtd_stat
                return

        self.last_mtd = None

    # def add_mtd_blocked_event(self, curr_time):
    #     if not self.last_mtd == None:
    #         self.last_mtd.add_blocked_event(curr_time, str(self.last_mtd))
    #     else:
    #         raise exceptions.CannotAddMTDEventToScorerError

    # def add_mtd_event(self, curr_time):
    #     """
    #     Logs when an MTD event is triggered
    #     """
    #     if not self.last_mtd == None:
    #         self.last_mtd.add_event(curr_time, 1)
    #     else:
    #         raise exceptions.CannotAddMTDEventToScorerError

    def add_host_compromise(self, curr_time, host_instance):
        # host_os_type = host_instance.os_type
        # host_os_version = host_instance.os_version
        # host_type = "{} {}".format(host_os_type, host_os_version)
        self.host_compromises.add_event(curr_time, host_instance)

    def add_host_vuln_compromise(self, curr_time, host_instance):
        self.add_host_compromise(curr_time, host_instance)
        # host_os_type = host_instance.os_type
        # host_os_version = host_instance.os_version
        # host_type = "{} {}".format(host_os_type, host_os_version)
        self.host_vuln_compromises.add_event(curr_time, host_instance)

    def add_host_reuse_pass_compromise(self, curr_time, host_instance):
        self.add_host_compromise(curr_time, host_instance)
        # host_os_type = host_instance.os_type
        # host_os_version = host_instance.os_version
        # host_type = "{} {}".format(host_os_type, host_os_version)
        self.host_reuse_pass_compromises.add_event(curr_time, host_instance)

    def add_host_pass_spray_compromise(self, curr_time, host_instance):
        self.add_host_compromise(curr_time, host_instance)
        # host_os_type = host_instance.os_type
        # host_os_version = host_instance.os_version
        # host_type = "{} {}".format(host_os_type, host_os_version)
        self.host_pass_spray_compromises.add_event(curr_time, host_instance)

    def add_user_account_leak(self, curr_time, username):
        self.user_account_leaks.add_event(curr_time, username)

    def add_vuln_compromise(self, curr_time, vuln):
        self.vuln_compromises.add_event(curr_time, vuln)

    def add_attack_path_exposure(self, score):
        self.attack_path_exposure.append(score)

    def add_shortest_path(self, score):
        self.shortest_path_record.append(score)


    def set_initial_statistics(self, network):
        """
        Collects statistics on the initial state of the network
        """

        self.stats = {}

        hosts = network.get_hosts()

        total_vulns = 0

        host_os_type_and_version_vuln_roa = {}

        os_types_in_network = {}
        hosts_without_vulns = 0

        for host_id, host_instance in hosts.items():
            host_os = host_instance.os_type
            host_version = host_instance.os_version

            os_type = "{} {}".format(host_os, host_version)
            os_types_in_network[os_type] = os_types_in_network.get(os_type, 0) + 1
            host_vulns = host_instance.get_all_vulns()

            total_vulns += len(host_vulns)

            if len(host_vulns) == 0:
                hosts_without_vulns += 1

            for v in host_vulns:
                roa = v.initial_roa()
                if not host_os in host_os_type_and_version_vuln_roa:
                    host_os_type_and_version_vuln_roa[host_os] = {}

                if not v in host_os_type_and_version_vuln_roa[host_os].get(host_version, []):
                    host_os_type_and_version_vuln_roa[host_os][host_version] = host_os_type_and_version_vuln_roa[
                                                                                   host_os].get(host_version, []) + [
                                                                                   roa]

        vulns_per_os = {}
        avg_roa_per_os = {}

        for host_os in host_os_type_and_version_vuln_roa:
            if not host_os in vulns_per_os:
                vulns_per_os[host_os] = {}
                avg_roa_per_os[host_os] = {}

            for host_os_type in host_os_type_and_version_vuln_roa[host_os]:
                roa_list = host_os_type_and_version_vuln_roa[host_os][host_os_type]
                vulns_per_os[host_os][host_os_type] = len(roa_list)
                avg_roa_per_os[host_os][host_os_type] = sum(roa_list) / len(roa_list)

        self.stats["Total Initial Vulnerabilities (Sum of all Vulns on all hosts)"] = total_vulns
        self.stats["Total Unique Vulnerabilities"] = len(network.get_vuln_dict())
        self.stats["Initial Vulns Per OS (Sum of all Vulns on those OS)"] = vulns_per_os
        self.stats["Average Initial RoA Per OS"] = avg_roa_per_os
        self.stats["OS Types In Initial Network"] = os_types_in_network
        self.stats["Initial Hosts Without Vulnerabilities"] = hosts_without_vulns

    def get_statistics(self):
        stats = self.stats
        stats["Host Compromises"] = self.host_compromises.get_dict()
        stats["Vuln Compromises"] = self.host_vuln_compromises.get_dict()
        stats["Reuse Password Compromises"] = self.host_reuse_pass_compromises.get_dict()
        stats["Password Spray Compromises"] = self.host_pass_spray_compromises.get_dict()
        stats["User Account Leaks"] = self.user_account_leaks.get_dict()
        stats["Vulnerabilities Exploited"] = self.vuln_compromises.get_dict()
        stats["MTD Statistics"] = [
            mtd_statistic.get_dict()
            for mtd_statistic in self.mtd_statistics
        ]
        stats["Total MTD Events"] = sum([
            mtd_statistic.get_dict()["total events"]
            for mtd_statistic in self.mtd_statistics
        ])
        stats["Total MTD Blocking Hacker Events"] = sum([
            mtd_statistic.get_dict()["total blocks"]
            for mtd_statistic in self.mtd_statistics
        ])
        stats["Attack Path Exposure Scores"] = self.attack_path_exposure

        return stats
