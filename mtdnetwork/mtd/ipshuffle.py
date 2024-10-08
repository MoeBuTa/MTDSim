from mtdnetwork.mtd import MTD
from mtdnetwork.component import host


class IPShuffle(MTD):

    def __init__(self, network=None):
        super().__init__(name="IPShuffle",
                         mtd_type='shuffle',
                         resource_type='network',
                         network=network)

    def mtd_operation(self, adversary=None):
        hosts = self.network.get_hosts()

        ip_addresses = []
        for host_id, host_instance in hosts.items():
            if host_id in self.network.exposed_endpoints:
                continue
            host_ip = host.Host.get_random_address(existing_addresses=ip_addresses)
            ip_addresses.append(host_ip)
            host_instance.ip = host_ip

        #update the new ip addresses
        self.network.update_current_hosts_ip(ip_addresses)
