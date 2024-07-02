import random, logging
import mtdnetwork.data.constants as constants
import pkg_resources
import uuid


class Vulnerability:
    def __init__(self, can_have_os_dependency=False, os_list=[]):
        """
        Creates a vulnerability that is assigned to a set of versions for a service.

        complexity: if set to 1 then the vulnerability is trivial to exploit, 0 for being impossible
        impact: if 1 then exploiting the vulnerability is the most severe, 0 for not doing anything

        Parameters:
            can_have_os_dependency:
                if True it means that the vulnerability does have the possibility that it can only be exploited if it is on a particular operating system
            os_list:
                a list of operating systems that the vulnerability can be on (depends on the service)
        """
        # 1 for easy, 0 for impossible
        # Change to fit distributions
        self.complexity = constants.VULN_MIN_COMPLEXITY + (1 - constants.VULN_MIN_COMPLEXITY) * random.random()
        # 1 for complete compromise
        # 0 for nothing
        self.impact = random.random() * 10
        self.cvss = (self.complexity + self.impact) / 2
        self.exploitability = self.cvss/5.5
        self.exploit_attempt = 0
        self.exploited = False
        self.has_os_dependency = False
        self.vuln_os_list = []
        self.has_dependent_vulns = random.random() < constants.VULN_PROB_DEPENDS_ON_OTHER_VULNS
        self.dependent_vuln_id = random.choice(
            [x for x in range(0, 101, int(100 * constants.VULN_PROB_DEPENDS_ON_OTHER_VULNS))])
        self.dependent_vulns = []

        self.id = str(uuid.uuid4())
        self.logger = logging.getLogger("vuln-{}".format(self.id))
        if can_have_os_dependency and len(os_list) > 1:
            if random.random() < constants.VULN_PROB_DEPENDS_ON_OS:
                self.has_os_dependency = True
                self.vuln_os_list = random.sample(os_list, k=random.randint(1, len(os_list) - 1))
                # self.vuln_os_list = random.sample(os_list, k=random.randint(1, 2))

    def is_exploited(self):
        return self.exploited

    def get_id(self):
        return self.id

    def can_exploit_with_dependent_vuln(self, vulns):
        """
        Checks if there is another vulnerability that the adversary can exploit which enables this vulnerability

        Parameters:
            vulns:
                a list of the vulnerabilities that can be exploited

        Returns:
            true if there is another vulnerability that enables this vulnerability or it has no depedencies
        """
        if not self.has_dependent_vulns:
            return True
        for v in vulns:
            if v.dependent_vuln_id == self.dependent_vuln_id and not v == self:
                return True
        return False

    def exploit_time(self, host=None):
        """"
        Returns:
            the random time it would take to exploit this vulnerability
            the more attempts a hacker tries at exploiting a particular vulnerability the faster the exploit time becomes

        """
        exp_time = constants.ATTACK_DURATION['EXPLOIT_VULN'] * (1 - self.complexity)
        if self.has_os_dependency and host is not None and host.os_type not in self.vuln_os_list:
            exp_time *= 2.5
        if self.exploited:
            return exp_time / 2
        return exp_time
        # return constants.VULN_MIN_EXPLOIT_TIME + (constants.VULN_MAX_EXPLOIT_TIME -
        # constants.VULN_MIN_EXPLOIT_TIME) * ( 1 - self.complexity) / ( self.exploit_attempt + 1)

    def network(self, host=None):
        """
        Tries to exploit the vulnerability

        Parameters:
            host:
                the host instance that has the vulnerability to use to check if the vulnerability is OS dependent

        Returns:
            the impact score if successfully exploited, otherwise 0.0
        """
        if self.exploited:
            return self.impact

        # if self.has_os_dependency and host is not None:
        #     if host.os_type not in self.vuln_os_list:
        #         return 0.0
        self.exploit_attempt += 1
        if random.random() < self.complexity:
            self.exploited = True
            # if self.has_os_dependency:
                # self.logger.info("OS DEPENDENT VULNERABILITY EXPLOITED!")
            return self.impact
        return 0.0

    # def network(self, host=None):
    #     """
    #     Tries to exploit the vulnerability
    #
    #     Parameters:
    #         host:
    #             the host instance that has the vulnerability to use to check if the vulnerability is OS dependent
    #
    #     Returns:
    #         the impact score if successfully exploited, otherwise 0.0
    #     """
    #     if self.exploited:
    #         return self.impact
    #
    #     self.exploit_attempt += 1
    #     if random.random() < self.complexity:
    #         self.exploited = True
    #     self.exploit_attempt += 1
    #     return self.impact

    def roa(self):
        """
        Psuedo return on attack metric
        
        This is because I have set the probability of an attack occuring to be the same as the 'attack cost'.
        
        This version uses the exploit_time as the attack cost.
        
        The x100 is because impact is expressed as a value 1-10 on CVE
        """
        return (self.complexity * self.impact) / self.exploit_time()
    
    def risk(self):
        """
        Psuedo risk of this vulnerability.
        
        Risk is defined as the product of the likelihood of exploitation and the impact.
        The likelihood can be adjusted based on exploit attempts and other factors if needed.

        Purpose: Measures the potential danger or threat posed by a vulnerability from the defender's perspective.

        Returns:
            The calculated risk value.
        """
        likelihood = self.complexity  # Likelihood of exploitation is based on complexity
        impact = self.impact  # Impact value
        
        return likelihood * impact

    def initial_roa(self):
        return (self.complexity * self.impact) / (constants.VULN_MIN_EXPLOIT_TIME +
                                                  (
                                                          constants.VULN_MAX_EXPLOIT_TIME - constants.VULN_MIN_EXPLOIT_TIME) * (
                                                          1 - self.complexity))

    def __eq__(self, other):
        """
        Checks if two instances a Vulnerability are equal
        """
        if not isinstance(other, Vulnerability):
            return False
        return other.id == self.id


class Service:

    def __init__(self, service_name, service_version, vulnerabilities):
        """
        Creates a Service instance that are assigned to Hosts

        Parameters:
            service_name:
                the name of the service
            service_version:
                the version of the service
            vulnerabilities:
                a list of the vulnerabilities that are on the service
        """
        self.name = service_name
        self.version = service_version
        self.vulnerabilities = sorted(vulnerabilities, key=lambda v: v.roa(), reverse=True)
        self.exploit_value = 0.0
        self.id = str(uuid.uuid4())

    def copy(self):
        """
        Returns:
            a copy of this service instance
        """
        return Service(self.name, self.version, [v for v in self.vulnerabilities])

    def get_vulns(self, roa_threshold=0):
        """
        Returns:
            the top X vulnerabilities in terms of RoA of the service that have not been exploited yet
        """
        return [
                   v
                   for v in self.vulnerabilities
                   if v.roa() > roa_threshold and not v.is_exploited()
               ][:constants.SERVICE_TOP_X_VULNS_TO_RETURN]
    
        
    # def get_vulns_risk(self, risk_threshold=0):
    #     """
    #     Returns:
    #         the top X vulnerabilities in terms of Risk of the service that have not been exploited yet
    #     """
    #     return [
    #                v
    #                for v in self.vulnerabilities
    #                if v.risk() > risk_threshold and not v.is_exploited()
    #            ][:constants.SERVICE_TOP_X_VULNS_TO_RETURN]
 

    def get_all_vulns(self):
        return self.vulnerabilities

    def get_id(self):
        return self.id

    def is_exploited(self):
        self.exploit_value = 0
        for vuln in self.vulnerabilities:
            if vuln.exploited:
                self.exploit_value += vuln.impact
        return self.exploit_value > constants.SERVICE_COMPROMISED_THRESHOLD

    def discover_vuln_time(self, roa_threshold=0):
        return len(self.get_vulns(roa_threshold=roa_threshold)) * constants.SERVICE_DISCOVER_EACH_VULN_TIME

    def get_highest_roa_vuln(self):
        vuln_len = len(self.get_vulns())
        if vuln_len < 1:
            return 0.0
        return self.get_vulns()[0].roa()

    def __eq__(self, other):
        if not isinstance(other, Service):
            return False
        return other.name == self.name and other.version == self.version


class ServicesGenerator:

    def __init__(self,
                 services_per_os=constants.SERVICE_NO_OF_SERVICES_PER_OS,
                 percent_cross_platform=constants.VULN_PERCENT_CROSS_PLATFORM,
                 max_vuln_probability=constants.VULN_MAX_PROB_FOR_OCCURING_FOR_SERVICE_VERSION,
                 vuln_patch_mean=constants.VULN_PATCH_MEAN,
                 vuln_patch_range=constants.VULN_PATCH_RANGE,
                 vuln_initial_chances=constants.VULN_INITIAL_CHANCES,
                 os_dependent_vuln_chance=constants.VULN_PROB_DEPENDS_ON_OS,
                 dependent_vuln_chance=constants.VULN_PROB_DEPENDS_ON_OTHER_VULNS):
        """
        Used to generator services for the simulation

        Parameters:
            services_per_os:
                the number of the services for each OS
            percent_cross_platform:
                percent of services that avaliable across all platforms
            max_vuln_probability:
                the maximum probability for older versions of service having a vulnerability
            vuln_patch_mean:
                thae number of versions on average it takes for a vulnerability to be ptched
            vuln_patch_range:
                the range from the vuln_patch_mean
            vuln_initial_chances:
                for the first version of a service it iterates vuln_initial_chances testing if the first service gets a new vulnerability
            os_dependent_vuln_chance (Unused):
                the probability of a vulnerability being only enabled on specific OS (if the service is available across platforms)
            dependent_vuln_chance (Unused):
                the chance that a vulnerability can only be exploited if there is another particular type vulnerability that can also be exploited
        """
        self.services = None
        self.service_names = None
        self.os_services = None
        self.services_per_os = services_per_os
        self.percent_cross_platform = percent_cross_platform
        self.os_dependent_vuln_chance = os_dependent_vuln_chance
        self.dependent_vuln_chance = dependent_vuln_chance
        self.vuln_patch_mean = vuln_patch_mean
        self.vuln_patch_range = vuln_patch_range
        self.vuln_initial_chances = vuln_initial_chances
        self.max_vuln_probability = max_vuln_probability
        self.gen_services()

    def get_random_service(self, os_type, os_version):
        """
        Gets a random service for a given OS type and version

        Parameters:
            os_type:
                the type of OS
            os_version:
                the version of the OS

        Returns:
            a random service for the provided os_type and os_version
        """
        service_name = random.choice(list(self.os_services[os_type][os_version].keys()))
        return random.choice(self.os_services[os_type][os_version][service_name]).copy()

    def get_random_service_latest_version(self, os_type, os_version):
        """
        Gets a random service set to the latest version for the provided OS type and version

        Parameters:
            os_type:
                the type of OS
            os_version:
                the version of the OS

        Returns:
            a random service set to the latest version for the provided os_type and os_version
        """
        service_name = random.choice(list(self.os_services[os_type][os_version].keys()))
        return self.os_services[os_type][os_version][service_name][-1].copy()

    def service_is_compatible_with_os(self, os_type, os_version, service):
        """
        Checks if a service is comptabile with a particular OS type and version

        Parameters:
            os_type:
                the type of OS
            os_version:
                the version of the OS
            service:
                the service to check if it is compatible with the provided os_type and os_version

        Returns:
            True is the service is compatible, false otherwise
        """
        return service in list(self.os_services[os_type][os_version].keys())

    def gen_services(self):
        """
        Generates all of the services for each OS type and version for the simulation
        """
        self.os_services = {os_name: {} for os_name in constants.OS_TYPES}

        for os_type in constants.OS_TYPES:
            for os_version in constants.OS_VERSION_DICT[os_type]:
                self.os_services[os_type][os_version] = {}

        wordlist = ServicesGenerator.get_service_name_list()
        types_of_os = len(constants.OS_TYPES)
        total_services = self.services_per_os * types_of_os
        self.service_names = random.choices(wordlist, k=total_services)

        s_versions = constants.SERVICE_VERSIONS
        s_versions_len = len(s_versions)

        self.services = {}
        # os_list = constants.OS_TYPES
        for s_index, service in enumerate(self.service_names):
            os_list = [constants.OS_TYPES[s_index // self.services_per_os]]

            if random.random() < self.percent_cross_platform:
                os_list = constants.OS_TYPES

            # can_have_os_depend_vuln = len(os_list) > 1
            can_have_os_depend_vuln = True

            self.services[service] = []
            vulns = {}

            # Code for vulnerability generation, commented out double generation from original code
            for i in range(self.vuln_initial_chances):
                if random.random() < self.max_vuln_probability:
                    vuln_patch_dist = i + random.randint(-self.vuln_patch_range, self.vuln_patch_range)
                    vulns[vuln_patch_dist] = Vulnerability(
                        can_have_os_dependency=can_have_os_depend_vuln,
                        os_list=os_list
                    )

            # Adding Vuln at version 99 to ensure there is a vuln in every version
            vulns[99] = Vulnerability(
                can_have_os_dependency=can_have_os_depend_vuln,
                os_list=os_list
            )
            for sv_index in range(s_versions_len):
                service_version = s_versions[sv_index]
                # version_scale = (s_versions_len - sv_index)/s_versions_len
                # print("service version: ", service_version, "version scale", version_scale)
                # if random.random() < self.max_vuln_probability*version_scale:
                #     vuln_patch_dist = self.vuln_patch_mean + random.randint(-self.vuln_patch_range, self.vuln_patch_range)
                #     vulns[sv_index+vuln_patch_dist] = Vulnerability(
                #         can_have_os_dependency=can_have_os_depend_vuln, 
                #         os_list=os_list
                #     )

                active_vulns = [vulns[i] for i in vulns if i >= sv_index]
                self.services[service] = self.services[service] + [Service(service, service_version, active_vulns)]

            for os_name in os_list:
                os_versions = constants.OS_VERSION_DICT[os_name]
                total_os_versions = len(os_versions)
                version_split = s_versions_len // total_os_versions
                for os_version_index, os_version in enumerate(os_versions):
                    service_versions = self.services[service][s_versions_len - (
                            os_version_index + 1) * version_split:s_versions_len - os_version_index * version_split]
                    self.os_services[os_name][os_version][service] = service_versions

    @staticmethod
    def get_service_name_list():
        return [x.decode() for x in pkg_resources.resource_string('mtdnetwork', "data/words.txt").splitlines()]

    def get_all_generated_services(self):
        return self.os_services
