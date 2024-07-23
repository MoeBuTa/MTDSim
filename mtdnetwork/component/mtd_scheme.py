import random
from collections import deque
from mtdnetwork.mtd.completetopologyshuffle import CompleteTopologyShuffle
from mtdnetwork.mtd.ipshuffle import IPShuffle
from mtdnetwork.mtd.hosttopologyshuffle import HostTopologyShuffle
from mtdnetwork.mtd.portshuffle import PortShuffle
from mtdnetwork.mtd.osdiversity import OSDiversity
from mtdnetwork.mtd.servicediversity import ServiceDiversity
from mtdnetwork.mtd.usershuffle import UserShuffle
from mtdnetwork.mtd.osdiversityassignment import OSDiversityAssignment
from mtdnetwork.data.constants import MTD_TRIGGER_INTERVAL
from heapq import heappush, heappop


class MTDScheme:

    def __init__(self, scheme: str, network, mtd_trigger_interval=None, mtd_trigger_std=0.5, custom_strategies=None):
        self._scheme = scheme
        self._mtd_trigger_interval = mtd_trigger_interval
        self._mtd_trigger_std = mtd_trigger_std
        self._mtd_register_scheme = None
        self._mtd_strategies = [
                                CompleteTopologyShuffle,
                                # HostTopologyShuffle,
                                IPShuffle,
                                OSDiversity,
                                # PortShuffle,
                                # OSDiversityAssignment,
                                ServiceDiversity,
                                # UserShuffle
                                ]
        self._mtd_custom_strategies = custom_strategies
        self.network = network
        self._init_mtd_scheme(scheme)

    def _init_mtd_scheme(self, scheme):
        """
        assign an MTD scheme based on the parameter
        """
        if self._mtd_custom_strategies is None:
            self._mtd_custom_strategies = self._mtd_strategies
        if self._mtd_trigger_interval is None:
            self._mtd_trigger_interval, self._mtd_trigger_std = MTD_TRIGGER_INTERVAL[scheme]
        if scheme == 'simultaneous':
            self._mtd_register_scheme = self._register_mtd_simultaneously
        elif scheme == 'random':
            self._mtd_register_scheme = self._register_mtd_randomly
        elif scheme == 'alternative':

            self._mtd_custom_strategies = deque(self._mtd_custom_strategies)
            self._mtd_register_scheme = self._register_mtd_alternatively
        elif scheme == 'single':
            self._mtd_register_scheme = self._register_mtd_single
        elif scheme == 'mtd_ai':
            self._mtd_register_scheme = self._register_mtd_ai

    def _mtd_register(self, mtd):
        """
        register an MTD strategy to the queue
        """
        if isinstance(mtd, type):
            mtd_strategy = mtd(network=self.network)
        else:
            mtd_strategy = mtd
        heappush(self.network.get_mtd_queue(), (mtd_strategy.get_priority(), mtd_strategy))

    def _register_mtd_simultaneously(self):
        """
        register all MTDs for simultaneous scheme
        """
        for mtd in self._mtd_custom_strategies:
            self._mtd_register(mtd=mtd)
        return self.network.get_mtd_queue()

    def _register_mtd_randomly(self):
        """
        register an MTD for random scheme
        """
        self._mtd_register(mtd=random.choice(self._mtd_custom_strategies))

    def _register_mtd_alternatively(self):
        """
        register an MTD for alternative scheme
        """
        mtd = self._mtd_custom_strategies.popleft()
        self._mtd_register(mtd=mtd)
        self._mtd_custom_strategies.append(mtd)

    def _register_mtd_single(self):
        self._mtd_register(mtd=self._mtd_custom_strategies)

    def _register_mtd_ai(self, mtd_technique):
        """
        register an MTD for AI scheme
        """
        self._mtd_register(mtd=self._mtd_custom_strategies[mtd_technique - 1])

    def trigger_suspended_mtd(self):
        """
        trigger an MTD from suspended list
        """
        suspend_dict = self.network.get_suspended_mtd()
        mtd = suspend_dict[min(suspend_dict.keys())]
        del suspend_dict[min(suspend_dict.keys())]
        return mtd

    def trigger_mtd(self):
        """
        trigger an MTD from mtd queue
        """
        return heappop(self.network.get_mtd_queue())[1]

    def suspend_mtd(self, mtd_strategy):
        """
        put an MTD into the suspended list
        """
        self.network.get_mtd_stats().add_total_suspended()
        self.network.get_suspended_mtd()[mtd_strategy.get_priority()] = mtd_strategy

    def register_mtd(self, mtd_action=None):
        """
        call an MTD register scheme function
        """
        if mtd_action is not None:
            self._mtd_register_scheme(mtd_action)
        self._mtd_register_scheme(mtd_action)

    def get_scheme(self):
        return self._scheme

    def get_mtd_trigger_interval(self):
        return self._mtd_trigger_interval

    def get_mtd_trigger_std(self):
        return self._mtd_trigger_std

    def set_mtd_strategies(self, mtd):
        self._mtd_strategies = mtd
