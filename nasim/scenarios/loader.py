"""This module contains functionality for loading network scenarios from yaml
files.
"""
import math
# kjkoo
import os
import yaml
import os.path as osp

import nasim.scenarios.utils as u
from nasim.scenarios import Scenario
from nasim.scenarios.host import Host


# dictionary of valid key names and value types for config file
VALID_CONFIG_KEYS = {
    u.SUBNETS: list,
    u.TOPOLOGY: list,
    u.SENSITIVE_HOSTS: dict,
    u.OS: list,
    u.SERVICES: list,
    u.PROCESSES: list,
    u.EXPLOITS: dict,
    u.PRIVESCS: dict,
    u.SERVICE_SCAN_COST: (int, float),
    u.SUBNET_SCAN_COST: (int, float),
    u.OS_SCAN_COST: (int, float),
    u.PROCESS_SCAN_COST: (int, float),
    u.HOST_CONFIGS: dict,
    u.FIREWALL: dict,
    # kjkoo : (23-0315) add ATTACK_TECHS_DIR
    #   -> (23-0515) u.ATTACK_PRE_/POST_STATE: 삭제
    u.FILES: list,
    u.VULNERABILITIES: list,
    u.ATTACK_TECHS: str,

    #u.ATTACK_PRE_STATE: list,
    #u.ATTACK_POST_STATE: list

    # defender
    u.DEFENDER_TECHS: str
}

OPTIONAL_CONFIG_KEYS = {u.STEP_LIMIT: int}

VALID_ACCESS_VALUES = ["user", "root", u.USER_ACCESS, u.ROOT_ACCESS]
ACCESS_LEVEL_MAP = {
    "user": u.USER_ACCESS,
    "root": u.ROOT_ACCESS
}

# required keys for exploits
EXPLOIT_KEYS = {
    u.EXPLOIT_SERVICE: str,
    u.EXPLOIT_OS: str,
    u.EXPLOIT_PROB: (int, float),
    u.EXPLOIT_COST: (int, float),
    u.EXPLOIT_ACCESS: (str, int)
}

# required keys for privesc actions
PRIVESC_KEYS = {
    u.PRIVESC_OS: str,
    u.PRIVESC_PROCESS: str,
    u.PRIVESC_PROB: (int, float),
    u.PRIVESC_COST: (int, float),
    u.PRIVESC_ACCESS: (str, int)
}

# required keys for host configs
HOST_CONFIG_KEYS = {
    u.HOST_OS: (str, None),
    u.HOST_SERVICES: list,
    u.HOST_PROCESSES: list,
    # kjkoo
    u.HOST_FILES: list,
    u.HOST_VULNERABILITIES: list,
    u.HOST_IPS: list
}

# kjkoo:
#   - (24-0521) 추가
#   - (24-0808) 추가+수정 : pps 정의 방안 (24-0731) 버전 적용
ATTACK_PPS_DICT = {

    'host_status_list': ['compromised', 'reachable', 'discovered'],

    # os
    'os_type_dict': {
        'linux': 1,
        'windows': 2,
        'macos': 3
    },
    'os_status_dict': {
        'running': 1,
        'shutdown': 2,
        'destroyed': 4
    },
    'os_v_dict': {
        'cve-1': 1,
        'cve-2': 2,
        'cve-3': 4,
        'cve-4': 8
    },

    # privilege
    'privilege_dict': {
        'user': 1,
        'admin': 2,
        'web': 3
    },

    # service
    'service_type_list': ['rdp', 'vnc', 'web', 'db', 'firewall', 'av', 'mail',
                          'ssh', 'ftp', 'log4j', 'eternalblue', 'tomcat', 'smb'],
    'service_status_list': ['rdp_s', 'vnc_s', 'web_s', 'db_s', 'firewall_s', 'av_s', 'mail_s',
                            'ssh_s', 'ftp_s', 'log4j_s', 'eternalblue_s', 'tomcat_s', 'smb_s'],
    'service_vuln_list': ['rdp_v', 'vnc_v', 'web_v', 'db_v', 'firewall_v', 'av_v', 'mail_v',
                          'ssh_v', 'ftp_v', 'log4j_v', 'eternalblue_v', 'tomcat_v', 'smb_v'],
    'service_status_dict': {
        'running': 1,
        'stop': 2,
        'unavailable': 4,
        'infected': 8
    },
    'service_vuln_dict': {
        'cve-1': 1,
        'cve-2': 2,
        'cve-3': 4,
        'cve-4': 8
    },

    # process
    'process_type_list': ['process', 'authprocess'],
    'process_status_list': ['process_s', 'authprocess_s'],
    'process_privilege_list': ['process_p', 'authprocess_p'],
    'process_status_dict': {
        'running': 1,
        'stop': 2,
        'malicious': 4
    },
    'process_priv_dict': {
        'user': 1,
        'admin': 2,
        'system': 4
    },

    # application
    'application_type_list': ['ms-product', 'hwp', 'chrome', 'ie'],
    'application_status_list': ['ms-product_s', 'hwp_s', 'chrome_s', 'ie_s'],
    'application_vuln_list': ['ms-product_v', 'hwp_v', 'chrome_v', 'ie_v'],
    'application_status_dict': {
        'running': 1,
        'stop': 2,
        'unavailable': 4,
        'infected': 4
    },
    'application_vuln_dict': {
        'cve-1': 1,
        'cve-2': 2,
        'cve-3': 4,
        'cve-4': 8
    },

    # file
    'file_type_list': ['password', 'db_file', 'document', 'networkinfo', 'userinfo', 'tool', 'log',
                       'exploit', 'malware', 'osinfo', 'persistence', 'defenseevasion'],
    'file_status_list': ['password_s', 'db_file_s', 'document_s', 'networkinfo_s', 'userinfo_s', 'tool_s', 'log_s',
                       'exploit_s', 'malware_s', 'osinfo_s', 'persistence_s', 'defenseevasion_s'],
    'file_access_list': ['password_a', 'db_file_a', 'document_a', 'networkinfo_a', 'userinfo_a', 'tool_a', 'log_a',
                       'exploit_a', 'malware_a', 'osinfo_a', 'persistence_a', 'defenseevasion_a'],
    # (24-0827) 추가 (임시)
    'file_id_list': ['password_i', 'db_file_i', 'document_i', 'networkinfo_i', 'userinfo_i', 'tool_i', 'log_i',
                       'exploit_i', 'malware_i', 'osinfo_i', 'persistence_i', 'defenseevasion_i'],
    'file_status_dict': {
        'init': 0,
        'created': 1,
        'deleted': 2,
        'modified': 4,
        'discovered': 8,
        'dumped': 16,   # == copied
        'encrypted': 32,
        'encoded': 64,
        'compression': 128,
        'cracked': 256,
        'received': 512,
        'leaked': 1024
    },
    'file_access_dict': {
        'read': 1,
        'write': 2,
        'execute': 4
    },
    'file_id_dict': {
        'id-1': 1,
        'id-2': 2,
        'id-3': 4,
        'id-4': 8
    }
}

DEFENDER_PPS_DICT = {

    'host_status_list': ['compromised'],

    # os
    'os_type_dict': {
        'linux': 1,
        'windows': 2,
        'macos': 3
    },
    'os_status_dict': {
        'running': 1,
        'shutdown': 2,
        'destroyed': 4
    },
    'os_v_dict': {
        'cve-1': 1,
    },

    # privilege
    'privilege_dict': {
        'user': 1,
        'admin': 2,
        'web': 3
    },

    # service
    'service_type_list': ['web', 'ssh', 'ftp', 'smb'],
    'service_status_list': ['web_s', 'ssh_s', 'ftp_s', 'smb_s'],
    'service_vuln_list': ['web_v', 'ssh_v', 'ftp_v', 'smb_v'],
    'service_status_dict': {
        'running': 1,
        'stop': 2,
        'unavailable': 4,
        'infected': 8
    },
    'service_vuln_dict': {
        'cve-1': 1,
    },

    # process
    'process_type_list': ['process', 'authprocess'],
    'process_status_list': ['process_s', 'authprocess_s'],
    'process_privilege_list': ['process_p', 'authprocess_p'],
    'process_status_dict': {
        'running': 1,
        'stop': 2,
        'malicious': 4
    },
    'process_priv_dict': {
        'user': 1,
        'admin': 2,
        'system': 4
    },

    # application
    'application_type_list': ['ms-product', 'hwp', 'chrome', 'ie'],
    'application_status_list': ['ms-product_s', 'hwp_s', 'chrome_s', 'ie_s'],
    'application_vuln_list': ['ms-product_v', 'hwp_v', 'chrome_v', 'ie_v'],
    'application_status_dict': {
        'running': 1,
        'stop': 2,
        'unavailable': 4,
        'infected': 4
    },
    'application_vuln_dict': {
        'cve-1': 1,
    },

    # file
    'file_type_list': ['password', 'db_file'],
    'file_status_list': ['password_s', 'db_file_s'],
    'file_access_list': ['password_a', 'db_file_a'],
    # (24-0827) 추가 (임시)
    'file_id_list': ['password_i', 'db_file_i'],
    'file_status_dict': {
        'init': 0,
        'created': 1,
        'deleted': 2,
        'modified': 4,
        'discovered': 8,
        'dumped': 16,   # == copied
        'encrypted': 32,
        'encoded': 64,
        'compression': 128,
        'cracked': 256,
        'received': 512,
        'leaked': 1024
    },
    'file_access_dict': {
        'read': 1,
        'write': 2,
        'execute': 4
    },
    'file_id_dict': {
        'id-1': 1,
        'id-2': 2,
        'id-3': 4,
        'id-4': 8
    }
}


class ScenarioLoader:

    def load(self, file_path, name=None):
        """Load the scenario from file

        Arguments
        ---------
        file_path : str
            path to scenario file
        name : str, optional
            the scenarios name, if None name will be generated from file path
            (default=None)

        Returns
        -------
        scenario_dict : dict
            dictionary with scenario definition

        Raises
        ------
        Exception
            If file unable to load or scenario file is invalid.
        """
        self.yaml_dict = u.load_yaml(file_path)
        if name is None:
            name = u.get_file_name(file_path)
        self.name = name
        self._check_scenario_sections_valid()

        self._parse_subnets()
        self._parse_topology()

        self._parse_os()
        self._parse_services()
        self._parse_processes()  # kjkoo : 일단 유지
        self._parse_sensitive_hosts()
        self._parse_exploits()  # kjkoo : 일단 유지
        self._parse_privescs()  # kjkoo : 일단 유지

        # kjkoo : attack_techs의 pre_/post_state 값만으로 RL 에이전트 학습 상태 테이블을 만든다.
        # kjkoo : load attack techniques
        self._parse_files()
        self._parse_vulnerabilities()

        self._parse_scan_costs()
        self._parse_host_configs()
        self._parse_firewall()
        self._parse_hosts()
        self._parse_step_limit()

        self._parse_attack_techs()  # kjkoo : (24-0503) 수정
        self._parse_defend_techs()

        # self._parse_attack_pre_state()
        # self._parse_attack_post_state()
        return self._construct_scenario()

    def _construct_scenario(self):
        scenario_dict = dict()
        scenario_dict[u.SUBNETS] = self.subnets
        scenario_dict[u.TOPOLOGY] = self.topology
        scenario_dict[u.OS] = self.os
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.PROCESSES] = self.processes
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.PRIVESCS] = self.privescs
        scenario_dict[u.OS_SCAN_COST] = self.os_scan_cost
        scenario_dict[u.SERVICE_SCAN_COST] = self.service_scan_cost
        scenario_dict[u.SUBNET_SCAN_COST] = self.subnet_scan_cost
        scenario_dict[u.PROCESS_SCAN_COST] = self.process_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario_dict[u.STEP_LIMIT] = self.step_limit

        # kjkoo : add attack_techs_dict, pre/post-state to scenario_dict
        scenario_dict[u.ATTACK_TECHS] = self.attack_techs_dict
        # kjkoo : [ ] TODO : (23-0324) attack_pre_state에 os, service, process
        #  등이 중복되면 제거해야 한다. (??)
        scenario_dict[u.ATTACK_PPS] = self.attack_pps
        scenario_dict[u.ATTACK_PPS_DICT] = self.attack_pps_dict

        # defender
        scenario_dict[u.DEFENDER_TECHS] = self.defend_techs_dict
        scenario_dict[u.DEFENDER_PPS] = self.defend_pps
        scenario_dict[u.DEFENDER_PPS_DICT] = self.defend_pps_dict

        # Attacker / Defender Using dict
        scenario_dict[u.FILES] = self.files
        scenario_dict[u.VULNERABILITIES] = self.vulnerabilities

        return Scenario(
            scenario_dict, name=self.name, generated=False
        )

    def _check_scenario_sections_valid(self):
        """Checks if scenario dictionary contains all required sections and
        they are valid type.
        """
        # 0. check correct number of keys
        assert len(self.yaml_dict) >= len(VALID_CONFIG_KEYS), \
            (f"Too few config file keys: {len(self.yaml_dict)} "
             f"< {len(VALID_CONFIG_KEYS)}")

        # 1. check keys are valid and values are correct type
        for k, v in self.yaml_dict.items():
            assert k in VALID_CONFIG_KEYS or k in OPTIONAL_CONFIG_KEYS, \
                f"{k} not a valid config file key"

            if k in VALID_CONFIG_KEYS:
                expected_type = VALID_CONFIG_KEYS[k]
            else:
                expected_type = OPTIONAL_CONFIG_KEYS[k]

            assert isinstance(v, expected_type), \
                (f"{v} invalid type for config file key '{k}': {type(v)}"
                 f" != {expected_type}")

    def _parse_subnets(self):
        subnets = self.yaml_dict[u.SUBNETS]
        self._validate_subnets(subnets)
        # insert internet subnet
        subnets.insert(0, 1)
        self.subnets = subnets
        self.num_hosts = sum(subnets)-1

    def _validate_subnets(self, subnets):
        # check subnets is valid list of positive ints
        assert len(subnets) > 0, "Subnets cannot be empty list"
        for subnet_size in subnets:
            assert type(subnet_size) is int and subnet_size > 0, \
                f"{subnet_size} invalid subnet size, must be positive int"

    def _parse_topology(self):
        topology = self.yaml_dict[u.TOPOLOGY]
        self._validate_topology(topology)
        self.topology = topology

    def _validate_topology(self, topology):
        # check topology is valid adjacency matrix
        assert len(topology) == len(self.subnets), \
            ("Number of rows in topology adjacency matrix must equal "
             f"number of subnets: {len(topology)} != {len(self.subnets)}")

        for row in topology:
            assert isinstance(row, list), \
                "topology must be 2D adjacency matrix (i.e. list of lists)"
            assert len(row) == len(self.subnets), \
                ("Number of columns in topology matrix must equal number of"
                 f" subnets: {len(topology)} != {len(self.subnets)}")
            for col in row:
                assert isinstance(col, int) and (col == 1 or col == 0), \
                    ("Subnet_connections adjaceny matrix must contain only"
                     f" 1 (connected) or 0 (not connected): {col} invalid")

    def _parse_os(self):
        os = self.yaml_dict[u.OS]
        self._validate_os(os)
        self.os = os

    def _validate_os(self, os):
        assert len(os) > 0, \
            f"{len(os)}. Invalid number of OSs, must be >= 1"
        assert len(os) == len(set(os)), \
            f"{os}. OSs must not contain duplicates"

    def _parse_services(self):
        services = self.yaml_dict[u.SERVICES]
        self._validate_services(services)
        self.services = services

    def _validate_services(self, services):
        assert len(services) > 0, \
           f"{len(services)}. Invalid number of services, must be > 0"
        assert len(services) == len(set(services)), \
            f"{services}. Services must not contain duplicates"

    def _parse_processes(self):
        processes = self.yaml_dict[u.PROCESSES]
        self._validate_processes(processes)
        self.processes = processes

    def _validate_processes(self, processes):
        assert len(processes) >= 1, \
            f"{len(processes)}. Invalid number of services, must be > 0"
        assert len(processes) == len(set(processes)), \
            f"{processes}. Processes must not contain duplicates"

    def _parse_sensitive_hosts(self):
        sensitive_hosts = self.yaml_dict[u.SENSITIVE_HOSTS]
        self._validate_sensitive_hosts(sensitive_hosts)

        self.sensitive_hosts = dict()
        for address, value in sensitive_hosts.items():
            self.sensitive_hosts[eval(address)] = value

    def _validate_sensitive_hosts(self, sensitive_hosts):
        # check sensitive_hosts is valid dict of (subnet, id) : value
        assert len(sensitive_hosts) > 0, \
            ("Number of sensitive hosts must be >= 1: "
             f"{len(sensitive_hosts)} not >= 1")

        assert len(sensitive_hosts) <= self.num_hosts, \
            ("Number of sensitive hosts must be <= total number of "
             f"hosts: {len(sensitive_hosts)} not <= {self.num_hosts}")

        # sensitive hosts must be valid address
        for address, value in sensitive_hosts.items():
            subnet_id, host_id = eval(address)
            assert self._is_valid_subnet_ID(subnet_id), \
                ("Invalid sensitive host tuple: subnet_id must be a valid"
                 f" subnet: {subnet_id} != non-negative int less than "
                 f"{len(self.subnets) + 1}")

            assert self._is_valid_host_address(subnet_id, host_id), \
                ("Invalid sensitive host tuple: host_id must be a valid"
                 f" int: {host_id} != non-negative int less than"
                 f" {self.subnets[subnet_id]}")

            assert isinstance(value, (float, int)) and value > 0, \
                (f"Invalid sensitive host tuple: invalid value: {value}"
                 f" != a positive int or float")

        # 5.c sensitive hosts must not contain duplicate addresses
        for i, m in enumerate(sensitive_hosts.keys()):
            h1_addr = eval(m)
            for j, n in enumerate(sensitive_hosts.keys()):
                if i == j:
                    continue
                h2_addr = eval(n)
                assert h1_addr != h2_addr, \
                    ("Sensitive hosts list must not contain duplicate host "
                     f"addresses: {m} == {n}")

    def _is_valid_subnet_ID(self, subnet_ID):
        if type(subnet_ID) is not int \
           or subnet_ID < 1 \
           or subnet_ID > len(self.subnets):
            return False
        return True

    def _is_valid_host_address(self, subnet_ID, host_ID):
        if not self._is_valid_subnet_ID(subnet_ID):
            return False
        if type(host_ID) is not int \
           or host_ID < 0 \
           or host_ID >= self.subnets[subnet_ID]:
            return False
        return True

    def _parse_exploits(self):
        exploits = self.yaml_dict[u.EXPLOITS]
        self._validate_exploits(exploits)
        self.exploits = exploits

    def _validate_exploits(self, exploits):
        for e_name, e in exploits.items():
            self._validate_single_exploit(e_name, e)

    def _validate_single_exploit(self, e_name, e):
        assert isinstance(e, dict), \
            f"{e_name}. Exploit must be a dict."

        for k, t in EXPLOIT_KEYS.items():
            assert k in e, f"{e_name}. Exploit missing key: '{k}'"
            assert isinstance(e[k], t), \
                f"{e_name}. Exploit '{k}' incorrect type. Expected {t}"

        assert e[u.EXPLOIT_SERVICE] in self.services, \
            (f"{e_name}. Exploit target service invalid: "
             f"'{e[u.EXPLOIT_SERVICE]}'")

        if str(e[u.EXPLOIT_OS]).lower() == "none":
            e[u.EXPLOIT_OS] = None

        assert e[u.EXPLOIT_OS] is None or e[u.EXPLOIT_OS] in self.os, \
            (f"{e_name}. Exploit target OS is invalid. '{e[u.EXPLOIT_OS]}'."
             " Should be None or one of the OS in the os list.")

        assert 0 <= e[u.EXPLOIT_PROB] < 1, \
            (f"{e_name}. Exploit probability, '{e[u.EXPLOIT_PROB]}' not "
             "a valid probability")

        assert e[u.EXPLOIT_COST] > 0, f"{e_name}. Exploit cost must be > 0."

        assert e[u.EXPLOIT_ACCESS] in VALID_ACCESS_VALUES, \
            (f"{e_name}. Exploit access value '{e[u.EXPLOIT_ACCESS]}' "
             f"invalid. Must be one of {VALID_ACCESS_VALUES}")

        if isinstance(e[u.EXPLOIT_ACCESS], str):
            e[u.EXPLOIT_ACCESS] = ACCESS_LEVEL_MAP[e[u.EXPLOIT_ACCESS]]

    def _parse_privescs(self):
        self.privescs = self.yaml_dict[u.PRIVESCS]
        self._validate_privescs(self.privescs)

    def _validate_privescs(self, privescs):
        for pe_name, pe in privescs.items():
            self._validate_single_privesc(pe_name, pe)

    def _validate_single_privesc(self, pe_name, pe):
        s_name = "Priviledge Escalation"

        assert isinstance(pe, dict), f"{pe_name}. {s_name} must be a dict."

        for k, t in PRIVESC_KEYS.items():
            assert k in pe, f"{pe_name}. {s_name} missing key: '{k}'"
            assert isinstance(pe[k], t), \
                (f"{pe_name}. {s_name} '{k}' incorrect type. Expected {t}")

        assert pe[u.PRIVESC_PROCESS] in self.processes, \
            (f"{pe_name}. {s_name} target process invalid: "
             f"'{pe[u.PRIVESC_PROCESS]}'")

        if str(pe[u.PRIVESC_OS]).lower() == "none":
            pe[u.PRIVESC_OS] = None

        assert pe[u.PRIVESC_OS] is None or pe[u.PRIVESC_OS] in self.os, \
            (f"{pe_name}. {s_name} target OS is invalid. '{pe[u.PRIVESC_OS]}'."
             f" Should be None or one of the OS in the os list.")

        assert 0 <= pe[u.PRIVESC_PROB] <= 1.0, \
            (f"{pe_name}. {s_name} probability, '{pe[u.PRIVESC_PROB]}' not "
                "a valid probability")

        assert pe[u.PRIVESC_COST] > 0, \
            f"{pe_name}. {s_name} cost must be > 0."

        assert pe[u.PRIVESC_ACCESS] in VALID_ACCESS_VALUES, \
            (f"{pe_name}. {s_name} access value '{pe[u.PRIVESC_ACCESS]}' "
             f"invalid. Must be one of {VALID_ACCESS_VALUES}")

        if isinstance(pe[u.PRIVESC_ACCESS], str):
            pe[u.PRIVESC_ACCESS] = ACCESS_LEVEL_MAP[pe[u.PRIVESC_ACCESS]]

    def _parse_scan_costs(self):
        self.os_scan_cost = self.yaml_dict[u.OS_SCAN_COST]
        self.service_scan_cost = self.yaml_dict[u.SERVICE_SCAN_COST]
        self.subnet_scan_cost = self.yaml_dict[u.SUBNET_SCAN_COST]
        self.process_scan_cost = self.yaml_dict[u.PROCESS_SCAN_COST]
        for (n, c) in [
                ("OS", self.os_scan_cost),
                ("Service", self.service_scan_cost),
                ("Subnet", self.subnet_scan_cost),
                ("Process", self.process_scan_cost),
        ]:
            self._validate_scan_cost(n, c)

    def _validate_scan_cost(self, scan_name, scan_cost):
        assert scan_cost >= 0, f"{scan_name} Scan Cost must be >= 0."

    def _parse_host_configs(self):
        self.host_configs = self.yaml_dict[u.HOST_CONFIGS]
        self._validate_host_configs(self.host_configs)

    def _validate_host_configs(self, host_configs):
        assert len(host_configs) == self.num_hosts, \
            ("Number of host configurations must match the number of hosts "
             f"in network: {len(host_configs)} != {self.num_hosts}")

        assert self._has_all_host_addresses(host_configs.keys()), \
            ("Host configurations must have no duplicates and have an"
             " address for each host on network.")

        for addr, cfg in host_configs.items():
            self._validate_host_config(addr, cfg)

    def _has_all_host_addresses(self, addresses):
        """Check that list of (subnet_ID, host_ID) tuples contains all
        addresses on network based on subnets list
        """
        for s_id, s_size in enumerate(self.subnets[1:]):
            for m in range(s_size):
                # +1 to s_id since first subnet is 1
                if str((s_id + 1, m)) not in addresses:
                    return False
        return True

    # kjkoo :
    #   - addr : Host address
    #   - cfg : Host configuration
    def _validate_host_config(self, addr, cfg):
        """Check if a host config is valid or not given the list of exploits available
        N.B. each host config must contain at least one service
        """
        # Host address
        err_prefix = f"Host {addr}"
        assert isinstance(cfg, dict) and len(cfg) >= len(HOST_CONFIG_KEYS), \
            (f"{err_prefix} configurations must be a dict of length >= "
             f"{len(HOST_CONFIG_KEYS)}. {cfg} is invalid")

        for k in HOST_CONFIG_KEYS:
            assert k in cfg, f"{err_prefix} configuration missing key: {k}"

        # Host services
        host_services = cfg[u.HOST_SERVICES]
        for service in host_services:
            assert service in self.services, \
                (f"{err_prefix} Invalid service in configuration services "
                 f"list: {service}")

        assert len(host_services) == len(set(host_services)), \
            (f"{err_prefix} configuration services list cannot contain "
             "duplicates")

        # Host processes
        host_processes = cfg[u.HOST_PROCESSES]
        for process in host_processes:
            assert process in self.processes, \
                (f"{err_prefix} invalid process in configuration processes"
                 f" list: {process}")

        assert len(host_processes) == len(set(host_processes)), \
            (f"{err_prefix} configuation processes list cannot contain "
             "duplicates")

        # Host OS
        host_os = cfg[u.HOST_OS]
        assert host_os in self.os, \
            f"{err_prefix} invalid os in configuration: {host_os}"

        # kjkoo (23-0523)
        #   - files, vulnerabilites 구성 확인 필요
        #   - 없거나, 전체 리스트에 포함되어야 한다.
        host_files = cfg[u.HOST_FILES]
        for file in host_files:
            assert file in self.files, \
                f"{err_prefix} invalid files in configuration: {host_files}"

        # Host vulnerabilities
        host_vulnerabilities = cfg[u.HOST_VULNERABILITIES]
        for vuln in host_vulnerabilities:
            assert vuln in self.vulnerabilities, \
                f"{err_prefix} invalid vulnerabilities in configuration: {host_vulnerabilities}"

        # Host IPs
        host_ips = cfg[u.HOST_IPS]
        for ip in host_ips:
            subnet_id, host_id = eval(ip)
            # ip_t = tuple(ip)   # list to tuple
            # subnet_id = ip_t[0]
            # host_id = ip_t[1]
            # subnet_id, host_id = eval(ip)
            assert self._is_valid_host_address(subnet_id, host_id), \
                ("Invalid multiple ip address : host_id must be a valid"
                 f" int: {host_id} != non-negative int less than"
                 f" {self.subnets[subnet_id]}")

        # Host pw_shared : (23-0620)
        host_pw_shared = cfg[u.HOST_PW_SHARED]
        for shared_ip in host_pw_shared:
            subnet_id, host_id = eval(shared_ip)
            assert self._is_valid_host_address(subnet_id, host_id), \
                ("Invalid pw_shared ip address : host_id must be a valid"
                 f" int: {host_id} != non-negative int less than"
                 f" {self.subnets[subnet_id]}")

        # Host Firewall
        fw_err_prefix = f"{err_prefix} {u.HOST_FIREWALL}"
        if u.HOST_FIREWALL in cfg:
            firewall = cfg[u.HOST_FIREWALL]
            assert isinstance(firewall, dict), \
                (f"{fw_err_prefix} must be a dictionary, with host "
                 "addresses as keys and a list of denied services as values. "
                 f"{firewall} is invalid.")
            for addr, srv_list in firewall.items():
                addr = self._validate_host_address(addr, err_prefix)
                assert self._is_valid_firewall_setting(srv_list), \
                    (f"{fw_err_prefix} setting must be a list, contain only "
                     f"valid services and contain no duplicates: {srv_list}"
                     " is not valid")
        else:
            cfg[u.HOST_FIREWALL] = dict()

        # Host value
        v_err_prefix = f"{err_prefix} {u.HOST_VALUE}"
        if u.HOST_VALUE in cfg:
            host_value = cfg[u.HOST_VALUE]
            assert isinstance(host_value, (int, float)), \
                (f"{v_err_prefix} must be an integer or float value. "
                 f"{host_value} is invalid")

            if addr in self.sensitive_hosts:
                sh_value = self.sensitive_hosts[addr]
                assert math.isclose(host_value, sh_value), \
                    (f"{v_err_prefix} for a sensitive host must either match "
                     f"the value specified in the {u.SENSITIVE_HOSTS} section "
                     f"or be excluded the host config. The value {host_value} "
                     f"is invalid as it does not match value {sh_value}.")

    def _validate_host_address(self, addr, err_prefix=""):
        try:
            addr = eval(addr)
        except Exception:
            raise AssertionError(
                f"{err_prefix} address invalid. Must be (subnet, host) tuple"
                f" of integers. {addr} is invalid."
            )
        assert isinstance(addr, tuple) \
            and len(addr) == 2 \
            and all([isinstance(a, int) for a in addr]), \
            (f"{err_prefix} address invalid. Must be (subnet, host) tuple"
             f" of integers. {addr} is invalid.")
        assert 0 < addr[0] < len(self.subnets), \
            (f"{err_prefix} address invalid. Subnet address must be in range"
             f" 0 < subnet addr < {len(self.subnets)}. {addr[0]} is invalid.")
        assert 0 <= addr[1] < self.subnets[addr[0]], \
            (f"{err_prefix} address invalid. Host address must be in range "
             f"0 < host addr < {self.subnets[addr[0]]}. {addr[1]} is invalid.")
        return True

    def _parse_firewall(self):
        firewall = self.yaml_dict[u.FIREWALL]
        self._validate_firewall(firewall)
        # convert (subnet_id, subnet_id) string to tuple
        self.firewall = {}
        for connect, v in firewall.items():
            self.firewall[eval(connect)] = v

    def _validate_firewall(self, firewall):
        assert self._contains_all_required_firewalls(firewall), \
            ("Firewall dictionary must contain two entries for each subnet "
             "connection in network (including from outside) as defined by "
             "network topology matrix")

        for f in firewall.values():
            assert self._is_valid_firewall_setting(f), \
                ("Firewall setting must be a list, contain only valid "
                 f"services and contain no duplicates: {f} is not valid")

    def _contains_all_required_firewalls(self, firewall):
        for src, row in enumerate(self.topology):
            for dest, col in enumerate(row):
                if src == dest:
                    continue
                if col == 1 and (str((src, dest)) not in firewall
                                 or str((dest, src)) not in firewall):
                    return False
        return True

    def _is_valid_firewall_setting(self, f):
        if type(f) != list:
            return False
        for service in f:
            if service not in self.services:
                return False
        for i, x in enumerate(f):
            for j, y in enumerate(f):
                if i != j and x == y:
                    return False
        return True

        # kjkoo : (23-0604)
        #   - 1) 기능 추가 : 초기에 공격자단말은 compromised=True 상태를 만들기 위해
        #   - 2) host_config에 files, vulns 추가

    def _parse_hosts(self):
        """Returns ordered dictionary of hosts in network, with address as
        keys and host objects as values
        """
        hosts = dict()
        for address, h_cfg in self.host_configs.items():
            compromised_for_attacker_host = False
            access_for_attacker_host = 0  # default

            formatted_address = eval(address)
            # kjkoo : files_cfg, vulns_cfg 추가
            # kjkoo : (23-0616) ips_cfg 추가 (list)
            os_cfg, srv_cfg, proc_cfg, files_cfg, vulns_cfg, ips_cfg, pws_cfg = self._construct_host_config(h_cfg)
            # os_cfg, srv_cfg, proc_cfg = self._construct_host_config(h_cfg)
            value = self._get_host_value(formatted_address, h_cfg)
            # kjkoo : (23-0607)
            # discovery_value config 파일에서 가져오는 방식으로 코드 수정 필요

            # kjkoo : 공격자 단말만 configuration yaml파일에 compromised=True로 설정해 놓음
            if 'compromised' in list(h_cfg.keys()):
                compromised_for_attacker_host = True
                access_for_attacker_host = 1  # user

            hosts[formatted_address] = Host(
                address=formatted_address,
                os=os_cfg,
                services=srv_cfg,
                processes=proc_cfg,
                files=files_cfg,
                vulnerabilities=vulns_cfg,
                ips=ips_cfg,
                pw_shared=pws_cfg,
                firewall=h_cfg[u.HOST_FIREWALL],
                value=value,
                discovery_value=5.0,  # kjkoo : (23-0607) default 0 -> 5.0 수정
                compromised=compromised_for_attacker_host,
                access=access_for_attacker_host
            )
        self.hosts = hosts

    # kjkoo : (23-0606)
    #   - files_cfg, vulns_cfg 추가
    #   - ips_cfg 추가
    #   - pw_shared_cfg 추가 (23-0620)
    def _construct_host_config(self, host_cfg):
        os_cfg = {}
        for os_name in self.os:
            os_cfg[os_name] = os_name == host_cfg[u.HOST_OS]
        services_cfg = {}
        for service in self.services:
            services_cfg[service] = service in host_cfg[u.HOST_SERVICES]
        processes_cfg = {}
        for process in self.processes:
            processes_cfg[process] = process in host_cfg[u.HOST_PROCESSES]
        # kjkoo
        files_cfg = {}
        for file in self.files:
            files_cfg[file] = file in host_cfg[u.HOST_FILES]
        vulnerabilities_cfg = {}
        for vuln in self.vulnerabilities:
            vulnerabilities_cfg[vuln] = vuln in host_cfg[u.HOST_VULNERABILITIES]
        # kjkoo : (23-0616)
        ips_cfg = host_cfg[u.HOST_IPS]
        # kjkoo : (23-0620) : password shared
        pws_cfg = host_cfg[u.HOST_PW_SHARED]

        return os_cfg, services_cfg, processes_cfg, files_cfg, vulnerabilities_cfg, ips_cfg, pws_cfg

    def _get_host_value(self, address, host_cfg):
        if address in self.sensitive_hosts:
            return float(self.sensitive_hosts[address])
        return float(host_cfg.get(u.HOST_VALUE, u.DEFAULT_HOST_VALUE))

    def _parse_step_limit(self):
        if u.STEP_LIMIT not in self.yaml_dict:
            step_limit = None
        else:
            step_limit = self.yaml_dict[u.STEP_LIMIT]
            assert step_limit > 0, \
                f"Step limit must be positive int: {step_limit} is invalid"

        self.step_limit = step_limit

        # kjkoo (23-0523)
        #   - files, vulnerabilities 추가
        #   - files :
        #       - Discovered_file_* 형태로 상태 테이블에 반영되어야 함
        #       - 호스트 구성에 포함됨
        #   - vulnerabilities : 호스트 구성(config)에 포함됨
    def _parse_files(self):
        files = self.yaml_dict[u.FILES]
        self._validate_files(files)
        self.files = files

    def _validate_files(self, files):
        assert len(files) > 0, \
            f"{len(files)}. Invalid number of files, must be > 0"
        assert len(files) == len(set(files)), \
            f"{files}. Files must not contain duplicates"

    def _parse_vulnerabilities(self):
        vulnerabilities = self.yaml_dict[u.VULNERABILITIES]
        self._validate_vulnerabilities(vulnerabilities)
        self.vulnerabilities = vulnerabilities

    def _validate_vulnerabilities(self, vulnerabilities):
        assert len(vulnerabilities) > 0, \
            f"{len(vulnerabilities)}. Invalid number of vulnerabilities, must be > 0"
        assert len(vulnerabilities) == len(set(vulnerabilities)), \
            f"{vulnerabilities}. Vulnerabilities must not contain duplicates"

    # kjkoo (23-0516)
    # kjkoo (23-0315) load attack techniques
    def _parse_attack_techs(self):
        # load attack techniques
        self.attack_techs_dict = self._load_attack_techs()
        # print(f"attack_techs_dict: \n{list(attack_techs_dict.keys())}")

        # _extract_pre_/post_state
        pre_state_dict, post_state_dict = self._extract_attack_techs_pre_post_state(self.attack_techs_dict)

        # extract pps(pre-/post-state) feature
        self.attack_pps = self._filter_attack_pps(pre_state_dict, post_state_dict)
        print(f"[+] attack_pps_feature [{len(self.attack_pps)}]: {self.attack_pps}")

        # (24-0521)
        #   - attack_pps_dict 설정
        self.attack_pps_dict = ATTACK_PPS_DICT
        print(f"[+] attack_pps_dict : {self.attack_pps_dict}")

        return

    def _load_attack_techs(self):
        cdir = osp.dirname(osp.abspath(__file__))
        atechs_dir = cdir + '/' + self.yaml_dict[u.ATTACK_TECHS]

        attack_techs_dict = dict()
        # kjkoo : (23-0613)
        #   - attack_list 정렬 : 테크닉 번호로
        attack_list = os.listdir(atechs_dir)
        attack_list_sorted = sorted(attack_list)
        for i, atech in enumerate(attack_list_sorted):
            fpath = atechs_dir + '/' + atech

            with open(fpath, encoding='UTF-8') as fin:
                content = yaml.load(fin, Loader=yaml.FullLoader)
                # 일단 중복을 체크해.
                if atech not in attack_techs_dict.keys():
                    atech_name = atech.split('.yaml')[0]
                    attack_techs_dict[atech_name] = content

        return attack_techs_dict
        # print(f"attack_techs_dict: \n{list(attack_techs_dict.keys())}")

    # kjkoo :
    #   - [24-0513] 추가 : input string에서 공백, 쉼표, and, or, not 제거
    def _string_split_strip(self, in_str):
        string = in_str.replace(",", "")
        string = string.split()
        out_str = list()
        for word in string:
            word = word.strip()
            if word in ['and', 'or', 'not']:
                continue
            else:
                out_str.append(word)
        return out_str

    # kjkoo
    #   - [24-0513] : pre/post-state에 정의된 key-value 사전 생성
    def _construct_state_key_val_dict(self, state_list, state_key_val_dict):
        for idx in range(len(state_list)):
            for key, value in state_list[idx].items():
                state_key = str(key).lower()
                state_value = str(value).lower()

                # state key 중복 체크
                if state_key not in state_key_val_dict.keys():
                    state_key_val_dict[state_key] = []  # key가 없으면 새로 생성
                    # value에 값이 여러개일 때 처리
                    state_key_val_dict[state_key] = self._string_split_strip(state_value)
                    # pre_state_key_val_dict
                else:
                    # value 중복 체크
                    state_value_trim = self._string_split_strip(state_value)
                    for i in range(len(state_value_trim)):
                        # value 있는지 확인
                        if state_value_trim[i] not in state_key_val_dict[state_key]:
                            # 새로운 value면 추가
                            state_key_val_dict[state_key].append(state_value_trim[i])
        return state_key_val_dict

        # kjkoo : pre_state, post_state 정보 추출

    def _extract_attack_techs_pre_post_state(self, attack_techs_dict):
        self.pre_state_key_value_dict = dict()
        self.post_state_key_value_dict = dict()

        pre_state_key_val_dict = self.pre_state_key_value_dict
        post_state_key_val_dict = self.post_state_key_value_dict

        for atk_tech in attack_techs_dict.items():
            content = atk_tech[1]
            atomic_tests = content['atomic_tests']
            atomic_tests_val = atomic_tests[0]

            # print(f'\n{atk_tech[0]}, {atomic_tests_val["technique_remote"]}\n')
            if atomic_tests_val['technique_remote']:
                #########
                ##  source : pre_state
                #########
                source = atomic_tests_val['source']
                src_pre_state = source['pre_state']
                # print(f'>> 여기가 source : pre_state 작업이야 !!!')
                pre_state_key_val_dict = self._construct_state_key_val_dict(src_pre_state, pre_state_key_val_dict)
                # print(f">> [source] pre_state key_val: \n{pre_state_key_val_dict}")

                #########
                ##  dest : post_state
                #########
                # post_state 필드는 정의되지 않을 수도 있다.
                if 'post_state' in source:
                    src_post_state = source['post_state']
                    # print(f'>> 여기가 source : post_state 작업이야 !!!')
                    post_state_key_val_dict = self._construct_state_key_val_dict(src_post_state,
                                                                                 post_state_key_val_dict)
                    # print("source post_state가 정의 되어 있음")
                # else:
                # print("source post_state가 정의 되어 있지 않음")
                # print(f">> [source] post_state key_val: \n{post_state_key_val_dict}")

            #########
            ##  dest : pre_state
            #########
            dest = atomic_tests_val['dest']
            dest_pre_state = dest['pre_state']
            # dest_pre_state

            # print(f'>> 여기가 dest : pre_state 작업이야 !!!')
            pre_state_key_val_dict = self._construct_state_key_val_dict(dest_pre_state, pre_state_key_val_dict)
            # print(f">> [dest] pre_state key_val: \n{pre_state_key_val_dict}")

            #########
            ##  dest : post_state
            #########
            dest_post_state = dest['post_state']
            # dest_pre_state
            # print(f'>> 여기가 dest : post_state 작업이야 !!!')
            post_state_key_val_dict = self._construct_state_key_val_dict(dest_post_state, post_state_key_val_dict)
            # print(f">> [dest] post_state key_val: \n{post_state_key_val_dict}")

        print(f"[+] Attack techniques [{len(list(attack_techs_dict.keys()))}]: {list(attack_techs_dict.keys())}")

        print(f'[+] pre_state_key_dict: {pre_state_key_val_dict}')
        self.pre_state_key_value_dict = pre_state_key_val_dict

        print(f'[+] post_state_key_dict: {post_state_key_val_dict}')
        self.post_state_key_val_dict = post_state_key_val_dict

        return self.pre_state_key_value_dict, self.post_state_key_val_dict


    # kjkoo : pps feacture 추출
    #   - 상태 테이블 컬럼에 사용될 특징 추출
    #   - 중복 제거
    #   - service, files, vulnerabilities 처리
    def _filter_attack_pps(self, pre_state_dict, post_state_dict):
        pre_key = list(pre_state_dict.keys())
        post_key = list(post_state_dict.keys())

        pps_key_sum = pre_key + post_key
        pps_key = list(set(pps_key_sum))

        resv_pps = ['compromised', 'discovered', 'reachable']
        for e in resv_pps:
            pps_key.remove(e)

        pps_key_sorted = self._readable_sort(pps_key)

        print(f"[+] pps_key [{len(pps_key)}] : {pps_key}")
        print(f"[+] pps_key_sorted [{len(pps_key_sorted)}] : {pps_key_sorted}")
        return pps_key_sorted

        # kjkoo
        #   - (24-0628) 상태 테이블 정렬
        #   - (24-0808) 특징 추가
    def _custom_sort(self, hlist):
        host_status_list = ['Address', 'Compr', 'Reach', 'Discv', 'SHVal', 'DHVal']
        # 13 종
        service_list = ['av', 'av_s', 'av_v',
                        'db', 'db_s', 'db_v',
                        'eternalblue', 'eternalblue_s', 'eternalblue_v',
                        'firewall', 'firewall_s', 'firewall_v',
                        'ftp', 'ftp_s', 'ftp_v',
                        'log4j', 'log4j_s', 'log4j_v',
                        'mail', 'mail_s', 'mail_v',
                        'rdp', 'rdp_s', 'rdp_v',
                        'smb', 'smb_s', 'smb_v',
                        'ssh', 'ssh_s', 'ssh_v',
                        'tomcat', 'tomcat_s', 'tomcat_v',
                        'vnc', 'vnc_s', 'vnc_v',
                        'web', 'web_s', 'web_v']
        # application_list
        # 12 종 (24-0808)
        file_list = ['db_file', 'db_file_s', 'db_file_a', 'db_file_i',
                     'password', 'password_s', 'password_a', 'password_i']

        headers_short_sorted = list()
        for e in hlist:
            if e in host_status_list:
                headers_short_sorted.append(e)
        if 'privilege' in hlist:
            headers_short_sorted.append('privilege')
        if 'os_type' in hlist:
            headers_short_sorted.append('os_type')
        if 'os_status' in hlist:
            headers_short_sorted.append('os_status')
        if 'os_v' in hlist:
            headers_short_sorted.append('os_v')

        for e in hlist:
            if e in service_list:
                headers_short_sorted.append(e)
        for e in hlist:
            if e in file_list:
                headers_short_sorted.append(e)
        return headers_short_sorted

        # kjkoo : (23-0612)
        #   - 가독성 높게 sorting
    def _readable_sort(self, pps_key):
        pps_key_sorted = list()
        pps_key_sorted = sorted(pps_key)

        # if 'discovered_net' in pps_key:
        #     pps_key_sorted.append('discovered_net')
        # if 'platform_'
        #
        return pps_key_sorted

    # kjkoo : [ ] TODO : 개별 공격테크닉(.yaml)이 유효한지 검증 필요 -> 문법 체크..
    # def _validate_single_attack_tech(self):
    #    return

    # kjkoo : 삭제 예정 (23-0522)
    # def _parse_attack_pre_state(self):
    #    attack_pre_state = self.yaml_dict[u.ATTACK_PRE_STATE]
    #    self._validate_attack_pre_state(attack_pre_state)
    #    self.attack_pre_state = attack_pre_state

    # def _validate_attack_pre_state(self, attack_pre_state):
    #     assert len(attack_pre_state) > 0, \
    #         f"{len(attack_pre_state)}. Invalid number of attack_pre_state, must be > 0"
    #     assert len(attack_pre_state) == len(set(attack_pre_state)), \
    #         f"{attack_pre_state}. Attack pre_state must not contain duplicates"

    # def _parse_attack_post_state(self):
    #     attack_post_state = self.yaml_dict[u.ATTACK_POST_STATE]
    #     self._validate_attack_post_state(attack_post_state)
    #     self.attack_post_state = attack_post_state
    #
    # def _validate_attack_post_state(self, attack_post_state):
    #     assert len(attack_post_state) > 0, \
    #         f"{len(attack_post_state)}. Invalid number of attack_post_state, must be > 0"
    #     assert len(attack_post_state) == len(set(attack_post_state)), \
    #         f"{attack_post_state}. Attack post_state must not contain duplicates"

    def _parse_defend_techs(self):
        # load attack techniques
        self.defend_techs_dict = self._load_defend_techs()
        # print(f"attack_techs_dict: \n{list(attack_techs_dict.keys())}")

        # _extract_pre_/post_state
        change_state_dict, stop_state_dict = self._extract_defend_techs_change_stop_state(self.defend_techs_dict)

        # extract pps(pre-/post-state) feature
        self.defend_pps = self._filter_defend_pps(change_state_dict, stop_state_dict)
        print(f"[+] defend_pps_feature [{len(self.defend_pps)}]: {self.defend_pps}")

        # (24-0521)
        #   - attack_pps_dict 설정
        self.defend_pps_dict = DEFENDER_PPS_DICT
        print(f"[+] defend_pps_dict : {self.defend_pps_dict}")

        return

    def _load_defend_techs(self):
        cdir = osp.dirname(osp.abspath(__file__))
        atechs_dir = cdir + '/' + self.yaml_dict[u.DEFENDER_TECHS]
        defend_techs_dict = dict()
        # kjkoo : (23-0613)
        #   - attack_list 정렬 : 테크닉 번호로
        defend_list = os.listdir(atechs_dir)
        defend_list_sorted = sorted(defend_list)
        for i, atech in enumerate(defend_list_sorted):
            fpath = atechs_dir + '/' + atech

            with open(fpath, encoding='UTF-8') as fin:
                content = yaml.load(fin, Loader=yaml.FullLoader)
                # 일단 중복을 체크해.
                if atech not in defend_techs_dict.keys():
                    atech_name = atech.split('.yaml')[0]
                    defend_techs_dict[atech_name] = content

        return defend_techs_dict
        # print(f"attack_techs_dict: \n{list(attack_techs_dict.keys())}")

        # kjkoo : pre_state, post_state 정보 추출

    def _extract_defend_techs_change_stop_state(self, defend_techs_dict):
        self.change_state_key_val_dict = dict()
        self.stop_state_key_val_dict = dict()

        pre_state_key_val_dict = self.change_state_key_val_dict
        post_state_key_val_dict = self.stop_state_key_val_dict

        for dfd_tech in defend_techs_dict.items():
            content = dfd_tech[1]
            atomic_tests = content['atomic_tests']
            atomic_tests_val = atomic_tests[0]
            # pre_state

            if 'source' in atomic_tests_val:
                source = atomic_tests_val['source']
                src_pre_state = source['pre_state']
                # print(f'>> 여기가 source : pre_state 작업이야 !!!')
                pre_state_key_val_dict = self._construct_state_key_val_dict(src_pre_state, pre_state_key_val_dict)

                if 'post_state' in source:
                    src_post_state = source['post_state']
                    # print(f'>> 여기가 source : post_state 작업이야 !!!')
                    post_state_key_val_dict = self._construct_state_key_val_dict(src_post_state,
                                                                                 post_state_key_val_dict)
            elif 'dest' in atomic_tests_val:
                dest = atomic_tests_val['dest']
                dest_pre_state = dest['pre_state']
                # dest_pre_state

                # print(f'>> 여기가 dest : pre_state 작업이야 !!!')
                pre_state_key_val_dict = self._construct_state_key_val_dict(dest_pre_state, pre_state_key_val_dict)
                # print(f">> [dest] pre_state key_val: \n{pre_state_key_val_dict}")

                dest_post_state = dest['post_state']
                # dest_pre_state
                # print(f'>> 여기가 dest : post_state 작업이야 !!!')
                post_state_key_val_dict = self._construct_state_key_val_dict(dest_post_state, post_state_key_val_dict)

        print(f"[+] Defend techniques: {list(defend_techs_dict.keys())}")

        print(f'[+] Defneder pre_state_key_val_dict: {pre_state_key_val_dict}')
        self.change_state_key_val_dict = pre_state_key_val_dict

        print(f'[+] Defneder post_state_key_val_dict: {post_state_key_val_dict}')
        self.stop_state_key_val_dict = post_state_key_val_dict

        return self.change_state_key_val_dict, self.stop_state_key_val_dict

        # kjkoo : pps feacture 추출
        #   - 상태 테이블 컬럼에 사용될 특징 추출
        #   - 중복 제거
        #   - service, files, vulnerabilities 처리

    def _filter_defend_pps(self, change_state_dict, stop_state_dict):
        change_key = list(change_state_dict.keys())
        stop_key = list(stop_state_dict.keys())

        defend_key_sum = change_key + stop_key
        defend_key = list(set(defend_key_sum))

        resv_pps = ['compromised', 'discovered', 'reachable']
        for e in resv_pps:
            if e in defend_key:
                defend_key.remove(e)

        defend_key_sorted = self._readable_sort(defend_key)
        #
        # pps_key_wo_st = [s.lower() for s in pps_key if not "{something}" in s.lower()]
        print(f"[+] Defneder pps_key [{len(defend_key)}] : {defend_key}")
        print(f"[+] Defneder pps_key_sorted [{len(defend_key_sorted)}] : {defend_key_sorted}")
        return defend_key_sorted
