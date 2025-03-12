import math
import numpy as np
from gym import spaces

from nasim.envs.utils import AccessLevel

def load_action_list(scenario):
    action_list = []
    for address in scenario.address_space:
        for atk_name, atk_def in scenario.attack_techs.items():
            attack_tech = AttackTech(atk_name, address, **atk_def)
            action_list.append(attack_tech)
    return action_list


class Action:
    def __init__(self,
                 name,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_exploit(self):
        return isinstance(self, Exploit)

    def is_privilege_escalation(self):
        return isinstance(self, PrivilegeEscalation)

    def is_scan(self):
        return isinstance(self, (ServiceScan, OSScan, SubnetScan, ProcessScan))

    def is_remote(self):
        return isinstance(self, (ServiceScan, OSScan, Exploit))

    def is_service_scan(self):
        return isinstance(self, ServiceScan)

    def is_os_scan(self):
        return isinstance(self, OSScan)

    def is_subnet_scan(self):
        return isinstance(self, SubnetScan)

    def is_process_scan(self):
        return isinstance(self, ProcessScan)


    # kjkoo (23-0407)
    def is_attack_tech(self):
        return isinstance(self, AttackTech)


    def is_noop(self):
        return isinstance(self, NoOp)

    # kjkoo : 아래는 어떻게 확인하는겨?

    def __str__(self):
        return (f"{self.__class__.__name__}: "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access

class AttackTech(Action):
    # kjkoo : cost, prob에 대한 정의는 다시할 필요가 있다.
    def __init__(self,
                 name,
                 target,
                 cost=1.0,
                 prob=1.0,
                 **kwargs):

        dp_name = kwargs['display_name']
        category = kwargs['category']
        atomic_tests = kwargs['atomic_tests'][0]

        # req_access 검색 및 공격테크닉이 권한을 필요로 하면 update
        # kjkoo : (24-0517) 수정
        if atomic_tests['technique_remote']:
            source = atomic_tests['source']
            req_access = self._assign_req_access(source['pre_state'])
        else:
            dest = atomic_tests['dest']
            req_access = self._assign_req_access(dest['pre_state'])

        self.os = None
        if atomic_tests['technique_remote']:
            source = atomic_tests['source']
            self.os = self._assign_os(source['pre_state'])
        else:
            dest = atomic_tests['dest']
            self.os = self._assign_os(dest['pre_state'])

        self.service = None
        if atomic_tests['technique_remote']:
            dest = atomic_tests['dest']
            self.service = self._assign_service(dest['pre_state'])

        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)


        self.name = name
        self.category = category
        self.dp_name = dp_name
        self.atomic_tests = atomic_tests
        self.remote = atomic_tests['technique_remote']

        # kjkoo
        #   - (24-0517) remote 공격은 source/dest 존재, local 공격은 dest만 존재
        self.source_pre_state = []
        self.source_post_state = []
        self.dest_pre_state = []
        self.dest_post_state = []
        if atomic_tests['technique_remote']:
            source = atomic_tests['source']
            self.source_pre_state = atomic_tests['source']['pre_state']
            # source.keys()
            if 'post_state' in source.keys():
                self.source_post_state = atomic_tests['source']['post_state']
            #print(f'>> source_pre_state : {self.source_pre_state}')
            #print(f'>> source_post_state : {self.source_post_state}')

        dest = atomic_tests['dest']
        self.dest_pre_state = atomic_tests['dest']['pre_state']
        # source.keys()
        if 'post_state' in dest.keys():
            self.dest_post_state = atomic_tests['dest']['post_state']

    def _assign_req_access(self, state_list):
        stop = False
        req_access = 0
        for idx in range(len(state_list)):
            for key, value in state_list[idx].items():
                st_key = str(key).lower()
                st_value = str(value).lower()
                # state key에 privilege가 있는지 확인
                if st_key == 'privilege':
                    if st_value == 'user':
                        req_access = 1
                    elif st_value == 'admin':
                        req_access = 2
                    elif st_value == 'web':
                        req_access = 3
                    elif st_value == 'db':
                        req_access = 4
                    else:
                        req_access = 0
                    stop = True
                    break
            if stop:
                break
        return req_access

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

    def _assign_os(self, state_list):
        stop = False
        os = 0
        for idx in range(len(state_list)):
            for key, value in state_list[idx].items():
                st_key = str(key).lower()
                st_value = str(value).lower()
                # state key에 os_type가 있는지 확인
                if st_key == 'os_type':
                    os = self._string_split_strip(st_value)
                    stop = True
                    break
            if stop:
                break
        return os

    def _assign_service(self, state_list):
        stop = False
        service = 0
        service_type = ['rdp', 'vnc', 'web', 'db', 'firewall', 'av', 'mail', 'ssh', 'ftp', 'log4j']
        for idx in range(len(state_list)):
            for key, value in state_list[idx].items():
                st_key = str(key).lower()
                # print(f'st_key : {st_key}')
                if st_key in service_type:
                    service = st_key
                    # print(f'service : {service}')
                    stop = True
                    break
            if stop:
                break
        return service



    def __str__(self):
        # kjkoo
        #   - (24-0517) pps-v2에 맞게 수정
        src_pre_out = str('')
        src_post_out = str('')
        if self.remote:
            for pre in self.source_pre_state:
                for key, val in pre.items():
                    src_pre_out = src_pre_out + key.lower() + '=' + str(val).lower() + ', '
            for post in self.source_post_state:
                for key, val in post.items():
                    src_post_out = src_post_out + key.lower() + '=' + str(val).lower() + ', '

        dest_pre_out = str('')
        dest_post_out = str('')
        for pre in self.dest_pre_state:
            for key, val in pre.items():
                dest_pre_out = dest_pre_out + key.lower() + '=' + str(val).lower() + ', '
        for post in self.dest_post_state:
            for key, val in post.items():
                dest_post_out = dest_post_out + key.lower() + '=' + str(val).lower() + ', '

        if self.remote:
            print_out = f"{super().__str__()}, tid={self.name}, dpname={self.dp_name}, " \
                        f"category={self.category}, remote={self.remote}, " \
                        f"source_pre_state: {src_pre_out} || source_post_state: {src_post_out}, " \
                        f"dest_pre_state: {dest_pre_out} || dest_post_state: {dest_post_out}"
        else:
            print_out = f"{super().__str__()}, tid={self.name}, dpname={self.dp_name}, " \
                        f"category={self.category}, remote={self.remote}, " \
                        f"dest_pre_state: {dest_pre_out} || dest_post_state: {dest_post_out}"

        return print_out

    # kjkoo : TODO : self.service, self.access 코드 마무리 해야 함.
    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.os == other.os

class Exploit(Action):
    def __init__(self,
                 name,
                 target,
                 cost,
                 service,
                 os=None,
                 access=0,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.os = os
        self.service = service
        self.access = access

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"service={self.service}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service \
            and self.os == other.os \
            and self.access == other.access


class PrivilegeEscalation(Action):
    def __init__(self,
                 name,
                 target,
                 cost,
                 access,
                 process=None,
                 os=None,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.access = access
        self.os = os
        self.process = process

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"process={self.process}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.process == other.process \
            and self.os == other.os \
            and self.access == other.access


class ServiceScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class OSScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class SubnetScan(Action):
    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class ProcessScan(Action):
    """A Process Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        super().__init__("process_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class NoOp(Action):
    def __init__(self, *args, **kwargs):
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)


class ActionResult:
    def __init__(self,
                 success,
                 value=0.0,
                 services=None,
                 os=None,
                 processes=None,
                 access=None,
                 discovered=None,
                 # kjkoo : (23-0616)
                 compromised=False,
                 connection_error=False,
                 permission_error=False,
                 undefined_error=False,
                 newly_discovered=None):
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.processes = {} if processes is None else processes
        self.access = {} if access is None else access
        self.discovered = {} if discovered is None else discovered
        self.compromised = compromised
        self.connection_error = connection_error
        self.permission_error = permission_error
        self.undefined_error = undefined_error
        if newly_discovered is not None:
            self.newly_discovered = newly_discovered
        else:
            self.newly_discovered = {}

    def info(self):
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            processes=self.processes,
            access=self.access,
            discovered=self.discovered,
            connection_error=self.connection_error,
            permission_error=self.permission_error,
            newly_discovered=self.newly_discovered
        )

    def __str__(self):
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
        return "\n".join(output)


class FlatActionSpace(spaces.Discrete):
    def __init__(self, scenario):
        self.actions = load_action_list(scenario)
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        assert isinstance(action_idx, int), \
            ("When using flat action space, action must be an integer"
             f" or an Action object: {action_idx} is invalid")
        return self.actions[action_idx]


class ParameterisedActionSpace(spaces.MultiDiscrete):
    action_types = [
        Exploit,
        PrivilegeEscalation,
        ServiceScan,
        OSScan,
        SubnetScan,
        ProcessScan
    ]

    def __init__(self, scenario):
        self.scenario = scenario
        self.actions = load_action_list(scenario)

        nvec = [
            len(self.action_types),
            len(self.scenario.subnets)-1,
            max(self.scenario.subnets),
            self.scenario.num_os+1,
            self.scenario.num_services,
            self.scenario.num_processes
        ]

        super().__init__(nvec)

    def get_action(self, action_vec):
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("When using parameterised action space, action must be an Action"
             f" object, a list or a numpy array: {action_vec} is invalid")
        a_class = self.action_types[action_vec[0]]
        # need to add one to subnet to account for Internet subnet
        subnet = action_vec[1]+1
        host = action_vec[2] % self.scenario.subnets[subnet]

        target = (subnet, host)

        if a_class not in (Exploit, PrivilegeEscalation):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        if a_class == Exploit:
            service = self.scenario.services[action_vec[4]]
            a_def = self._get_exploit_def(service, os)
        else:
            proc = self.scenario.processes[action_vec[5]]
            a_def = self._get_privesc_def(proc, os)

        if a_def is None:
            return NoOp()
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class):
        """Get the constants for scan actions definitions """
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")
        return {"cost": cost}

    def _get_exploit_def(self, service, os):
        """Check if exploit parameters are valid """
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]

    def _get_privesc_def(self, proc, os):
        """Check if privilege escalation parameters are valid """
        pe_map = self.scenario.privesc_map
        if proc not in pe_map:
            return None
        if os not in pe_map[proc]:
            return None
        return pe_map[proc][os]
