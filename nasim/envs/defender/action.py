import math
import numpy as np
from gym import spaces

from nasim.envs.utils import AccessLevel

def load_action_list(scenario):
    action_list = []
    for address in scenario.address_space:
        for defend_name, defend_def in scenario.defend_techs.items():
            defend_tech = DefendTech(defend_name, address, **defend_def)
            action_list.append(defend_tech)
    return action_list


class Action_Defender:          # defender
    def __init__(self,
                 name,
                 target,
                 cost=1.0,
                 prob=1.0,
                 req_access=AccessLevel.NONE,
                 **kwargs):
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_defend_tech(self):
        return isinstance(self, DefendTech)

    def is_noop(self):
        return isinstance(self, NoOp)

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


class DefendTech(Action_Defender):  # defender action#defender action
    def __init__(self,
                 name,
                 target,
                 cost=1.0,
                 prob=1.0,
                 **kwargs):

        dp_name = kwargs['display_name']
        category = kwargs['category']
        atomic_tests = kwargs['atomic_tests'][0]

        if 'source' in atomic_tests:
            source = atomic_tests['source']
            req_access = self._assign_req_access(source['pre_state'])
        else:
            dest = atomic_tests['dest']
            req_access = self._assign_req_access(dest['pre_state'])

        self.os = None
        if 'source' in atomic_tests:
            source = atomic_tests['source']
            self.os = self._assign_os(source['pre_state'])
        else:
            dest = atomic_tests['dest']
            self.os = self._assign_os(dest['pre_state'])

        self.service = None
        if 'source' in atomic_tests:
            source = atomic_tests['source']
            self.service = self._assign_service(source['pre_state'])
        else :
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

        self.source_pre_state = []
        self.source_post_state = []
        self.dest_pre_state = []
        self.dest_post_state = []

        if 'source' in atomic_tests:
            source = atomic_tests['source']
            self.source_pre_state = atomic_tests['source']['pre_state']
            # source.keys()
            if 'post_state' in source.keys():
                self.source_post_state = atomic_tests['source']['post_state']

        if 'dest' in atomic_tests:
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
        src_pre_out = str('')
        src_post_out = str('')

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


        print_out = f"{super().__str__()}, tid={self.name}, dpname={self.dp_name}, " \
                    f"category={self.category}, remote={self.remote}, " \
                    f"dest_pre_state: {dest_pre_out} || dest_post_state: {dest_post_out}"

        return print_out

    # kjkoo : TODO : self.service, self.access 코드 마무리 해야 함.
    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.os == other.os

class NoOp(Action_Defender):
    def __init__(self, *args, **kwargs):
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)


class DefenderActionResult:     # Defender_Result
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
            undefined_error=self.undefined_error,
        )

    def __str__(self):
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
        return "\n".join(output)


# Defender action Space
class FlatDefenderActionSpace(spaces.Discrete):
    def __init__(self, scenario):
        self.actions = load_action_list(scenario)
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        assert isinstance(action_idx, int), \
            ("When using flat action space, action must be an integer"
             f" or an Action object: {action_idx} is invalid")
        return self.actions[action_idx]
