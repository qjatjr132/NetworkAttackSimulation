import numpy as np
import random

from nasim.envs.utils import AccessLevel
from nasim.envs.attacker.action import ActionResult

#PRINT_OUT = True
PRINT_OUT = False

# kjkoo : (24-0625)
USE_INC_RWD = False

class HostVector:
    address_space_bounds = (0, 0)

    num_os = None
    os_idx_map = {}
    num_services = None
    service_idx_map = {}
    num_processes = None
    process_idx_map = {}

    num_attack_pps = None
    attack_pps_idx_map = {}

    recv_files_list = list()

    state_size = None

    _subnet_address_idx = 0
    _host_address_idx = None
    _compromised_idx = None
    _reachable_idx = None
    _discovered_idx = None
    _value_idx = None
    _discovery_value_idx = None
    _access_idx = None

    _attack_pps_start_idx = None            # (23-0526) 추가
    _attack_pre_states_start_idx = None     # 삭제 가능
    _attack_post_states_start_idx = None    # 삭제 가능

    USE_STD_VALUE = True

    check_seq_action_dict = dict()


    def __init__(self, vector):
        self.vector = vector

    @classmethod
    def _set_compromised_host(cls, host, vector, attack_pps, attack_pps_dict):
        if host.compromised:
            # os 설정 (os_type(int), os_status(int))
            for os_key, os_val in host.os.items():
                if os_val == True:
                    # os_type 설정
                    vector[cls._get_attack_pps_idx(attack_pps.index('os_type'))] = int(
                        attack_pps_dict["os_type_dict"][os_key.lower()])
                    # os_status 설정 (running)
                    vector[cls._get_attack_pps_idx(attack_pps.index('os_status'))] = int(
                        attack_pps_dict["os_status_dict"]['running'])

            # service 설정 (service_type(bool), service_status(int))
            for service_key, service_val in host.services.items():
                if service_val == True:
                    # service_type 설정
                    vector[cls._get_attack_pps_idx(attack_pps.index(service_key.lower()))] = True
                    # service_status 설정
                    vector[cls._get_attack_pps_idx(attack_pps.index(service_key.lower() + '_s'))] = int(
                        attack_pps_dict["service_status_dict"]['running'])

            # privilege 설정
            vector[cls._get_attack_pps_idx(attack_pps.index('privilege'))] = host.access
        return vector

    @classmethod
    def _set_file_access(cls, host, vector, attack_pps, attack_pps_dict):
        if len(host.files) > 0:
            for f_key, f_val in host.files.items():
                if f_val:
                    # file 이름 분리
                    f_key_name = f_key.lower().split('_id')[0]

                    # file_access 설정
                    f_key_name_a = f_key_name + '_a'      # 예: password_a
                    vector[cls._get_attack_pps_idx(attack_pps.index(f_key_name_a))] = int(
                        attack_pps_dict["file_access_dict"]['read'])
        return vector

    # kjkoo :
    #   - (24-0829) file_id 초기값 설정
    @classmethod
    def _set_file_id(cls, host, vector, attack_pps, attack_pps_dict):
        if len(host.files) > 0:
            for f_key, f_val in host.files.items():
                if f_val:
                    if '_ID' in f_key:
                        # file 이름 분리
                        f_key_name = f_key.lower().split('_id')[0]

                        # file_id 설정
                        f_key_name_i = f_key_name + '_i'  # 예: password_i
                        f_key_name_v = 'id' + f_key.lower().split('_id')[1]    # 예: '-1'

                        vector[cls._get_attack_pps_idx(attack_pps.index(f_key_name_i))] = int(
                            attack_pps_dict["file_id_dict"][f_key_name_v])
        return vector


    @classmethod
    def vectorize(cls, host, address_space_bounds, attack_pps, attack_pps_dict, vector=None):
        if cls.address_space_bounds is None:
            cls._initialize(
                address_space_bounds, host.services, host.os, host.processes,
                attack_pps       # kjkoo 추가
            )

        if vector is None:
            vector = np.zeros(cls.state_size, dtype=np.float32)
        else:
            assert len(vector) == cls.state_size

        vector[cls._subnet_address_idx + host.address[0]] = 1
        vector[cls._host_address_idx + host.address[1]] = 1
        vector[cls._compromised_idx] = int(host.compromised)
        vector[cls._reachable_idx] = int(host.reachable)
        vector[cls._discovered_idx] = int(host.discovered)
        vector[cls._value_idx] = host.value
        vector[cls._discovery_value_idx] = host.discovery_value

        for pps_num, pps_key in enumerate(attack_pps):
            vector[cls._get_attack_pps_idx(pps_num)] = int(-1)

        vector = cls._set_compromised_host(host, vector, attack_pps, attack_pps_dict)

        vector = cls._set_file_access(host, vector, attack_pps, attack_pps_dict)

        vector = cls._set_file_id(host, vector, attack_pps, attack_pps_dict)

        return cls(vector)

    @classmethod
    def vectorize_random(cls, host, address_space_bounds, vector=None):
        # kjkoo : 23-1027
        #   - 코드 에러 방지 차원 : pseudo-code
        #   - vectorize_random(attack_pps) 형태로 바꿔야 함
        attack_pps = []
        hvec = cls.vectorize(host, vector, attack_pps)
        # random variables
        for srv_num in cls.service_idx_map.values():
            srv_val = np.random.randint(0, 2)
            hvec.vector[cls._get_service_idx(srv_num)] = srv_val

        chosen_os = np.random.choice(list(cls.os_idx_map.values()))
        for os_num in cls.os_idx_map.values():
            hvec.vector[cls._get_os_idx(os_num)] = int(os_num == chosen_os)

        for proc_num in cls.process_idx_map.values():
            proc_val = np.random.randint(0, 2)
            hvec.vector[cls._get_process_idx(proc_num)] = proc_val
        return hvec

    @property
    def compromised(self):
        return self.vector[self._compromised_idx]

    @compromised.setter
    def compromised(self, val):
        self.vector[self._compromised_idx] = int(val)

    @property
    def discovered(self):
        return self.vector[self._discovered_idx]

    @discovered.setter
    def discovered(self, val):
        self.vector[self._discovered_idx] = int(val)

    @property
    def reachable(self):
        return self.vector[self._reachable_idx]

    @reachable.setter
    def reachable(self, val):
        self.vector[self._reachable_idx] = int(val)

    @property
    def address(self):
        return (
            self.vector[self._subnet_address_idx_slice()].argmax(),
            self.vector[self._host_address_idx_slice()].argmax()
        )

    @property
    def value(self):
        return self.vector[self._value_idx]

    @property
    def discovery_value(self):
        return self.vector[self._discovery_value_idx]

    @property
    def access(self):
        privilege = self.vector[self._get_attack_pps_idx(self.attack_pps_idx_map['privilege'])]
        return privilege

    @access.setter
    def access(self, val):
        self.vector[self._get_attack_pps_idx(self.attack_pps_idx_map['privilege'])] = int(val)

    @property
    def services(self):
        services = {}
        for srv, srv_num in self.service_idx_map.items():
            services[srv] = self.vector[self._get_service_idx(srv_num)]
        return services

    @property
    def os(self):
        os = {}
        for os_key, os_num in self.os_idx_map.items():
            os[os_key] = self.vector[self._get_os_idx(os_num)]
        return os

    @property
    def processes(self):
        processes = {}
        for proc, proc_num in self.process_idx_map.items():
            processes[proc] = self.vector[self._get_process_idx(proc_num)]
        return processes

    @property
    def attack_pps(self):
        attack_pps = {}
        for pps, pps_num in self.attack_pps_idx_map.items():
            attack_pps[pps] = self.vector[self._get_attack_pps_idx(pps_num)]
        return attack_pps

    @attack_pps.setter
    def attack_pps(self, pps_key_val):
        pps_key = pps_key_val[0]
        val = pps_key_val[1]
        pps_num = self.attack_pps_idx_map[pps_key]
        self.vector[self._get_attack_pps_idx(pps_num)] = int(val)

    @property
    def recv_files(self):
        return self.recv_files_list

    @recv_files.setter
    def recv_files(self, flist):
        self.recv_files_list = flist


    def is_running_service(self, srv):
        srv_num = self.service_idx_map[srv]
        return bool(self.vector[self._get_service_idx(srv_num)])

    def is_running_os(self, os):
        os_num = self.os_idx_map[os]
        return bool(self.vector[self._get_os_idx(os_num)])

    def is_running_process(self, proc):
        proc_num = self.process_idx_map[proc]
        return bool(self.vector[self._get_process_idx(proc_num)])

    def is_running_pps(self, pps):
        pps_num = self.attack_pps_idx_map[pps]
        return self.vector[self._get_attack_pps_idx(pps_num)]

    def _check_host_defended(self, network, action_state_key, host_state_value):
        # defended 설정 대상 : os_status, service_status, file_status
        defended_value = network._find_value_from_pps_dict(action_state_key, 'defended')
        # defended 설정 되어있으면 True(1)
        if int(host_state_value) & defended_value:
            return 1
        else:
            return 0

    def _check_pps_host(self, network, dhost, action_pre_state, source):
        check_result = False
        shost = self
        if source:
            host = shost
        else:
            host = dhost

        # action.target(source) 호스트의 상태가 공격테크닉(action)의 source:pre_state 조건과 일치하는지 확인
        # 1) source 호스트의 pre_state 모든 조건을 만족해야 공격 테크닉 수행 가능
        for pre in action_pre_state:
            if PRINT_OUT: print(f'>> pre_state : {pre}')
            for key, value in pre.items():
                action_pre_state_key = str(key).lower()
                action_pre_state_value_str = str(value).lower()
                # state_value가 1개 이상이면 리스트로 생성
                action_pre_state_value_list = network._string_split_strip(action_pre_state_value_str)

                if len(action_pre_state_value_list) > 1:
                    # 호스트의 현재상태와 공격테크닉의 pre_state가 일치하는지 확인
                    # [1] action 필드(key, value)의 value가 'not admin'인 경우,
                    if action_pre_state_value_str == 'not admin':
                        # (24-0628)
                        if int(host.attack_pps[action_pre_state_key]) == -1:
                            host_state_value = 0
                        else:
                            host_state_value = host.attack_pps[action_pre_state_key]
                        action_value = network._find_value_from_pps_dict(shost, dhost, action_pre_state_key, 'admin')   # source, dest
                        # 호스트 상태 값과 action 상태 값 비교
                        if int(host_state_value) == int(action_value):
                            if PRINT_OUT: print(f'>> Host Privilege is already "Admin" : {host_state_value} : {float(action_value)}')
                            check_result = False
                            return check_result
                        else:
                            continue        # check success

                    # [2] action필드의 value가 'or'를 포함하는 경우
                    elif 'or' in action_pre_state_value_list:
                        for s_val in action_pre_state_value_list:
                            if s_val == 'or':
                                continue
                            else:
                                # (24-0628)
                                if int(host.attack_pps[action_pre_state_key]) == -1:
                                    host_state_value = 0
                                else:
                                    host_state_value = host.attack_pps[action_pre_state_key]
                                action_value = network._find_value_from_pps_dict(shost, dhost, action_pre_state_key, s_val)
                                # 호스트 상태 값과 action 상태 값 비교
                                #   - host_state_value는 one-hot coding으로 여러값을 중복해서 가질수 있음.
                                #   - bit-masking으로 비교해야 함.
                                # 하나라도 만족하면 True

                                if int(host_state_value) & int(action_value):
                                    if PRINT_OUT: print(
                                        f'>> pre_state가 or를 포함하는 경우 : '
                                        f'pre_state: {action_pre_state_key}:  {action_pre_state_value_str}, '
                                        f'host_state: {host_state_value}')
                                    check_result = True
                                    return check_result
                                else:
                                    continue  # check success
                        check_result = False
                        return check_result

                    # kjkoo : (24-0820) 추가 : 'not Unavailable' 처리 로직 추가
                    # 예: {service}_s 상태가 'not Unavailable' 이면, Unavailable의 인덱스 값과, 호스트의 {service}_s가
                    # 같지 않으면 pre_state 조건을 만족
                    elif (action_pre_state_value_list[0] == 'not') and (len(action_pre_state_value_list)==2):
                        s_val = action_pre_state_value_list[1]      # pre_state의 value
                        # 호스트 pre_state 상태값
                        if int(host.attack_pps[action_pre_state_key]) == -1:
                            host_state_value = 0
                        else:
                            host_state_value = host.attack_pps[action_pre_state_key]
                        # action의 pre_state 상태값 (사전에 정의된 값)
                        action_value = network._find_value_from_pps_dict(shost, dhost, action_pre_state_key, s_val)
                        # 호스트 상태 값과 action 상태 값 비교 : bitmasking 비교
                        if int(host_state_value) & int(action_value):
                            # 호스트 상태가 action_value를 포함하는 경우 : pre_state 조건을 만족하지 않는 경우
                            check_result = False
                        else:
                            # pre_state 조건을 만족하는 경우
                            check_result = True
                        return check_result


                # action 상태값 가져오기
                action_value = network._find_value_from_pps_dict(shost, dhost, action_pre_state_key,
                                                              action_pre_state_value_str)
                # 호스트 상태 값 가져오기
                if action_pre_state_key == 'compromised':
                    host_state_value = host.compromised
                elif action_pre_state_key == 'reachable':
                    host_state_value = host.reachable
                elif action_pre_state_key == 'discovered':
                    host_state_value = host.discovered
                else:
                    # kjkoo
                    #   - (24-0628) -1은 상태 테이블의 default 설정 (False) : DQNAgent 학습을 위해 0보다 -1을 선택
                    if int(host.attack_pps[action_pre_state_key]) == -1:
                        host_state_value = 0
                    else:
                        host_state_value = host.attack_pps[action_pre_state_key]   # 호스트 상태 : 정수값

                # 호스트 상태 값과 action 상태 값 비교
                #   1) 정확하게 일치하는지를 비교하고
                #   2) bit-masking 했을 때 조건을 만족하는지 확인한다.

                if int(host_state_value) & int(action_value):
                    if action_pre_state_key == 'privilege':
                        if int(host_state_value) == int(action_value):
                            continue
                        else:
                            if PRINT_OUT: print(f'>> pps not equal: host={host_state_value} : action={float(action_value)}')
                            check_result = False
                            return check_result
                    else:
                        continue
                else:
                    if int(host_state_value) != int(action_value):
                        if PRINT_OUT: print(f'>> pps not equal: host={host_state_value} : action={float(action_value)}')
                        check_result = False
                        return check_result

        if PRINT_OUT: print(f'>> pre_state : all satisfied !!')
        check_result = True
        return check_result

    def _update_host_status(self, host, state_key, state_value):
        state_actions = {
            'compromised': 'compromised',
            'reachable': 'reachable',
            'discovered': 'discovered'
            # 다른 상태 키와 속성들을 추가할 수 있습니다.
        }
        value = 0
        compromised = False
        num_update_ps_hs = 0

        if state_key in state_actions:
            # host의 기존 상태 값을 확인
            host_state_value = getattr(host, state_actions[state_key])
            if PRINT_OUT: print(f'>> 호스트 상태값 (host.{state_key}) : {host_state_value}')

            # host의 기존 상태 값과 (action)state_value (=post_state_value)가 다른 경우 업데이트
            if host_state_value != state_value:
                setattr(host, state_actions[state_key], state_value)    # post_state 값 업데이트
                num_update_ps_hs += 1
                if state_key == 'compromised':
                    value = host.value  # host value 할당
                    compromised = state_value
        else:
            raise ValueError(f"Unknown post state key: {state_key}")

        return value, compromised, num_update_ps_hs

    def _check_multi_status(self, network, host, key, value, file_exfilt_flag):
        num_update_ps_ms = 0    # 새롭게 업데이트되는 상태가 있으면 1 증가

        # _check_host_file : host가 action-key 파일을 가지고 있는지 확인
        if key in network.attack_pps_dict['file_type_list']:
            # file exfiltration 관련 공격은 dest 호스트가 파일을 가지고 있지 않아도 호스트 상태 벡터를 업데이트 한다. (24-0702)
            if file_exfilt_flag:
                host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
                # host_state_value에 이미 value(action-post_state)가 설정되어 있다면 건너뜀
                if int(host_state_value) & int(value):
                    return host_state_value, num_update_ps_ms  # 기존 host_state 설정값을 유지
                else:
                    value_result = int(host_state_value) + int(value)
                    num_update_ps_ms += 1
                    return value_result, num_update_ps_ms
            else:
                # action을 수행하기 위해서는 host가 action-key 파일을 가지고 있어야 한다.
                for h_file in network.hosts[host.address].files.items():     # action target 호스트의 file을 확인
                    # h_file_name = h_file[0].lower()     # string
                    # kjkoo : (24-0829) 'Password_ID-1' 형식의 파일 처리 가능토록 변경
                    #h_file_name = h_file[0].lower().split('_id')[0]  # string
                    h_file_name = h_file[0].split('_ID')[0].lower()  # string
                    h_file_value = h_file[1]            # T/F
                    if (key == h_file_name) and h_file_value:   # action key 파일을 호스트가 가지고 있음.
                        host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
                        # host_state_value에 이미 value(action-post_state)가 설정되어 있다면 건너뜀
                        if int(host_state_value) & int(value):
                            return host_state_value, num_update_ps_ms  # 기존 host_state 설정값을 유지
                        else:
                            if PRINT_OUT: print(f'>> 호스트 파일 : {h_file_name}, {h_file_value}')
                            value_result = int(host_state_value) + int(value)
                            num_update_ps_ms += 1
                            return value_result, num_update_ps_ms
                # action-key와 타겟 호스트의 파일 조건이 맞지 않는 경우
                return host.attack_pps[key], num_update_ps_ms   # (기존 상태값, 0)

        # 중복으로 체크되는 호스트 상태인 경우 기존 호스트 상태값과 action 성공 후 post_state 값을 더한다.
        # 1) file_s
        if key in network.attack_pps_dict['file_status_list']:
            if file_exfilt_flag:
                host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
                # host_state_value에 이미 value가 설정되어 있다면 건너뜀
                if int(host_state_value) & int(value):
                    return host_state_value, num_update_ps_ms  # 기존 host_state 설정값을 유지
                else:
                    value_result = host_state_value + value
                    num_update_ps_ms += 1
                    return value_result, num_update_ps_ms
            else:
                # 호스트의 file이 True이면 file_s값을 action.pps로 설정하고, 그렇지 않으면 pass
                file_name = key.split('_')[0]
                if int(host.attack_pps[file_name]) == 1:
                    # host_state_value = host.attack_pps[key]
                    # (24-0628) 상태 테이블의 디폴트 값을 -1로 설정. DQNAgent 학습을 위해.
                    host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
                    # host_state_value에 이미 value가 설정되어 있다면 건너뜀
                    if int(host_state_value) & int(value):
                        return host_state_value, num_update_ps_ms      # 기존 host_state 설정값을 유지
                    else:
                        value_result = host_state_value + value
                        num_update_ps_ms += 1
                        return value_result, num_update_ps_ms
                else:
                    return host.attack_pps[key], num_update_ps_ms   # (기존 상태값, 0)

        # (23-0812) multi_value를 갖는 pps 필드 처리
        multi_value_status_list = (['os_v']
                                   + network.attack_pps_dict['service_status_list']
                                   + network.attack_pps_dict['service_vuln_list']
                                   + network.attack_pps_dict['process_status_list']
                                   + network.attack_pps_dict['process_privilege_list']
                                   + network.attack_pps_dict['application_status_list']
                                   + network.attack_pps_dict['application_vuln_list']
                                   + network.attack_pps_dict['file_access_list']
                                   + network.attack_pps_dict['file_id_list'])

        if key in multi_value_status_list:
            host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
            # host_state_value에 이미 value(action)가 설정되어 있다면 건너뜀
            if int(host_state_value) & int(value):
                return host_state_value, num_update_ps_ms  # 기존 host_state 설정값을 유지
            else:
                value_result = host_state_value + value
                num_update_ps_ms += 1
                return value_result, num_update_ps_ms

        host_state_value = 0 if int(host.attack_pps[key]) == -1 else host.attack_pps[key]
        if int(host_state_value) == int(value):
            return host_state_value, num_update_ps_ms
        else:
            value_result = value        # 새로운 action value(post_state:value)로 호스트 상태값 업데이트
            num_update_ps_ms += 1
            return value_result, num_update_ps_ms

    def _add_reward(self, network, host, post_state_key, post_state_value, post_state_value_str):
        reward = 0

        if post_state_key == 'privilege':
            if post_state_value_str == 'admin':      # admin
                reward = 5
        if post_state_key == 'password_s':
            if post_state_value_str == 'discovered':
                reward = 3
            if post_state_value_str == 'dumped':
                reward = 3
            if post_state_value_str == 'leaked':
                reward = 5
            if post_state_value_str == 'cracked':
                reward = 5
            if post_state_value_str == 'received':
                reward = 3

        return reward

    def _check_file_exfiltration(self, network, host, dhost, post_state_list):
        file_exfilt_flag = False
        for post in post_state_list:
            if PRINT_OUT: print(f'>> post_state : {post}')
            for key, value in post.items():
                action_post_state_key = str(key).lower()
                action_post_state_value_str = str(value).lower()
                # (24-0815) value가 여러개일 수 있음
                #   - 예: password_a : read, write, execute
                action_value_list = action_post_state_value_str.split(', ')
                for action_ps_value_str in action_value_list:
                    # kjkoo : (24-0830) source(RL-target), dest 호스트 추가
                    action_value = network._find_value_from_pps_dict(self,                 # source
                                                                     dhost,       # host
                                                                     action_post_state_key,
                                                                     action_ps_value_str)
                    if action_post_state_key in network.attack_pps_dict['file_status_list']:
                        # post_state_value는 값이 1개 => (24-0815) 값이 여러개 일 수 있음.
                        #   - 예: password_a : read, write, execute
                        if action_ps_value_str == 'received':
                            host_state_value = 0 if int(host.attack_pps[action_post_state_key]) == -1 \
                                else host.attack_pps[action_post_state_key]
                            # host_state_value에 이미 value가 설정되어 있다면 건너뜀
                            if int(host_state_value) & int(action_value):
                                file_exfilt_flag = False
                            else:
                                file_exfilt_flag = True
        return file_exfilt_flag

    def _update_post_state(self, network, shost, dhost, action, source):
        reward_value = 0
        compromised = False
        num_update_ps = 0
        host_state_updated = False

        host = self if source else dhost

        post_state_list = action.source_post_state if source else action.dest_post_state

        file_exfilt_flag = self._check_file_exfiltration(network, host, dhost, post_state_list)

        for post in post_state_list:
            if PRINT_OUT: print(f'>> post_state : {post}')
            for key, value in post.items():
                action_post_state_key = str(key).lower()
                action_post_state_value_str = str(value).lower()
                action_value_list = action_post_state_value_str.split(', ')
                for action_ps_value_str in action_value_list:
                    action_value = network._find_value_from_pps_dict(self, dhost, action_post_state_key,
                                                                     action_ps_value_str)

                    if action_post_state_key in network.attack_pps_dict['host_status_list']:
                        if PRINT_OUT: print(f'>> post key: {action_post_state_key}')
                        val, comp, num_update_ps_hs = self._update_host_status(host, action_post_state_key, action_value)
                        reward_value += val
                        compromised = comp
                        num_update_ps += num_update_ps_hs
                        continue

                    if action_post_state_key in list(host.attack_pps.keys()):
                        if PRINT_OUT: print(f'>> post key: {action_post_state_key}, value: {action_value}')

                        action_value, num_update_ps_ms = self._check_multi_status(network, host, action_post_state_key, action_value, file_exfilt_flag)
                        num_update_ps += num_update_ps_ms
                        host.attack_pps = [action_post_state_key, action_value]
                        if num_update_ps_ms > 0:
                            reward_value += self._add_reward(network, host, action_post_state_key, action_value, action_post_state_value_str)
                        else:
                            reward_value += 0
                        continue

        # TODO : compromised=True가 되면 value += 호스트value를 하지만, 나머지는 동일하게 +2를 해도 될 듯.
        if self.USE_STD_VALUE:
            if num_update_ps == 0:
                reward_value = 0
                host_state_updated = False
            else:
                reward_value += 3       # (24-0626)
                host_state_updated = True

        return reward_value, compromised, host_state_updated

    def _update_source_post_state(self, network, shost, dhost, action):   # rshost == self
        reward_value = 0
        compromised = False
        if len(action.source_post_state) > 0:
            reward_value, compromised, host_state_updated = self._update_post_state(network, shost, dhost, action, source=True)
        return reward_value

    def _update_dest_post_state(self, network, shost, dhost, action):
        reward_value = 0
        compromised = False
        reward_value, compromised, host_state_updated = self._update_post_state(network, shost, dhost, action, source=False)
        return reward_value, compromised, host_state_updated

    def _check_seq_action_failed(self, action):
        if action.target in self.check_seq_action_dict.keys():
            # 호스트가 등록 되어 있으면, 연속/중복 여부 체크
            prev_fail_count = self.check_seq_action_dict[action.target]['seq_fail_count']
            prev_action_name = self.check_seq_action_dict[action.target]['prev_action_name']
            if action.name == prev_action_name:     # 중복
                self.check_seq_action_dict[action.target]['seq_fail_count'] = prev_fail_count + 1
            else:
                self.check_seq_action_dict[action.target]['seq_fail_count'] = 0
                self.check_seq_action_dict[action.target]['prev_action_name'] = action.name
        else:
            # 새로운 호스트는 등록한다.
            self.check_seq_action_dict[action.target] = dict()
            self.check_seq_action_dict[action.target]['seq_fail_count'] = 0
            self.check_seq_action_dict[action.target]['prev_action_name'] = action.name
        return self.check_seq_action_dict[action.target]['seq_fail_count']

    def perform_action_with_pps_v2(self, network, next_network_state, action, host_literal):

        pps_list = list(self.attack_pps.keys())

        if action.is_attack_tech():
            # pre_state 조건을 만족하는지 확인
            #   - host_vector만으로 작업할 수 있는지 확인하고, 안되면 host_literal 받아라.

            # 현재 action name
            #cur_action_name = action.name
            inc_rwd = 0

            # remote 공격
            # remote 공격이면, action.target이 source host임
            # 공격테크닉의 destination host는 주소공간의 모든 호스트가 대상이며, dest의 pre_state 조건을 만족하는 호스트에 대해 공격 성공
            # rshost : remote source host
            # rdhost : remote dest host
            if action.remote:
                if PRINT_OUT: (f'>> remote 공격 : 단일 호스트 대상 ')
                rshost = self.copy()   # remote 공격이면, action.target이 source host임

                # action.target(source) 호스트의 상태가 공격테크닉(action)의 source:pre_state 조건과 일치하는지 확인
                # 1) source 호스트의 pre_state 모든 조건을 만족해야 공격 테크닉 수행 가능
                check_source_pps_result = self._check_pps_host(network, rshost, action.source_pre_state, source=True)
                if not check_source_pps_result:
                    if USE_INC_RWD:
                        inc_rwd = -self._check_seq_action_failed(action)
                    else:
                        inc_rwd = 0
                    next_state = rshost
                    return next_state, ActionResult(False, value=inc_rwd, #value=0,
                                                undefined_error=True)
                if PRINT_OUT: print(f'>> source_pre_state : all satisfied !!')
                num_host_satisfied = 0
                for dest_addr in network.address_space:
                    if PRINT_OUT: print(f'>> dest_addr : {dest_addr} / address_space : {network.address_space}')

                    # source / destination이 동일하면 제외
                    if rshost.address == dest_addr:
                        if PRINT_OUT: print(f'>> source_host address와 dest_host address가 동일 - skip!!')
                        continue

                    # kjkoo :
                    #   - (24-0703) 동일한 노드에 있는 IP 주소는 제외 : 다중 IP를 갖는 노드
                    if len(host_literal.ips) > 0:
                        if dest_addr == eval(host_literal.ips[0]):
                            if PRINT_OUT: print(f'>> 동일한 노드에 있는 IP 주소는 제외 : 다중 IP를 갖는 노드 - skip!!')
                            continue
                    rdhost_vec = next_network_state.get_host(dest_addr)
                    rdhost_literal = network.hosts[dest_addr]   # host_literal

                    check_dest_pps_result = self._check_pps_host(network, rdhost_vec, action.dest_pre_state, source=False)
                    if not check_dest_pps_result:
                        if PRINT_OUT: print(f'>> dest pps check failed')
                        continue

                    if PRINT_OUT: print(f'>> dest_pre_state : all satisfied !!')
                    num_host_satisfied += 1     # 모든 조건을 만족한 호스트 개수

                    s_value = self._update_source_post_state(network, rshost, rdhost_vec, action)   # self==target

                    if PRINT_OUT: print(f'>> dest_post_state : will be updated to the dest host {dest_addr} !!')
                    d_value, compromised, host_state_updated = self._update_dest_post_state(network, rshost, rdhost_vec, action)

                    value = s_value + d_value
                    next_state = rdhost_vec     # 호스트 상태 업데이트
                    result = ActionResult(
                        #True,
                        host_state_updated,
                        value=value,
                        compromised=compromised,
                        undefined_error=not(host_state_updated)
                    )
                    return next_state, result  # state, obs

                if num_host_satisfied == 0:
                    # 공격 테크닉의 dest_pre_state 조건을 만족하는 호스트가 하나도 없는 경우, 공격 실패
                    #   - (24-0624) 동일한 action이 반복해서 실패하는 경우, 점진적으로 마이너스 보상을 줄수 있을까?
                    if USE_INC_RWD:
                        inc_rwd = -self._check_seq_action_failed(action)
                    else:
                        inc_rwd = 0
                    next_state = rshost
                    return next_state, ActionResult(False, value=inc_rwd,  #value=0,
                                                    undefined_error=True)

            else:
                ldhost = self.copy()
                # destination host의 pre_state 조건 일치 확인
                if PRINT_OUT: print(f'>> remote=False')

                # action.target(source) 호스트의 상태가 공격테크닉(action)의 dest:pre_state 조건과 일치하는지 확인
                # 1) dest 호스트의 pre_state 모든 조건을 만족해야 공격 테크닉 수행 가능
                check_pps_result = self._check_pps_host(network, ldhost, action.dest_pre_state, source=False)
                if not check_pps_result:
                    if USE_INC_RWD:
                        inc_rwd = -self._check_seq_action_failed(action)
                    else:
                        inc_rwd = 0
                    next_state = ldhost
                    return next_state, ActionResult(False, value=inc_rwd,  #value=0,
                                                    undefined_error=True)

                if PRINT_OUT: print(f'>> dest_post_state : will be updated to the dest host {ldhost.address} !!')
                value, compromised, host_state_updated = self._update_dest_post_state(network, ldhost, ldhost, action)
                next_state = ldhost
                result = ActionResult(
                    host_state_updated,
                    value=value,  # reward에 해당
                    compromised=compromised,
                    undefined_error=not (host_state_updated)
                )
                return next_state, result  # state, obs

        return next_state, ActionResult(False, value=0)

    def observe(self,
                address=False,
                compromised=False,
                reachable=False,
                discovered=False,
                access=False,
                value=False,
                discovery_value=False,
                services=False,
                processes=False,
                os=False):
        obs = np.zeros(self.state_size, dtype=np.float32)
        if address:
            subnet_slice = self._subnet_address_idx_slice()
            host_slice = self._host_address_idx_slice()
            obs[subnet_slice] = self.vector[subnet_slice]
            obs[host_slice] = self.vector[host_slice]
        if compromised:
            obs[self._compromised_idx] = self.vector[self._compromised_idx]
        if reachable:
            obs[self._reachable_idx] = self.vector[self._reachable_idx]
        if discovered:
            obs[self._discovered_idx] = self.vector[self._discovered_idx]
        if value:
            obs[self._value_idx] = self.vector[self._value_idx]
        if discovery_value:
            v = self.vector[self._discovery_value_idx]
            obs[self._discovery_value_idx] = v
        if access:
            obs[self._access_idx] = self.vector[self._access_idx]
        if os:
            idxs = self._os_idx_slice()
            obs[idxs] = self.vector[idxs]
        if services:
            idxs = self._service_idx_slice()
            obs[idxs] = self.vector[idxs]
        if processes:
            idxs = self._process_idx_slice()
            obs[idxs] = self.vector[idxs]
        return obs

    def readable(self):
        return self.get_readable(self.vector)

    def copy(self):
        vector_copy = np.copy(self.vector)
        return HostVector(vector_copy)

    def numpy(self):
        return self.vector

    # kjkoo : TODO: attack_pre_states, attack_post_states 정보를 받아서, idx_map을 만들어야 함.
    # kjkoo : (23-0606)
    #   - 초기 상태 테이블 수정을 위해 service_idx_map, process_idx_map 을 사용하지 않음
    @classmethod
    def _initialize(cls, address_space_bounds, services, os_info, processes, attack_pps):
        cls.os_idx_map = {}
        cls.service_idx_map = {}
        cls.process_idx_map = {}
        cls.address_space_bounds = address_space_bounds
        cls.num_os = len(os_info)
        cls.num_services = len(services)
        cls.num_processes = len(processes)
        # kjkoo : attack_pps_idx_map 추가
        cls.attack_pps_idx_map = {}
        cls.num_attack_pps = len(attack_pps)

        cls._update_vector_idxs()

        for os_num, (os_key, os_val) in enumerate(os_info.items()):
            cls.os_idx_map[os_key] = os_num

        for srv_num, (srv_key, srv_val) in enumerate(services.items()):
            cls.service_idx_map[srv_key] = srv_num
        for proc_num, (proc_key, proc_val) in enumerate(processes.items()):
            cls.process_idx_map[proc_key] = proc_num

        for pps_num, pps_key in enumerate(attack_pps):
            cls.attack_pps_idx_map[pps_key] = pps_num
        print(f"[+] attack_pps_idx_map : {cls.attack_pps_idx_map}")

    @classmethod
    def _get_address_space_bounds(cls, idx):
        return cls.address_space_bounds[idx]

    @classmethod
    def _update_vector_idxs(cls):
        cls._subnet_address_idx = 0

        # kjkoo : (23-1027)
        cls._host_address_idx = cls._get_address_space_bounds(0)
        cls._compromised_idx = (
                cls._host_address_idx + cls._get_address_space_bounds(1)
        )

        cls._reachable_idx = cls._compromised_idx + 1
        cls._discovered_idx = cls._reachable_idx + 1
        cls._value_idx = cls._discovered_idx + 1
        cls._discovery_value_idx = cls._value_idx + 1

        cls._attack_pps_start_idx = cls._discovery_value_idx + 1
        # state_size
        cls.state_size = cls._attack_pps_start_idx + cls.num_attack_pps

    @classmethod
    def _subnet_address_idx_slice(cls):
        return slice(cls._subnet_address_idx, cls._host_address_idx)

    @classmethod
    def _host_address_idx_slice(cls):
        return slice(cls._host_address_idx, cls._compromised_idx)

    # kjkoo : (23-0607) 사용 안할 예정 (service, process 관련)
    @classmethod
    def _get_service_idx(cls, srv_num):
        return cls._service_start_idx+srv_num

    # kjkoo : (23-0607) 사용 안할 예정 (service, process 관련)
    @classmethod
    def _service_idx_slice(cls):
        return slice(cls._service_start_idx, cls._process_start_idx)

    @classmethod
    def _get_os_idx(cls, os_num):
        return cls._os_start_idx+os_num

    # kjkoo : (23-0607)
    #   - attack_pps를 반영해서, os, 다음에 바로 attack_pps가 나옴
    #   - 사용 안할 예정 (23-0612)
    @classmethod
    def _os_idx_slice(cls):
        return slice(cls._os_start_idx, cls._attack_pps_start_idx)
        #return slice(cls._os_start_idx, cls._service_start_idx)

    # kjkoo : (23-0607) 사용 안할 예정 (service, process 관련)
    @classmethod
    def _get_process_idx(cls, proc_num):
        return cls._process_start_idx+proc_num

    # kjkoo : (23-0607) 사용 안할 예정 (service, process 관련)
    @classmethod
    def _process_idx_slice(cls):
        #return slice(cls._process_start_idx, cls.state_size)
        # kjkoo
        return slice(cls._process_start_idx, cls._attack_pre_states_start_idx)


    # kjkoo : attack_pps 추가 (23-0526)
    @classmethod
    def _get_attack_pps_idx(cls, attack_pps_num):
        return cls._attack_pps_start_idx + attack_pps_num

    # kjkoo : (23-0607) idx_slice 추가
    @classmethod
    def _attack_pps_idx_slice(cls):
        return slice(cls._attack_pps_start_idx, cls.state_size)

    @classmethod
    def get_readable(cls, vector):
        readable_dict = dict()
        hvec = cls(vector)
        readable_dict["Address"] = hvec.address
        readable_dict["Compromised"] = bool(hvec.compromised)
        readable_dict["Reachable"] = bool(hvec.reachable)
        readable_dict["Discovered"] = bool(hvec.discovered)
        readable_dict["Value"] = hvec.value
        readable_dict["Discovery Value"] = hvec.discovery_value

        for pps_name in cls.attack_pps_idx_map:
            readable_dict[f"{pps_name}"] = hvec.is_running_pps(pps_name)

        return readable_dict

    @classmethod
    def reset(cls):
        cls.address_space_bounds = None

    def __repr__(self):
        return f"Host: {self.address}"

    def __hash__(self):
        return hash(str(self.vector))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, HostVector):
            return False
        return np.array_equal(self.vector, other.vector)
