import numpy as np

from nasim.envs.attacker.action import ActionResult
from nasim.envs.utils import get_minimal_steps_to_goal, min_subnet_depth, AccessLevel

# kjkoo
PRINT_OUT = False
#PRINT_OUT = True
USE_INC_RWD = False

INTERNET = 0


class Network:
    """A computer network """

    def __init__(self, scenario):
        self.hosts = scenario.hosts
        self.host_num_map = scenario.host_num_map
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.address_space_bounds = scenario.address_space_bounds
        self.sensitive_addresses = scenario.sensitive_addresses
        self.sensitive_hosts = scenario.sensitive_hosts
        # kjkoo
        self.attack_techs = scenario.attack_techs
        self.attack_pps = scenario.attack_pps
        self.attack_pps_dict = scenario.attack_pps_dict
        # kjkoo : (24-0625) 보상함수 : 점진적 음의 보상 (inc_reward)
        self.seq_fail_count = 0
        self.prev_action_name = str()



    def reset(self, state):
        """Reset the network state to initial state """
        next_state = state.copy()
        for host_addr in self.address_space:
            host = next_state.get_host(host_addr)
            host.compromised = False
            #host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable
        return next_state

    def _set_file_access_net(self, host_lit, h_vector):
        if len(host_lit.files) > 0:
            for f_key, f_val in host_lit.files.items():
                if f_val:
                    # file 이름 분리
                    f_key_name = f_key.lower().split('_id')[0]

                    # file_access 설정
                    f_key_name_a = f_key_name + '_a'      # 예: password_a
                    h_vector.attack_pps = [f_key_name_a, self.attack_pps_dict["file_access_dict"]['read']]
        return h_vector

    def _set_file_id_net(self, host_lit, h_vector):
        if len(host_lit.files) > 0:
            for f_key, f_val in host_lit.files.items():
                if f_val:
                    if '_ID' in f_key:
                        # file 이름 분리
                        f_key_name = f_key.lower().split('_id')[0]

                        # file_id 설정
                        f_key_name_i = f_key_name + '_i'  # 예: password_i
                        f_key_name_v = 'id' + f_key.lower().split('_id')[1]  # 예: '-1'

                        h_vector.attack_pps = [f_key_name_i, self.attack_pps_dict["file_id_dict"][f_key_name_v]]
        return h_vector

    def reset_with_pps(self, state):
        """Reset the network state to initial state """
        next_state = state.copy()

        # State를 구성하는 모든 호스트 벡터를 초기화
        for host_addr in self.address_space:
            host_lit = self.hosts[host_addr]        # 공격자 단말 식별용도로 사용
            host = next_state.get_host(host_addr)
            host.reachable = self.subnet_public(host_addr[0])

            host.recv_files = list()

            if host_lit.compromised and host.reachable:
                host.discovered = True
                # 공격자 단말 설정
                for pps_name in self.attack_pps:
                    if pps_name == 'os_type':
                        for os_key, os_val in host_lit.os.items():
                            if os_val == True:
                                host.attack_pps['os_type'] = int(self.attack_pps_dict["os_type_dict"][os_key.lower()])
                    elif pps_name == 'os_status':
                        host.attack_pps['os_status'] = int(self.attack_pps_dict["os_status_dict"]['running'])
                    elif pps_name in self.attack_pps_dict["service_type_list"]:
                        for service_key, service_val in host_lit.services.items():
                            if service_val == True:
                                host.attack_pps = [service_key.lower(), True]
                                host.attack_pps[service_key.lower()+'_s'] = int(
                                    self.attack_pps_dict["service_status_dict"]['running'])
                    elif pps_name in self.attack_pps_dict["service_status_list"]:
                        continue
                    elif pps_name == 'privilege':
                        host.attack_pps = [pps_name, host_lit.access]
                    else:
                        host.attack_pps = [pps_name, int(-1)]
            else:
                # 그 외 단말 초기화
                host.compromised = False
                host.discovered = False
                for pps_name in self.attack_pps:
                    host.attack_pps = [pps_name, int(-1)]

            host = self._set_file_access_net(host_lit, host)
            host = self._set_file_id_net(host_lit, host)

        return next_state

    def _string_split_strip(self, in_str):
        string = in_str.replace(",", "")
        string = string.split()
        out_str = list()
        for word in string:
            word = word.strip()
            if word in ['and']:
                continue
            else:
                out_str.append(word)
        return out_str

    def _find_key_by_value(self, d, target_value):
        for key, value in d.items():
            if value == target_value:
                return key
        return None

    def _str_to_bool(self, s):
        return s.lower() in ('true', '1', 'yes', 'y')

    def _find_value_from_pps_dict(self, shost, dhost, key, val_str):
        value_int = 0   # default
        key_origin = key

        if key in self.attack_pps_dict['host_status_list']:
            return self._str_to_bool(val_str)

        # service type / status / vulnerability
        if key in self.attack_pps_dict['service_type_list']:
            return self._str_to_bool(val_str)
        if key in self.attack_pps_dict['service_status_list']:
            key = 'service_status'  # key 변환 (예: rdp_s -> service_status).
                                    # 왜? rdp_s의 value(예: running)에 대한 index(예: 1)값을 얻기 위해.
        if key in self.attack_pps_dict['service_vuln_list']:    # 예: rdp_v
            key = 'service_vuln'

        # process
        if key in self.attack_pps_dict['process_type_list']:
            return self._str_to_bool(val_str)
        if key in self.attack_pps_dict['process_status_list']:
            key = 'process_status'
        if key in self.attack_pps_dict['process_privilege_list']:
            key = 'process_priv'

        # application type / status / vulnerability
        if key in self.attack_pps_dict['application_type_list']:
            return self._str_to_bool(val_str)
        if key in self.attack_pps_dict['application_status_list']:
            key = 'application_status'
        if key in self.attack_pps_dict['application_vuln_list']:
            key = 'application_vuln'

        # 파일 type / status / access / id
        if key in self.attack_pps_dict['file_type_list']:
            return self._str_to_bool(val_str)
        if key in self.attack_pps_dict['file_status_list']:
            key = 'file_status'  # key 수정
        if key in self.attack_pps_dict['file_access_list']:
            key = 'file_access'
        if key in self.attack_pps_dict['file_id_list']:     # 24-0830 추가
            key = 'file_id'

        key_dict = key + '_dict'
        if val_str == 'source-id':
            value_int = shost.attack_pps[key_origin]
        elif val_str == 'dest-id':
            value_int = dhost.attack_pps[key_origin]
        else:
            value_int = self.attack_pps_dict[key_dict][val_str]

        return value_int

    def _get_true_keys(self, d):
        return [key for key, value in d.items() if value]

    def _check_pps_net(self, host, action_pre_state):
        for pre in action_pre_state:
            if PRINT_OUT: print(f'>> pre_state : {pre}')
            for key, value in pre.items():
                action_pre_state_key = str(key).lower()
                action_pre_state_value_str = str(value).lower()

                action_pre_state_value_list = self._string_split_strip(action_pre_state_value_str)

                if action_pre_state_value_str == 'not admin':
                    # (24-0628)
                    if int(host.attack_pps[action_pre_state_key]) == -1:
                        host_state_value = 0
                    else:
                        host_state_value = host.attack_pps[action_pre_state_key]
                    action_value = self._find_value_from_pps_dict(host, host, action_pre_state_key, 'admin')
                    if int(host_state_value) == int(action_value):
                        if PRINT_OUT: print(
                            f'>> Host Privilege is already "Admin" : {host_state_value} : {float(action_value)}')
                        check_result = False
                        return check_result
                    else:
                        continue  # check success

                action_value = self._find_value_from_pps_dict(host, host, action_pre_state_key, action_pre_state_value_str)

                if action_pre_state_key == 'compromised':
                    host_state_value = host.compromised
                elif action_pre_state_key == 'reachable':
                    host_state_value = host.reachable
                elif action_pre_state_key == 'discovered':
                    host_state_value = host.discovered
                else:
                    if int(host.attack_pps[action_pre_state_key]) == -1:
                        host_state_value = 0
                    else:
                        host_state_value = host.attack_pps[action_pre_state_key]

                if int(host_state_value) & int(action_value):
                    continue
                else:
                    if int(host_state_value) != int(action_value):
                        if PRINT_OUT: print(f'>> pps not equal: {host_state_value} : {float(action_value)}')
                        check_result = False
                        return check_result

        if PRINT_OUT: print(f'>> pre_state : all satisfied !!')
        check_result = True
        return check_result

    def _update_scan_dest_post_state(self, rdhost, rdhost_literal, action):
        newly_discovered = False  # 새로 발견된 호스트
        discovery_reward = 0  # 호스트 발견에 대한 보상
        host_state_updated = False  # 어느 루틴에서든 한번이라도 updated되면 스캐닝 공격 성공으로 간주

        # action의 dest_post_state로 dest host 상태를 업데이트
        for post in action.dest_post_state:
            if PRINT_OUT: print(f'>> ip scan dest_post_state : {post}')
            for key, value in post.items():
                action_post_state_key = str(key).lower()
                action_post_state_value = str(value).lower()

                # discovered 필드 업데이트
                if action_post_state_key == 'discovered':
                    if not rdhost.discovered:
                        newly_discovered = True
                        host_state_updated = True
                    else:
                        return host_state_updated

                # os_type 필드 설정 (int), os_status 필드 설정 (bitmasking int)
                if action_post_state_key == 'os_type':
                    # dest host에 설치된 os 확인하기
                    rdhost_literal_os = rdhost_literal.os
                    if PRINT_OUT: print(f'>> rdhost_literal_os : {rdhost_literal_os}')
                    rdhost_os_type_list = self._get_true_keys(rdhost_literal_os)
                    for os_t in rdhost_os_type_list:
                        current_state_val = rdhost.attack_pps['os_type']
                        host_os_t_val = self.attack_pps_dict['os_type_dict'][str(os_t).lower()]
                        if int(current_state_val) != int(host_os_t_val):   # 상태값과 action값이 다르면 업데이트
                            # os_type 설정
                            rdhost.attack_pps = ['os_type',
                                                    self.attack_pps_dict['os_type_dict'][str(os_t).lower()]]
                            # os_status 설정 (pps를 알고 있어서 설정)
                            rdhost.attack_pps = ['os_status', self.attack_pps_dict['os_status_dict']['running']]
                            host_state_updated = True
                        else:
                            break    # 다음 pps 검사
                service_list = list(rdhost_literal.services.keys())
                lowercased_service_list = [s.lower() for s in service_list]
                if action_post_state_key in lowercased_service_list:
                    if rdhost_literal.services[key]:
                        if PRINT_OUT: print(f'>> service_type : {action_post_state_key} : Running !!!')
                        current_state_val = rdhost.attack_pps[action_post_state_key]
                        if int(current_state_val) == -1:  # 미설정 상태  (24-0628)
                            rdhost.attack_pps = [str(key).lower(), True]
                            rdhost.attack_pps = [str(key).lower() + '_s',
                                                    self.attack_pps_dict['service_status_dict']['running']]
                            host_state_updated = True
                        else:   # 이미 스캔해서 설정된 상태
                            break

                # 조건 외의 pps는 skip
                #   - os_status, [service]_s

        return newly_discovered, host_state_updated

    # kjkoo :
    #   - (24-0625) 연속 실패한 action 여부 확인
    def _check_seq_scan_failed(self, action):
        if action.name == self.prev_action_name:
            self.seq_fail_count += 1
        else:
            self.seq_fail_count = 0
            self.prev_action_name = action.name
        return self.seq_fail_count

    def _perform_subnet_scan_with_pps_v2(self, next_state, action):
        # kjkoo : (23-0808)
        USE_STD_VALUE = True

        # pre-state 조건 matching
        pps_list = self.attack_pps

        if action.is_attack_tech():
            pre_cond_satisfied = True
            source_host_cond_satisfied = False
            check_recv_file_none = False
            check_dumped_none = False
            check_recv_file = False
            check_dumped = False

            # [1] 원격 (remote=True) 공격인 경우, source / dest 호스트의 pre_state가 존재
            if PRINT_OUT: print(f'>> remote 공격 : subnet scan ')
            if action.remote:
                rshost = next_state.get_host(action.target)     # source 호스트
                # source 호스트의 pre_state를 체크
                check_pps_result = self._check_pps_net(rshost, action.source_pre_state) # dest=rshost 더비 입력
                if not check_pps_result:
                    return next_state, ActionResult(False, undefined_error=True)
                if PRINT_OUT: print(f'>> source_pre_state : all satisfied !!')

                discovered = {}
                newly_discovered = {}   # 새로 발견된 호스트
                discovery_reward = 0    # 호스트 발견에 대한 보상
                action_success = False  # 공격 성공했는지? 확인
                num_host_satisfied = 0

                for dest_addr in self.address_space:
                    # dest_addr 호스트의 서브넷과 action.target(source) 호스트의 서브넷이 연결되어 있지 않으면, continue
                    if PRINT_OUT: print(f'>> dest_addr : {dest_addr} / address_space : {self.address_space}')

                    if not self.subnets_same(dest_addr[0], action.target[0]):
                        if PRINT_OUT: print(f'>> subnet check continue : dest{dest_addr}, act.target{action.target}')
                        continue
                    else:
                        if PRINT_OUT: print(f'>> subnet check  : dest{dest_addr}, act.target{action.target}')

                    dest_host_pre_state_cond_satisfied = True

                    rdhost = next_state.get_host(dest_addr)
                    rdhost_literal = self.hosts[dest_addr]

                    check_pps_result = self._check_pps_net(rdhost, action.dest_pre_state)
                    if not check_pps_result:
                        continue        # 다음 호스트로 넘어감 (pps 불일치)
                    else:               # 현재 호스트의 post_state value 설정

                        if PRINT_OUT: print(f'>> dest_pre_state : all satisfied !!')

                        discover_result, host_state_updated = (
                            self._update_scan_dest_post_state(rdhost, rdhost_literal, action))

                        # 새로 발견된 호스트면, discovered 필드 설정
                        if discover_result:
                            newly_discovered[dest_addr] = True          # 불필요한 것 같은데. (24-0617)
                            discovered[dest_addr] = True
                            next_state.set_host_discovered(dest_addr)  # discovered 필드 설정

                        if host_state_updated:
                            if len(rdhost_literal.ips) > 0:
                                for ip in rdhost_literal.ips:  # type(ip):str
                                    next_state.update_host_with_mips(eval(ip), rdhost)

                            num_host_satisfied += 1

                            if USE_STD_VALUE:
                                discovery_reward += 3  # cost=1 빼는것 고려해서    (24-0626)
                            else:
                                discovery_reward += rdhost.discovery_value       # d

                            # 적어도 하나의 호스트에 대해서는 공격 성공했음
                            action_success = True

                if action_success:
                    if num_host_satisfied == 0 :
                        discovery_reward = 0
                    obs = ActionResult(
                        True,
                        value=discovery_reward,
                        discovered=discovered,
                        newly_discovered=newly_discovered
                    )
                    return next_state, obs
            else:
                print(f'>> 오류! local 공격 : 서브넷 스캔은 remote 공격임.')
                return next_state, ActionResult(False, value=0, undefined_error=True)
        #print(f'>> 오류! action이 attack_tech이 아님!')
        return next_state, ActionResult(False, value=0, undefined_error=True)

    def perform_action_with_pps_v2(self, state, action):

        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        next_state = state.copy()

        if action.is_noop():
            return next_state, ActionResult(True)

        if not state.host_reachable(action.target) \
                or not state.host_discovered(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if action.service != None \
            and not self.traffic_permitted(
            state, action.target, action.service
        ):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        host_compromised = state.host_compromised(action.target)

        if action.service != None and host_compromised:
            pass
        elif np.random.rand() > action.prob:
            return next_state, ActionResult(False, 0.0, undefined_error=True)

        ATECH_SCAN_SUBNET = 'T1595.001'

        if action.name.split('@')[0] == ATECH_SCAN_SUBNET:
            next_state, action_obs = self._perform_subnet_scan_with_pps_v2(next_state, action)

            return next_state, action_obs

        t_host = state.get_host(action.target)
        t_host_literal = self.hosts[action.target]

        dest_host, action_obs = t_host.perform_action_with_pps_v2(self, next_state, action, t_host_literal)

        if action_obs.success:
            if action.remote:
                if len(action.source_post_state) > 0:
                    next_state.update_host(t_host.address, t_host)      # source host 상태 업데이트
            dest_host_literal = self.hosts[dest_host.address]
            next_state.update_host(dest_host.address, dest_host)  # dest host 상태 업데이트 (7-1)

            # 멀티 ips
            if len(dest_host_literal.ips) > 0:
                for ip in dest_host_literal.ips:  # type(ip):str
                    next_state.update_host_with_mips(eval(ip), dest_host)   # (7-2)

        self._update_with_pps(next_state, action, action_obs)

        return next_state, action_obs

    def _update_with_pps(self, state, action, action_obs):
        if action.is_attack_tech() and action_obs.success \
                and action_obs.compromised:
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        """Updates the reachable status of hosts on network, based on current
        state and newly exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    # kjkoo
    #   - (24-0816) : 서브넷 연결성 검토 : action.target과 동일한 서브넷으로 스캔 제한
    def subnets_same(self, subnet_1, subnet_2):
        return subnet_1 == subnet_2


    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    # kjkoo : action.target == dest_addr
    def host_traffic_permitted(self, src_addr, dest_addr, service):
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    # kjkoo : action.target == host_addr
    def traffic_permitted(self, state, host_addr, service):
        """Checks whether the subnet and host firewalls permits traffic to a
        given host and service, based on current set of compromised hosts on
        network.
        """
        for src_addr in self.address_space:
            if not state.host_compromised(src_addr) and \
               not self.subnet_public(src_addr[0]):
                continue
            if not self.subnet_traffic_permitted(
                    src_addr[0], host_addr[0], service
            ):
                continue
            if self.host_traffic_permitted(src_addr, host_addr, service):
                return True
        return False

    def subnet_public(self, subnet):
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        return len(self.subnets)

    # kjkoo : (23-0622)
    #   - 알파인랩의 공격시나리오에서는 모든 호스트가 root 권한을 가지지 않아도,
    #     공격 목표를 달성할 수 있다.
    #     ; 일단, 모든 sensitive 호스트를 compromised하면 episode를 끝내는 것으로 한다.
    #     ; 또한, compromised 여부는 컬럼값 확인하는 것으로 함. access 권한 아님
    def all_sensitive_hosts_compromised(self, state):
        for host_addr in self.sensitive_addresses:
            # kjkoo : (23-0622)
            if not state.host_compromised(host_addr):
            #if not state.host_has_access(host_addr, AccessLevel.ROOT):
                return False
        return True

    def get_total_sensitive_host_value(self):
        total = 0
        for host_value in self.sensitive_hosts.values():
            total += host_value
        return total

    def get_total_discovery_value(self):
        total = 0
        for host in self.hosts.values():
            total += host.discovery_value
        return total

    def get_minimal_steps(self):
        return get_minimal_steps_to_goal(
            self.topology, self.sensitive_addresses
        )

    def get_subnet_depths(self):
        return min_subnet_depth(self.topology)

    def __str__(self):
        output = "\n--- Network ---\n"
        output += "Subnets: " + str(self.subnets) + "\n"
        output += "Topology:\n"
        for row in self.topology:
            output += f"\t{row}\n"
        output += "Sensitive hosts: \n"
        for addr, value in self.sensitive_hosts.items():
            output += f"\t{addr}: {value}\n"
        output += "Num_services: {self.scenario.num_services}\n"
        output += "Hosts:\n"
        for m in self.hosts.values():
            output += str(m) + "\n"
        output += "Firewall:\n"
        for c, a in self.firewall.items():
            output += f"\t{c}: {a}\n"
        return output
