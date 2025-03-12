from nasim.envs.defender.action  import DefenderActionResult
from nasim.envs.utils import get_minimal_steps_to_goal, min_subnet_depth, AccessLevel
import numpy as np
# column in topology adjacency matrix that represents connection between
# subnet and public

PRINT_OUT = False
#PRINT_OUT = True
USE_INC_RWD = True

INTERNET = 0


class Network:

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

        self.defend_techs = scenario.defend_techs
        self.defend_pps = scenario.defend_pps
        self.defend_pps_dict = scenario.defend_pps_dict

        self.seq_fail_count = 0
        self.prev_action_name = str()

    def _set_file_access_net(self, host_lit, h_vector):
        if len(host_lit.files) > 0:
            for f_key, f_val in host_lit.files.items():
                if f_val:
                    f_key_name = f_key.lower().split('_id')[0]
                    f_key_name_a = f_key_name + '_a'      # 예: password_a
                    h_vector.defend_pps = [f_key_name_a, self.defend_pps_dict["file_access_dict"]['read']]
        return h_vector

    def reset_with_pps(self, state):
        next_state = state.copy()

        for host_addr in self.address_space:
            host_lit = self.hosts[host_addr]
            host = next_state.get_D_host(host_addr)
            host.reachable = self.subnet_public(host_addr[0])

            host.recv_files = list()

            if host_lit.compromised and host.reachable:
                host.discovered = True
                # 공격자 단말 설정
                for pps_name in self.defend_pps:
                    if pps_name == 'os_type':
                        for os_key, os_val in host_lit.os.items():
                            if os_val == True:
                                host.defend_pps['os_type'] = int(self.defend_pps_dict["os_type_dict"][os_key.lower()])
                    elif pps_name == 'os_status':
                        host.defend_pps['os_status'] = int(self.defend_pps_dict["os_status_dict"]['running'])
                    elif pps_name in self.defend_pps_dict["service_type_list"]:
                        for service_key, service_val in host_lit.services.items():
                            if service_val == True:
                                host.defend_pps = [service_key.lower(), True]
                                host.defend_pps[service_key.lower() + '_s'] = int(
                                    self.defend_pps_dict["service_status_dict"]['running'])
                    elif pps_name in self.defend_pps_dict["service_status_list"]:
                        continue
                    elif pps_name == 'privilege':
                        host.defend_pps = [pps_name, host_lit.access]
                    else:
                        host.defend_pps = [pps_name, int(-1)]  # 24-0628
            else:
                # 그 외 단말 초기화
                host.compromised = False
                host.discovered = False
                for pps_name in self.defend_pps:
                    host.defend_pps = [pps_name, int(-1)]
            host = self._set_file_access_net(host_lit, host)
            #host = self._set_file_id_net(host_lit, host)

        return next_state

    def perform_defender_action(self, state, action):
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        next_state = state.copy()

        if action.is_noop():
            return next_state, DefenderActionResult(False)

        if not state.host_d_reachable(action.target) \
                or not state.host_d_discovered(action.target):
            result = DefenderActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if action.service != None \
            and not self.traffic_permitted(
            state, action.target, action.service
        ):

            result = DefenderActionResult(False, 0.0, connection_error=True)
            return next_state, result

        host_compromised = state.host_d_compromised(action.target)

        if action.service != None and host_compromised:
            # host already compromised so exploits don't fail due to randomness
            pass

        t_host = state.get_D_host(action.target)
        t_host_literal = self.hosts[action.target]
        dest_host, action_obs = t_host.perform_action_with_pps(self, next_state, action, t_host_literal)

        if action_obs.success:
            if len(action.source_post_state) > 0:
                next_state.update_host(t_host.address, t_host)  # source host 상태 업데이트
            dest_host_literal = self.hosts[dest_host.address]
            next_state.update_host(dest_host.address, dest_host)  # dest host 상태 업데이트 (7-1)

            # 멀티 ips
            if len(dest_host_literal.ips) > 0:
                for ip in dest_host_literal.ips:  # type(ip):str
                    next_state.update_host_with_D_mips(eval(ip), dest_host)  # (7-2)

        # self._update_with_pps(next_state, action, action_obs)

        return next_state, action_obs

    def _string_split_strip(self, in_str):
        string = in_str.replace(",", "")
        string = string.split()
        out_str = list()
        for word in string:
            word = word.strip()
            if word in ['and']:  # kjkoo (24-0820) : or는 복수 비교를 위해 제외, 'not'
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
        value_int = 0  # default
        key_origin = key

        if key in self.defend_pps_dict['host_status_list']:
            return self._str_to_bool(val_str)

        # service type / status / vulnerability
        if key in self.defend_pps_dict['service_type_list']:
            return self._str_to_bool(val_str)
        if key in self.defend_pps_dict['service_status_list']:
            key = 'service_status'
        if key in self.defend_pps_dict['service_vuln_list']:
            key = 'service_vuln'

        # process
        if key in self.defend_pps_dict['process_type_list']:
            return self._str_to_bool(val_str)
        if key in self.defend_pps_dict['process_status_list']:
            key = 'process_status'
        if key in self.defend_pps_dict['process_privilege_list']:
            key = 'process_priv'

        # application type / status / vulnerability
        if key in self.defend_pps_dict['application_type_list']:
            return self._str_to_bool(val_str)
        if key in self.defend_pps_dict['application_status_list']:
            key = 'application_status'
        if key in self.defend_pps_dict['application_vuln_list']:
            key = 'application_vuln'

        if key in self.defend_pps_dict['file_type_list']:
            return self._str_to_bool(val_str)
        if key in self.defend_pps_dict['file_status_list']:
            key = 'file_status'  # key 수정
        if key in self.defend_pps_dict['file_access_list']:
            key = 'file_access'
        if key in self.defend_pps_dict['file_id_list']:
            key = 'file_id'

        key_dict = key + '_dict'

        if val_str == 'source-id':
            value_int = shost.defend_pps[key_origin]
        elif val_str == 'dest-id':
            value_int = dhost.defend_pps[key_origin]
        else:
            if 'not ' in val_str:
                real_val_str = val_str.split('not ')
                value_int = self.defend_pps_dict[key_dict][real_val_str[1]]
            else :
                value_int = self.defend_pps_dict[key_dict][val_str]
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
                    if int(host.defend_pps[action_pre_state_key]) == -1:
                        host_state_value = 0
                    else:
                        host_state_value = host.defend_pps[action_pre_state_key]

                    action_value = self._find_value_from_pps_dict(host, host, action_pre_state_key, 'admin')
                    if int(host_state_value) == int(action_value):
                        if PRINT_OUT: print(
                            f'>> Host Privilege is already "Admin" : {host_state_value} : {float(action_value)}')
                        check_result = False
                        return check_result
                    else:
                        continue  # check success

                action_value = self._find_value_from_pps_dict(host, host, action_pre_state_key, action_pre_state_value_str)

                # 호스트 상태 값 가져오기
                if action_pre_state_key == 'compromised':
                    host_state_value = host.compromised
                elif action_pre_state_key == 'reachable':
                    host_state_value = host.reachable
                elif action_pre_state_key == 'discovered':
                    host_state_value = host.discovered
                else:
                    # (24-0628)
                    if int(host.defend_pps[action_pre_state_key]) == -1:
                        host_state_value = 0
                    else:
                        host_state_value = host.defend_pps[action_pre_state_key]

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

        for post in action.dest_post_state:
            if PRINT_OUT: print(f'>> ip scan dest_post_state : {post}')
            for key, value in post.items():
                action_post_state_key = str(key).lower()
                action_post_state_value = str(value).lower()

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
                        current_state_val = rdhost.defend_pps['os_type']
                        host_os_t_val = self.defend_pps_dict['os_type_dict'][str(os_t).lower()]
                        if int(current_state_val) != int(host_os_t_val):  # 상태값과 action값이 다르면 업데이트
                            # os_type 설정
                            rdhost.defend_pps = ['os_type',
                                                 self.defend_pps_dict['os_type_dict'][str(os_t).lower()]]
                            # os_status 설정 (pps를 알고 있어서 설정)
                            rdhost.defend_pps = ['os_status', self.defend_pps_dict['os_status_dict']['running']]
                            host_state_updated = True
                        else:
                            break  # 다음 pps 검사

                service_list = list(rdhost_literal.services.keys())
                lowercased_service_list = [s.lower() for s in service_list]
                if action_post_state_key in lowercased_service_list:
                    if rdhost_literal.services[key]:
                        if PRINT_OUT: print(f'>> service_type : {action_post_state_key} : Running !!!')
                        current_state_val = rdhost.defend_pps[action_post_state_key]
                        # if int(current_state_val) == 0:     # 미설정 상태
                        if int(current_state_val) == -1:  # 미설정 상태  (24-0628)
                            rdhost.defend_pps = [str(key).lower(), True]
                            rdhost.defend_pps = [str(key).lower() + '_s',
                                                 self.defend_pps_dict['service_status_dict']['running']]
                            host_state_updated = True
                        else:  # 이미 스캔해서 설정된 상태
                            break

        return newly_discovered, host_state_updated

    def _update_with_pps(self, state, action, action_obs):
        if action.is_defend_tech() and action_obs.success \
                and not(action_obs.compromised):
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_d_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    def host_traffic_permitted(self, src_addr, dest_addr, service):
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    def has_required_remote_permission(self, state, action):
        if self.subnet_public(action.target[0]):
            return True

        for src_addr in self.address_space:
            if not state.host_d_compromised(src_addr):
                continue
            if action.is_scan() and \
               not self.subnets_connected(src_addr[0], action.target[0]):
                continue
            if action.is_exploit() and \
               not self.subnet_traffic_permitted(
                   src_addr[0], action.target[0], action.service
               ):
                continue
            if state.host_has_access(src_addr, action.req_access):
                return True
        return False

    def traffic_permitted(self, state, host_addr, service):
        for src_addr in self.address_space:
            if not state.host_d_compromised(src_addr) and \
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

    def all_sensitive_hosts_compromised(self, state):
        for host_addr in self.sensitive_addresses:
            if state.host_compromised(host_addr):
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
