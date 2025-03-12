import numpy as np

from nasim.envs.attacker.host_vector import HostVector
from nasim.envs.defender.host_vector import HostVector as DHostVector
from nasim.envs.attacker.observation import Observation
from nasim.envs.defender.observation import Observation as DObservation


class State:
    def __init__(self, network_tensor, host_num_map):
        self.tensor = network_tensor
        self.host_num_map = host_num_map

    @classmethod
    def tensorize(cls, network):
        h0 = network.hosts[(1, 0)]  # network.hosts : dict (1,0) key
        # kjkoo : add attack techniques
        h0_vector = HostVector.vectorize(h0, network.address_space_bounds,
                                         network.attack_pps, network.attack_pps_dict)
        # h0_vector = HostVector.vectorize(h0, network.address_space_bounds)
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),
            dtype=np.float32
        )

        for host_addr, host in network.hosts.items():
            host_num = network.host_num_map[host_addr]
            HostVector.vectorize(
                # kjkoo
                host, network.address_space_bounds, network.attack_pps,
                network.attack_pps_dict, tensor[host_num]
            )
        # print(f"[+] tensor :\n{tensor}")
        return cls(tensor, network.host_num_map)

    @classmethod
    def D_tensorize(cls, network):
        h0 = network.hosts[(1, 0)]  # network.hosts : dict (1,0) key
        # kjkoo : add attack techniques
        h0_vector = DHostVector.vectorize(h0, network.address_space_bounds,
                                          network.defend_pps, network.defend_pps_dict)
        # h0_vector = HostVector.vectorize(h0, network.address_space_bounds)
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),
            dtype=np.float32
        )
        for host_addr, host in network.hosts.items():
            host_num = network.host_num_map[host_addr]
            DHostVector.vectorize(
                host, network.address_space_bounds, network.defend_pps,
                network.defend_pps_dict, tensor[host_num]
            )
        # print(f"[+] tensor :\n{tensor}")
        return cls(tensor, network.host_num_map)

    # kjkoo : (23-0605)
    #   - network.reset -> _with_pps 함수로 변경
    #       ; 공격자 단말 설정, attack_pps 상태 설정등 변경 필요해서 바꿈
    @classmethod
    def generate_initial_state(cls, network):
        cls.reset()
        state = cls.tensorize(network)
        # kjkoo
        # return network.reset(state)
        return network.reset_with_pps(state)

    @classmethod
    def D_generate_initial_state(cls, network):
        cls.reset()
        state = cls.D_tensorize(network)
        # kjkoo
        # return network.reset(state)
        return network.reset_with_pps(state)

    @classmethod
    def sync_dicts_by_value(cls, a_net, d_net, mode:str):
        if mode == "defender":
            for host_addr, host in a_net.hosts:
                a_host_obs = a_net.get_host(host_addr)
                b_host_obs = d_net.get_D_host(host_addr)

                common_keys = set(a_host_obs.attack_pps.keys()) & set(b_host_obs.defend_pps.keys())

                b_host_obs.compromised = a_host_obs.compromised
                b_host_obs.reachable = a_host_obs.reachable
                b_host_obs.discovered = a_host_obs.discovered

                for key in common_keys:
                    value_to_sync = a_host_obs.attack_pps[key]
                    b_host_obs.defend_pps = [key, value_to_sync]

        elif mode == "attacker":
            for host_addr, host in a_net.hosts:
                a_host_obs = a_net.get_D_host(host_addr)
                b_host_obs = d_net.get_host(host_addr)

                common_keys = set(a_host_obs.defend_pps.keys()) & set(b_host_obs.attack_pps.keys())

                b_host_obs.compromised = a_host_obs.compromised
                # b_host_obs.reachable = a_host_obs.reachable
                # b_host_obs.discovered = a_host_obs.discovered

                for key in common_keys:
                    value_to_sync = a_host_obs.defend_pps[key]
                    b_host_obs.attack_pps = [key, value_to_sync]

    @classmethod
    def from_numpy(cls, s_array, state_shape, host_num_map):
        if s_array.shape != state_shape:
            s_array = s_array.reshape(state_shape)
        return State(s_array, host_num_map)

    @classmethod
    def reset(cls):
        """Reset any class attributes for state """
        HostVector.reset()
        DHostVector.reset()

    @property
    def hosts(self):
        hosts = []
        for host_addr in self.host_num_map:
            hosts.append((host_addr, self.get_host(host_addr)))
        return hosts

    @property
    def D_hosts(self):
        hosts = []
        for host_addr in self.host_num_map:
            hosts.append((host_addr, self.get_D_host(host_addr)))
        return hosts

    def copy(self):
        new_tensor = np.copy(self.tensor)
        return State(new_tensor, self.host_num_map)

    def get_initial_observation(self, fully_obs):
        """Get the initial observation of network.

        Returns
        -------
        Observation
            an observation object
        """
        # kjkoo : self.shape() = tensor.shape : aux-row 빼고, host_vector로만 구성
        #   - aux-row는 마지막 row
        obs = Observation(self.shape())
        if fully_obs:
            obs.from_state(self)
            return obs

        for host_addr, host in self.hosts:
            if not host.reachable:
                continue
            host_obs = host.observe(address=True,
                                    reachable=True,
                                    discovered=True)
            host_idx = self.get_host_idx(host_addr)
            obs.update_from_host(host_idx, host_obs)
        return obs

    def get_observation(self, action, action_result, fully_obs):
        obs = Observation(self.shape())
        obs.from_action_result(action_result)
        if fully_obs:
            obs.from_state(self)
            return obs

        if action.is_noop():
            return obs

        if not action_result.success:
            # action failed so no observation
            return obs

        t_idx, t_host = self.get_host_and_idx(action.target)
        obs_kwargs = dict(
            address=True,       # must be true for success
            compromised=False,
            reachable=True,     # must be true for success
            discovered=True,    # must be true for success
            value=False,
            # discovery_value=False,    # this is only added as needed
            services=False,
            processes=False,
            os=False,
            access=False
        )
        if action.is_exploit():
            # exploit action, so get all observations for host
            obs_kwargs["compromised"] = True
            obs_kwargs["services"] = True
            obs_kwargs["os"] = True
            obs_kwargs["access"] = True
            obs_kwargs["value"] = True
        elif action.is_privilege_escalation():
            obs_kwargs["compromised"] = True
            obs_kwargs["access"] = True
        elif action.is_service_scan():
            obs_kwargs["services"] = True
        elif action.is_os_scan():
            obs_kwargs["os"] = True
        elif action.is_process_scan():
            obs_kwargs["processes"] = True
            obs_kwargs["access"] = True
        elif action.is_subnet_scan():
            for host_addr in action_result.discovered:
                discovered = action_result.discovered[host_addr]
                if not discovered:
                    continue
                d_idx, d_host = self.get_host_and_idx(host_addr)
                newly_discovered = action_result.newly_discovered[host_addr]
                d_obs = d_host.observe(
                    discovery_value=newly_discovered, **obs_kwargs
                )
                obs.update_from_host(d_idx, d_obs)
            # this is for target host (where scan was performed on)
            obs_kwargs["compromised"] = True
        else:
            raise NotImplementedError(f"Action {action} not implemented")

        target_obs = t_host.observe(**obs_kwargs)
        obs.update_from_host(t_idx, target_obs)
        return obs

    def get_D_initial_observation(self, fully_obs):
        # kjkoo : self.shape() = tensor.shape : aux-row 빼고, host_vector로만 구성
        #   - aux-row는 마지막 row
        obs = DObservation(self.shape())
        if fully_obs:
            obs.from_state(self)
            return obs

        for host_addr, host in self.D_hosts:
            if not host.reachable:
                continue
            host_obs = host.defender_observe(address=True,
                                             reachable=True,
                                             discovered=True)
            host_idx = self.get_D_host_and_idx(host_addr)
            obs.update_from_host(host_idx, host_obs)
        return obs

    def get_observation_defender(self, action_defender, action_result, fully_obs):
        obs = DObservation(self.shape())
        obs.from_action_result(action_result)
        if fully_obs:
            obs.from_state(self)
            return obs

        if action_defender.is_noop():
           return obs

        if not action_result.success:
            # action failed so no observation
            return obs

        t_idx, t_host = self.get_D_host_and_idx(action_defender.target)
        obs_kwargs = dict(
            address=True,       # must be true for success
            compromised=False,
            reachable=True,     # must be true for success
            discovered=True,    # must be true for success
            value=False,
            # discovery_value=False,    # this is only added as needed
            services=False,
            processes=False,
            os=False,
            access=False,
        )

        if action_defender.is_change_os():
            obs_kwargs["os"] = True
        elif action_defender.is_change_firewall():
            obs_kwargs["firewall"] = True
        elif action_defender.is_stop_service():
            obs_kwargs["services"] = True
        elif action_defender.is_stop_process():
            obs_kwargs["processes"] = True

        else:
            raise NotImplementedError(f"Defender Action {action_defender} not implemented")

        target_obs = t_host.defender_observe(**obs_kwargs)
        print(target_obs)
        obs.update_from_host(t_idx, target_obs)
        return obs

    def shape_flat(self):
        return self.numpy_flat().shape

    def shape(self):
        return self.tensor.shape

    def numpy_flat(self):
        return self.tensor.flatten()

    def numpy(self):
        return self.tensor

    def update_host(self, host_addr, host_vector):
        host_idx = self.host_num_map[host_addr]
        self.tensor[host_idx] = host_vector.vector

    # kjkoo : (23-0617)
    #   - 호스트가 다수 IP 주소를 가지고 있을 때, 상태테이블 업데이트
    def update_host_with_mips(self, multi_host_addr, target_host_vector):
        # target_host_idx = self.host_num_map[target_host_addr]
        multi_host_idx = self.host_num_map[multi_host_addr]

        multi_host_vector = self.get_host(multi_host_addr)
        # 값 할당
        multi_host_vector.compromised = target_host_vector.compromised
        multi_host_vector.reachable = target_host_vector.reachable
        multi_host_vector.discovered = target_host_vector.discovered

        # for pps, pps_num in self.attack_pps_idx_map.items():
        for pps_key, pps_val in target_host_vector.attack_pps.items():
            multi_host_vector.attack_pps = [pps_key, pps_val]

        # 다중 IP에 대한 상태테이블 업데이트
        self.tensor[multi_host_idx] = multi_host_vector.vector

    def update_host_with_D_mips(self, multi_host_addr, target_host_vector):
        # target_host_idx = self.host_num_map[target_host_addr]
        multi_host_idx = self.host_num_map[multi_host_addr]

        multi_host_vector = self.get_D_host(multi_host_addr)
        # 값 할당
        multi_host_vector.compromised = target_host_vector.compromised
        multi_host_vector.reachable = target_host_vector.reachable
        multi_host_vector.discovered = target_host_vector.discovered

        # for pps, pps_num in self.attack_pps_idx_map.items():
        for pps_key, pps_val in target_host_vector.defend_pps.items():
            multi_host_vector.defend_pps = [pps_key, pps_val]

        # 다중 IP에 대한 상태테이블 업데이트
        self.tensor[multi_host_idx] = multi_host_vector.vector

    def get_host(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return HostVector(self.tensor[host_idx])

    def get_D_host(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return DHostVector(self.tensor[host_idx])

    def get_host_idx(self, host_addr):
        return self.host_num_map[host_addr]

    def get_host_and_idx(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return host_idx, HostVector(self.tensor[host_idx])

    def get_D_host_and_idx(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return host_idx, DHostVector(self.tensor[host_idx])

    def host_reachable(self, host_addr):
        return self.get_host(host_addr).reachable

    def host_d_reachable(self, host_addr):
        return self.get_D_host(host_addr).reachable

    def host_compromised(self, host_addr):
        return self.get_host(host_addr).compromised

    def host_d_compromised(self, host_addr):
        return self.get_D_host(host_addr).compromised

    def host_discovered(self, host_addr):
        return self.get_host(host_addr).discovered

    def host_d_discovered(self, host_addr):
        return self.get_D_host(host_addr).discovered

    def host_has_access(self, host_addr, access_level):
        return self.get_host(host_addr).access >= access_level

    def set_host_compromised(self, host_addr):
        self.get_host(host_addr).compromised = True

    def set_host_reachable(self, host_addr):
        self.get_host(host_addr).reachable = True

    def set_host_discovered(self, host_addr):
        self.get_host(host_addr).discovered = True

    def state_size(self):
        return self.tensor.size

    def get_readable(self):
        host_obs = []
        for host_addr in self.host_num_map:
            host = self.get_host(host_addr)
            readable_dict = host.readable()
            host_obs.append(readable_dict)
        return host_obs

    def get_d_readable(self):
        host_obs = []
        for host_addr in self.host_num_map:
            host = self.get_D_host(host_addr)
            readable_dict = host.readable()
            host_obs.append(readable_dict)
        return host_obs

    def __str__(self):
        output = "\n--- State ---\n"
        output += "Hosts:\n"
        for host in self.hosts:
            output += str(host) + "\n"
        return output

    def __hash__(self):
        return hash(str(self.tensor))

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)