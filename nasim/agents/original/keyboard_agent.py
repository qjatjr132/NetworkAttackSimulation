"""An agent that lets the user interact with NASim using the keyboard.

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python keyboard_agent.py tiny

This will run the agent and display the game in stdout.

To see available running arguments:

$ python keyboard_agent.py--help
"""
import nasim as nasim
from nasim.envs.attacker.action import Exploit, PrivilegeEscalation


LINE_BREAK = "-"*60
LINE_BREAK2 = "="*60


def print_actions(action_space):
    for a in range(action_space.n):
        print(f"{a} {action_space.get_action(a)}")
    print(LINE_BREAK)


def choose_flat_action(env):
    print_actions(env.action_space)
    while True:
        try:
            idx = int(input("Choose action number: "))
            action = env.action_space.get_action(idx)
            print(f"Performing: {action}")
            return action
        # kjkoo : 23-1027
        #except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")

def choose_d_flat_action(env):
    print_actions(env.defender_action_space)
    while True:
        try:
            idx = int(input("Choose action number: "))
            action = env.defender_action_space.get_action(idx)
            print(f"Performing: {action}")
            return action
        # kjkoo : 23-1027
        #except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")

def choose_flat_action_semi_automated(env):
    print_actions(env.action_space)
    while True:
        try:
            idx = int(input("Choose action number: "))
            return idx
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")


# kjkoo : (23-0621)
#   - 공격 시퀀스를 자동으로 수행할 수 있도록 사용자 입력 조정
def choose_flat_action_automated(env, action_idx, num_attack):
    if num_attack == 0:
        print_actions(env.action_space)
    while True:
        try:
            #idx = int(input("Choose action number: "))
            idx = action_idx
            action = env.action_space.get_action(idx)
            print(f"Performing: [{idx}] {action}")
            return action
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")


def display_actions(actions):
    action_names = list(actions)
    for i, name in enumerate(action_names):
        a_def = actions[name]
        output = [f"{i} {name}:"]
        output.extend([f"{k}={v}" for k, v in a_def.items()])
        print(" ".join(output))


def choose_item(items):
    while True:
        try:
            idx = int(input("Choose number: "))
            return items[idx]
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")


def choose_param_action(env):
    print("1. Choose Action Type:")
    print("----------------------")
    for i, atype in enumerate(env.action_space.action_types):
        print(f"{i} {atype.__name__}")
    while True:
        try:
            atype_idx = int(input("Choose index: "))
            # check idx valid
            atype = env.action_space.action_types[atype_idx]
            break
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")

    print("------------------------")
    print("2. Choose Target Subnet:")
    print("------------------------")
    num_subnets = env.action_space.nvec[1]
    while True:
        try:
            subnet = int(input(f"Choose subnet in [1, {num_subnets}]: "))
            if subnet < 1 or subnet > num_subnets:
                raise ValueError()
            break
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")

    print("----------------------")
    print("3. Choose Target Host:")
    print("----------------------")
    num_hosts = env.scenario.subnets[subnet]
    while True:
        try:
            host = int(input(f"Choose host in [0, {num_hosts-1}]: "))
            if host < 0 or host > num_hosts-1:
                raise ValueError()
            break
        # kjkoo : 23-1027
        # except Exception:
        except ValueError:
            print("Invalid choice. Try again.")
        except IndexError:
            print("Invalid choice. Try again.")

    # subnet-1, since action_space handles exclusion of internet subnet
    avec = [atype_idx, subnet-1, host, 0, 0]
    if atype not in (Exploit, PrivilegeEscalation):
        action = env.action_space.get_action(avec)
        print("----------------")
        print(f"ACTION SELECTED: {action}")
        return action

    target = (subnet, host)
    if atype == Exploit:
        print("------------------")
        print("4. Choose Exploit:")
        print("------------------")
        exploits = env.scenario.exploits
        display_actions(exploits)
        e_name = choose_item(list(exploits))
        action = Exploit(name=e_name, target=target, **exploits[e_name])
    else:
        print("------------------")
        print("4. Choose Privilege Escalation:")
        print("------------------")
        privescs = env.scenario.privescs
        display_actions(privescs)
        pe_name = choose_item(list(privescs))
        action = PrivilegeEscalation(
            name=pe_name, target=target, **privescs[pe_name]
        )

    print("----------------")
    print(f"ACTION SELECTED: {action}")
    return action


def choose_action(env):
    input("Press enter to choose next action..")
    print("\n" + LINE_BREAK2)
    print("Attacker_CHOOSE ACTION")
    print(LINE_BREAK2)
    if env.flat_actions:
        return choose_flat_action(env)
    return choose_param_action(env)

def choose_d_action(env):
    input("Press enter to choose next action..")
    print("\n" + LINE_BREAK2)
    print("Defender_CHOOSE ACTION")
    print(LINE_BREAK2)
    if env.flat_actions:
        return choose_d_flat_action(env)
    return choose_param_action(env)

# kjkoo: (23-0621)
#   -
def choose_action_automated(env, action_idx, num_attack):
    print("Press enter to choose next action..")
    print("\n" + LINE_BREAK2)
    print("CHOOSE ACTION")
    print(LINE_BREAK2)
    if env.flat_actions:
        return choose_flat_action_automated(env, action_idx, num_attack)


# kjkoo : (23-0621)
#   - 사용자 입력 없이 정답 공격시퀀스 수행
def run_keyboard_agent_automated(env, render_mode="readable"):
    """Run Keyboard agent

    Parameters
    ----------
    env : NASimEnv
        the environment
    render_mode : str, optional
        display mode for environment (default="readable")

    Returns
    -------
    int
        final return
    int
        steps taken
    bool
        whether goal reached or not
    """
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o = env.reset()
    env.render()
    total_reward = 0
    total_action_list = list()
    total_reward_list = list()
    total_steps = 0
    done = False
    step_limit_reached = False
    # kjkoo : (23-0725)
    #   - IP scanning technique 결과 반영
    attack_seq_v01 = [10, 25, 20, 36, 50, 47,
                      43, 40, 39, 28, 18, 19,
                      55, 53, 61]
    #   - 10 -> 23, 36 -> 49
    #attack_seq_v02 = [23, 25, 20, 49, 50, 47,
    #                  43, 40, 39, 28, 18, 19,
    #                  55, 53, 61]

    # kjkoo : (24-0413)
    #   - Action.target이 action을 실행하는 소스 호스트로 해석
    #   - 공격 테크닉 pps를 적용하기 위해
    #   - 23->10(서브넷 스캐닝), 49->36
    # (24-0502)
    #   - 공격 테크닉 총 개수 : 14개 (1개 증가)
    #   - 11
    #   - 13 (24-0527)
    #   - 8  (24-0528)
    # (24-0613) 정답
    # attack_seq_v02_gt = [11, 13, 8, 39, 40, 37, 47, 44, 42, 45, 6, 21, 32, 69]
    #
    # (24-0807) 정답
    # attack_seq_v02_gt = [ ]
    attack_seq_v02_gt = [12, 14, 9, 44, 45, 42, 63, 53, 50, 48, 51, 7, 8, 6, 36]

    # (24-0816) Random Agent
    attack_seq_v02 = [12, 14, 9, 44, 45, 42, 63, 53, 50, 48, 51, 7, 8, 6, 36]
    #
    # [12, 14, 9, 44, 29, 42, 63, 50, 53, 48, 51, 40, 38, 36]
    #

    # attack_seq_v02 = [11, 27, 22, 39, 54,
    #                   51, 47, 44, 42, 31,
    #                   20, 21, 60]  # , 53, 61]

    # debugging
    attack_seq_v23_0803 = [1, 23, 55, 25, 11,
                            20, 62, 37, 50, 9,
                            47, 8, 40, 43, 4,
                            60, 39, 13, 64, 56,
                            37, 1, 26, 2, 41,
                            28, 19] #55
    attack_seq_map_idx = 0
    num_attack = 0
    while not done and not step_limit_reached:
        # kjkoo : (23-0621) : 자동으로 action을 수행하도록 변경
        #   - 나중에 해야 겠네.. 반자동이 필요
        #a = choose_action_automated(env, attack_seq_v02[attack_idx])
        if attack_seq_map_idx <= len(attack_seq_v02) - 1:
            a = choose_action_automated(env, attack_seq_v02[attack_seq_map_idx], num_attack)
            attack_seq_map_idx += 1
        else:
            attack_idx = choose_flat_action_semi_automated(env)
            a = choose_action_automated(env, attack_idx, num_attack)

        o, r, done, step_limit_reached, _ = env.step(a, 'attacker')

        total_reward += r
        total_action_list.append(a.name)
        total_reward_list.append(r)
        total_steps += 1
        print("\n" + LINE_BREAK2)
        print("OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        env.render()
        print(f"Reward={r}")
        print(f"Done={done}")
        print(f"Step limit reached={step_limit_reached}")
        print(LINE_BREAK)

        num_attack += 1

        #if attack_idx < len(attack_seq_v02)-1:
        #if attack_idx < len(attack_seq_v23_0803) - 1:
        #    attack_idx += 1
        #else:
        #    print(f"[+] Episode ended forced")
        #    attack_idx = choose_flat_action_semi_automated(env)
        #    #break

    return total_reward, total_steps, done, total_action_list, total_reward_list


def run_keyboard_agent(env, render_mode="readable"):
    """Run Keyboard agent

    Parameters
    ----------
    env : NASimEnv
        the environment
    render_mode : str, optional
        display mode for environment (default="readable")

    Returns
    -------
    int
        final return
    int
        steps taken
    bool
        whether goal reached or not
    """
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o = env.reset()
    env.render()
    total_reward = 0
    total_steps = 0
    done = False
    step_limit_reached = False
    while not done and not step_limit_reached:
        # kjkoo : (23-0621) : 자동으로 action을 수행하도록 변경
        #   - 나중에 해야 겠네.. 반자동이 필요
        #a = choose_action_automated(env, )
        a = choose_action(env)
        o, r, done, step_limit_reached, _ = env.step(a, 'attacker')
        total_reward += r
        total_steps += 1
        print("\n" + LINE_BREAK2)
        print("Attacker_OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        env.render()
        print(f"Reward={r}")
        print(f"Done={done}")
        print(f"Step limit reached={step_limit_reached}")
        print(LINE_BREAK)

    return total_reward, total_steps, done






def run_generative_keyboard_agent(env, render_mode="human"):
    """Run Keyboard agent in generative mode.

    The experience is the same as the normal mode, this is mainly useful
    for testing.

    Parameters
    ----------
    env : NASimEnv
        the environment
    render_mode : str, optional
        display mode for environment (default="readable")

    Returns
    -------
    int
        final return
    int
        steps taken
    bool
        whether goal reached or not
    """
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o, o_d, _ = env.reset()

    env.render()

    total_reward = 0
    total_reward_d = 0
    total_steps = 0
    done = False
    while not done:

        a_d = choose_d_action(env)
        on_d, r_d, done_d, _, _ = env.step(a_d, 'defender')
        total_reward_d += r_d
        total_steps += 1
        print(LINE_BREAK2)
        print("NEXT STATE")
        print(LINE_BREAK2)
        env.render_obs(render_mode, None)
        print("\n" + LINE_BREAK2)
        print("Defender_OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        print(f"Reward = {r_d}")
        print(f"Done = {done_d}")
        print(LINE_BREAK)

        a = choose_action(env)
        on, r, done, _, _ = env.step(a,'attacker')
        total_reward += r
        total_steps += 1
        print(LINE_BREAK2)
        print("NEXT STATE")
        print(LINE_BREAK2)
        env.render_obs(render_mode, None)
        print("\n" + LINE_BREAK2)
        print("Attacker_OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        print(f"Reward = {r}")
        print(f"Done = {done}")
        print(LINE_BREAK)

        if done:
            done = env.attacker_goal_reached()
            break

    return total_reward, total_steps, done


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="random seed (default=None)")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-g", "--use_generative", action="store_true",
                        help=("Generative environment mode. This makes no"
                              " difference for the player, but is useful"
                              " for testing."))
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=not args.param_actions,
                               flat_obs=True)

    # kjkoo :
    #   - auto
    #auto = False
    auto = True
    if args.use_generative:
        total_reward, steps, goal = run_generative_keyboard_agent(env)
    else:
        if auto:
            total_reward, steps, goal, total_action_list, total_reward_list = run_keyboard_agent_automated(env)
        else:
            total_reward, steps, goal = run_keyboard_agent(env)

    print(LINE_BREAK2)
    print("EPISODE FINISHED")
    print(LINE_BREAK)
    print(f"Goal reached = {goal}")
    print(f"Total reward = {total_reward}")
    if auto:
        print(f"Total action list = {total_action_list}")
        print(f"Total reward list = {total_reward_list}")
    print(f"Steps taken = {steps}")
