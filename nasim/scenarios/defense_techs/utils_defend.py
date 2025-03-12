import os
import yaml
import os.path as osp

# constant
LINE_BREAK = f"-" * 80
LINE_BREAK2 = f"=" * 80


class DefendTechs:
    def __init__(self):
        self.pre_state_key_value_dic = dict()
        self.post_state_key_value_dic = dict()

        self.pre_state_feature = list()
        self.post_state_feature = list()

    def load_techs(self):
        DEFEND_TECH_DIR = osp.dirname(osp.abspath(__file__)) + '/Mitigation'
        print(DEFEND_TECH_DIR)

        defend_list = os.listdir(DEFEND_TECH_DIR)
        print(f"[+] defend_techniques_files = {defend_list}")

        pre_state_key_value_dic = self.pre_state_key_value_dic
        post_state_key_value_dic = self.post_state_key_value_dic

        for i, defend_tech in enumerate(defend_list):
            print(f"    [{i}] {defend_tech}")

            # 공격테크닉.yaml 파일 읽기
            file_path = DEFEND_TECH_DIR + '/' + defend_tech
            with open (file_path, encoding='UTF-8') as fin:
                content = yaml.load(fin, Loader=yaml.FullLoader)

            # 테크닉 body
            atomic_tests = content['atomic_tests']
            atomic_tests_val = atomic_tests[0]

            # change_state
            pre_state = atomic_tests_val['pre_state']
            for idx in range(len(pre_state)):
                chs = pre_state[idx]
                prs_key = chs['key']
                prs_val = chs['value']

                # pre_state key 중복 체크
                if prs_key not in pre_state_key_value_dic.keys():
                    pre_state_key_value_dic[prs_key] = [prs_val]     # key가 없으면 생성 (초기)
                else:
                    # value 중복 체크
                    if prs_val not in pre_state_key_value_dic[prs_key]:
                        pre_state_key_value_dic[prs_key].append(prs_val)

            # post_state
            post_state = atomic_tests_val['post_state']
            for idx in range(len(post_state)):
                pos = post_state[idx]
                pos_key = pos['key']
                pos_val = pos['value']

                # post_state key 중복 체크
                if pos_key not in post_state_key_value_dic.keys():
                    post_state_key_value_dic[pos_key] = [pos_val]     # key가 없으면 생성 (초기)
                else:
                    # value 중복 체크
                    if pos_val not in post_state_key_value_dic[pos_key]:
                        post_state_key_value_dic[pos_key].append(pos_val)
        #
        self.pre_state_key_value_dic = pre_state_key_value_dic
        self.post_state_key_value_dic = post_state_key_value_dic

        print(f"[+] pre_state_key_value_dic: \n    {self.pre_state_key_value_dic}")
        print(f"[+] post_state_key_value_dic: \n    {self.post_state_key_value_dic}")

        return

    def gen_feature_change_state(self):
        prs_dict = self.pre_state_key_value_dic
        list_pre_state = list(prs_dict.keys())

        if 'Platform' in list_pre_state:
            os_list = prs_dict['Platform']
            list_pre_state.remove('Platform')
            print(f"[+] os_list : {os_list}")
            for os in os_list:
                list_pre_state.append(os)

        print(f"[+] pre_state feature : {list_pre_state}")
        self.pre_state_feature = list_pre_state
        return

    def gen_feature_stop_state(self):
        pos_dict = self.post_state_key_value_dic
        list_post_state = list(pos_dict.keys())

        print(f"[+] post_state feature : {list_post_state}")
        self.post_state_feature = list_post_state
        return

    def print_pps_feature(self):
        print(f"[+] pre_state feature")
        for pre in self.pre_state_feature:
            print(f"- {pre}")

        print(f"[+] post_state feature")
        for pos in self.post_state_feature:
            print(f"- {pos}")




if __name__=='__main__':
    #Rextract_attack_pre_post_state()
    atechs = DefendTechs()
    #atechs.analysis_attack_techniques()
    atechs.load_techs()
    atechs.gen_feature_change_state()
    atechs.gen_feature_stop_state()
    atechs.print_pps_feature()