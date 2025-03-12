# kjkoo (23-0131)

# 공격 도구 파일에서 pre-/post-state 정보 읽어오기
# TODOv : 공격 도구 파일(.yaml)에서 pre-/post-state 정보 읽어오기


import os
import yaml
import os.path as osp


# TODOv : category, supported_platform 도 추출해라. (23-0131)
# TODOv : category별로 technique 정렬
# TODOv : category별 테크닉 개수 표시
#


# constant
LINE_BREAK = f"-" * 80
LINE_BREAK2 = f"=" * 80


class AttackTechs:
    def __init__(self):
        self.pre_state_key_value_dict = dict()
        self.post_state_key_value_dict = dict()
        self.category_list = list()
        self.techs_per_category_dict = dict()
        self.pre_state_type_list = list()

        self.pre_state_feature = list()
        self.post_state_feature = list()


    def load_techs(self):
        ATTACK_TECH_DIR = osp.dirname(osp.abspath(__file__)) + '/tiny-etri'
        print(ATTACK_TECH_DIR)

        attack_list = os.listdir(ATTACK_TECH_DIR)
        print(f"[+] attack_techniques_files = {attack_list}")

        category_list = []
        techs_per_category_dict = {}
        supported_platforms_list = []

        prs_key_val_dict = self.pre_state_key_value_dict
        pos_key_val_dict = self.post_state_key_value_dict

        for i, attack_tech in enumerate(attack_list):
            print(f"    [{i}] {attack_tech}")

            # 공격테크닉.yaml 파일 읽기
            file_path = ATTACK_TECH_DIR + '/' + attack_tech
            with open (file_path, encoding='UTF-8') as fin:
                content = yaml.load(fin, Loader=yaml.FullLoader)

            # 테크닉 body
            atomic_tests = content['atomic_tests']
            atomic_tests_val = atomic_tests[0]

            # pre_state
            pre_state = atomic_tests_val['pre_state']
            for idx in range(len(pre_state)):
                prs = pre_state[idx]
                prs_key = prs['key']
                prs_val = prs['value']

                # pre_state key 중복 체크
                if prs_key not in prs_key_val_dict.keys():
                    prs_key_val_dict[prs_key] = [prs_val]     # key가 없으면 생성 (초기)
                else:
                    # value 중복 체크
                    if prs_val not in prs_key_val_dict[prs_key]:
                        prs_key_val_dict[prs_key].append(prs_val)

            # post_state
            pos_state = atomic_tests_val['post_state']
            for idx in range(len(pos_state)):
                pos = pos_state[idx]
                pos_key = pos['key']
                pos_val = pos['value']

                # post_state key 중복 체크
                if pos_key not in pos_key_val_dict.keys():
                    pos_key_val_dict[pos_key] = [pos_val]     # key가 없으면 생성 (초기)
                else:
                    # value 중복 체크
                    if pos_val not in pos_key_val_dict[pos_key]:
                        pos_key_val_dict[pos_key].append(pos_val)
        #
        self.pre_state_key_value_dict = prs_key_val_dict
        self.post_state_key_value_dict = pos_key_val_dict

        print(f"[+] pre_state_key_value_dict: \n    {self.pre_state_key_value_dict}")
        print(f"[+] post_state_key_value_dict: \n    {self.post_state_key_value_dict}")

        return

    def gen_feature_pre_state(self):
        prs_dict = self.pre_state_key_value_dict
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

    def gen_feature_post_state(self):
        pos_dict = self.post_state_key_value_dict
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
    atechs = AttackTechs()
    #atechs.analysis_attack_techniques()
    atechs.load_techs()
    atechs.gen_feature_pre_state()
    atechs.gen_feature_post_state()
    atechs.print_pps_feature()