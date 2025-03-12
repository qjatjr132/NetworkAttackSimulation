import os
import yaml
import os.path as osp


SCENARIO_DIR = osp.dirname(osp.abspath(__file__))

# default subnet address for internet
INTERNET = 0

# Constants
NUM_ACCESS_LEVELS = 2
NO_ACCESS = 0
USER_ACCESS = 1
ROOT_ACCESS = 2
DEFAULT_HOST_VALUE = 0

# scenario property keys
SUBNETS = "subnets"
TOPOLOGY = "topology"
SENSITIVE_HOSTS = "sensitive_hosts"
SERVICES = "services"
OS = "os"
PROCESSES = "processes"
EXPLOITS = "exploits"
PRIVESCS = "privilege_escalation"
SERVICE_SCAN_COST = "service_scan_cost"
OS_SCAN_COST = "os_scan_cost"
SUBNET_SCAN_COST = "subnet_scan_cost"
PROCESS_SCAN_COST = "process_scan_cost"
HOST_CONFIGS = "host_configurations"
FIREWALL = "firewall"
HOSTS = "host"
STEP_LIMIT = "step_limit"
ACCESS_LEVELS = "access_levels"
ADDRESS_SPACE_BOUNDS = "address_space_bounds"

# kjkoo (23-0315) attack techniques
ATTACK_TECHS = "attack_techs"
ATTACK_PPS = "attack_pps"       # attack pre/post-state
ATTACK_PPS_DICT = "attack_pps_dict"       # attack pre/post-state
FILES = "files"
VULNERABILITIES = "vulnerabilities"


# scenario exploit keys
EXPLOIT_SERVICE = "service"
EXPLOIT_OS = "os"
EXPLOIT_PROB = "prob"
EXPLOIT_COST = "cost"
EXPLOIT_ACCESS = "access"

# scenario privilege escalation keys
PRIVESC_PROCESS = "process"
PRIVESC_OS = "os"
PRIVESC_PROB = "prob"
PRIVESC_COST = "cost"
PRIVESC_ACCESS = "access"

# host configuration keys
HOST_SERVICES = "services"
HOST_PROCESSES = "processes"
HOST_OS = "os"
HOST_FIREWALL = "firewall"
HOST_VALUE = "value"
# kjkoo (23-0523 추가)
HOST_FILES = "files"
HOST_VULNERABILITIES = "vulnerabilities"
# kjkoo : (23-0616) 추가
HOST_IPS = "ips"
HOST_PW_SHARED = "pw_shared"

# Defender
DEFENDER_TECHS = "defense_techs"
DEFENDER_PPS = "defend_pps"
DEFENDER_PPS_DICT = "defend_pps_dict"

def load_yaml(file_path):
    """Load yaml file located at file path.

    Parameters
    ----------
    file_path : str
        path to yaml file

    Returns
    -------
    dict
        contents of yaml file

    Raises
    ------
    Exception
        if theres an issue loading file. """
    with open(file_path, encoding='UTF-8') as fin:
        content = yaml.load(fin, Loader=yaml.FullLoader)
    return content


def get_file_name(file_path):
    """Extracts the file or dir name from file path

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    str
        file name with any path and extensions removed
    """
    full_file_name = file_path.split(os.sep)[-1]
    file_name = full_file_name.split(".")[0]
    return file_name