import collections
import os
import shutil
import random
import hashlib
import math
import json
from copy import deepcopy


def del_all_files(root_dir):
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exits = os.path.exists(path)
    if not is_exits:
        os.makedirs(path)
        print(path + ' created')
        return True
    else:
        print(path + ' already existed')
        return False


def get_sim_configs(configs):
    sim_configs = {"sys_os": configs["sys_os"], "output": configs["output"], "freq": configs["freq"],
                   "vin": configs["vin"], "D": configs["D"], "must_have_switch": configs["must_have_switch"],
                   "prohibit_path": configs["prohibit_path"], "approve_path": configs["approve_path"]}
    return sim_configs


def get_args(args_file_name, configs):
    fo_conf = open(args_file_name, "r")

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["max_episode_length"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["deterministic"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["deterministic"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["deterministic"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["ucb_scalar"] = float(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["gamma"] = float(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["rave_k"] = float(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["rave_scalar"] = float(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["leaf_value"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["end_episode_value"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["algorithm"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["root_parallel"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["root_parallel"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["root_parallel"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["act_selection"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["tree_num"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["thread_num"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["position_num"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["game_num"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["dep_start"] = int(text_list[1])
    text = fo_conf.readline()
    text_list = text.split("=")
    configs["dep_end"] = int(text_list[1])
    text = fo_conf.readline()
    text_list = text.split("=")
    configs["dep_step_len"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["traj_start"] = int(text_list[1])
    text = fo_conf.readline()
    text_list = text.split("=")
    configs["traj_end"] = int(text_list[1])
    text = fo_conf.readline()
    text_list = text.split("=")
    configs["traj_step_len"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["sys_os"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["output"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["output"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["output"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["num_component"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["freq"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["vin"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    text_list[1] = text_list[1][:-1]
    configs["D"] = text_list[1]

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["mutate_num"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["mutate_generation"] = int(text_list[1])

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["must_have_switch"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["must_have_switch"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["must_have_switch"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["prohibit_path"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["prohibit_path"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["prohibit_path"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["approve_path"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["approve_path"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["approve_path"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["whole_default_policy"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["whole_default_policy"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["whole_default_policy"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["component_default_policy"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["component_default_policy"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["component_default_policy"] = True

    text = fo_conf.readline()
    text_list = text.split("=")
    configs["path_default_policy"] = True
    text_list[1] = text_list[1][:-1]
    if text_list[1] == "False" or text_list[1] == "F":
        configs["path_default_policy"] = False
    elif text_list[1] == "True" or text_list[1] == "T":
        configs["path_default_policy"] = True

    fo_conf.close()
    return 0


def init_position(position_number):
    init_position_X = []
    init_position_Y = []
    initFood = []
    init_nums = position_number

    init_size = 0
    ran_size = 0
    while True:
        init_flag = 0
        tmp_x = random.randint(0, 4)
        tmp_y = random.randint(0, 4)
        tmp_food = random.randint(0, 4)
        ran_size += 1
        for init_index in range(0, len(init_position_X)):
            if (init_position_X[init_index] == tmp_x and init_position_Y[init_index] == tmp_y) \
                    or tmp_y == tmp_food:
                init_flag = 1
                break
        if init_flag == 1:
            continue
        else:
            init_position_X.append(tmp_x)
            init_position_Y.append(tmp_y)
            initFood.append(tmp_food)
            init_size += 1
            if init_size == init_nums:
                break
    # (self, _sim, _maxDepth, _numRuns, _ucbScalar, _gamma, _leafValue, _endEpisodeValue):
    position_list = []
    for i in range(0, init_size):
        position_list.append((init_position_X[i], init_position_Y[i], initFood[i]))
    return position_list


def generate_depth_list(start, end, step_len):
    max_depth = []
    for i in range(start, end, step_len):
        max_depth.append(i)
    return max_depth


def read_approve_path(file_name='approve_path_vout_FET.txt'):
    approve_path = []
    path_fo = open(file_name, "r")
    while True:
        line = path_fo.readline()
        if not line:
            break
        lines = line.split(':')
        path_str = lines[1]
        line = path_fo.readline()
        lines = line.split(':')
        freq_str = lines[1]
        approve_path.append((path_str[1:-1], float(freq_str[1:-1])))
    print("path with frequency:")
    for path_freq in approve_path:
        print(path_freq)
    return approve_path


def sort_components(components):
    # components = ['capacitor', 'FET-A', 'inductor', 'FET-A']
    priorities = []
    # component_2_priority = {'Sa': 0, 'Sb': 1, 'L': 2, 'C': 3}
    component_2_priority = {'FET-A': 0, 'FET-B': 1, 'inductor': 2, 'capacitor': 3}
    # priority_2_component = {0: 'Sa', 1: 'Sb', 2: 'L', 3: 'C'}
    priority_2_component = {0: 'FET-A', 1: 'FET-B', 2: 'inductor', 3: 'capacitor'}
    for component in components:
        priorities.append(component_2_priority[component])
    priorities.sort()
    components = []
    for priority in priorities:
        components.append(priority_2_component[priority])
    return components


def read_joint_component_prob(num_component, basic_components=None,
                              file_name='node_joint_probs_FET.json'):
    if basic_components is None:
        basic_components = ['FET-A', 'FET-B', 'inductor', 'capacitor']
    component_condition_prob = []
    component_prob_dict = {}
    component_condition_prob = {}
    with open(file_name) as f:
        data = json.load(f)
    i = 0
    sub_graphs = [[]]
    for sub_graph in sub_graphs:

        for component in basic_components:
            tmp_sub_graph = deepcopy(sub_graph)
            tmp_sub_graph.append(component)
            if len(tmp_sub_graph) > num_component:
                break
            sub_graphs.append(tmp_sub_graph)
        if len(tmp_sub_graph) > num_component:
            break
    # print(sub_graphs)
    for sub_graph_prob in data:
        component_prob_dict[str(sort_components(sub_graph_prob[0]))] = sub_graph_prob[1]
    # for k, v in component_prob_dict.items():
    #     print(k, ":", v)
    component_prob_dict[str([])] = 1
    for sub_graph in sub_graphs:
        # print("")
        sum_freq = 0
        if len(sub_graph) < num_component:
            if str(sort_components(sub_graph)) not in component_prob_dict:
                for component in basic_components:
                    tmp_sub_graph = deepcopy(sub_graph)
                    tmp_sub_graph.append(component)
                    if len(tmp_sub_graph) <= num_component:
                        condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                        str(sort_components(sub_graph))
                        component_condition_prob[condition_str] = 1
                        sum_freq += 1
            else:
                for component in basic_components:
                    tmp_sub_graph = deepcopy(sub_graph)
                    tmp_sub_graph.append(component)
                    if len(tmp_sub_graph) <= num_component:
                        condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                        str(sort_components(sub_graph))
                        if str(sort_components(tmp_sub_graph)) in component_prob_dict:
                            condition_freq = component_prob_dict[str(sort_components(tmp_sub_graph))] \
                                             / component_prob_dict[str(sort_components(sub_graph))]
                        else:
                            condition_freq = 0
                        component_condition_prob[condition_str] = condition_freq
                        sum_freq += condition_freq
            for component in basic_components:
                tmp_sub_graph = deepcopy(sub_graph)
                tmp_sub_graph.append(component)
                if len(tmp_sub_graph) <= num_component:
                    condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                    str(sort_components(sub_graph))
                    component_condition_prob[condition_str] = component_condition_prob[condition_str] / sum_freq
                    # print(condition_str, ':', component_condition_prob[condition_str])
    return component_condition_prob


def sample_component(sub_graph, joint_component_prob, allowed_actions, basic_components=None):
    if basic_components is None:
        basic_components = ["FET-A", "FET-B", "capacitor", "inductor"]
    tmp_sub_graph = deepcopy(sub_graph)
    tmp_sub_graph.remove('VOUT')
    tmp_sub_graph.remove('VIN')
    tmp_sub_graph.remove('GND')
    for i in range(len(tmp_sub_graph)):
        if 'FET-A' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'FET-A'
        elif 'FET-B' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'FET-B'
        elif 'inductor' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'inductor'
        elif 'capacitor' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'capacitor'
    sample_prob = random.random()
    sum_prob = 0
    action_weights = []
    for action in allowed_actions:
        component = basic_components[action.value]
        append_sub_graph = deepcopy(tmp_sub_graph)
        append_sub_graph.append(component)
        condition_str = str(sort_components(append_sub_graph)) + '|' + str(sort_components(tmp_sub_graph))
        action_weights.append(joint_component_prob[condition_str])

    print(action_weights)
    # In case all the prob is 0, which means random
    if sum(action_weights) == 0:
        action_weights = [1 for _ in action_weights]
    if action_weights:
        action_list = random.choices(allowed_actions, weights=action_weights, k=1)
        return action_list[0]
    else:
        return None


def generate_traj_List(start, end, step_len):
    traj = []
    for i in range(start, end, step_len):
        traj.append(i)
    return traj


def save_reward_hash(sim):
    hash_file_name = "reward_hash.txt"
    hash_fo = open(hash_file_name, "w")
    for (k, v) in sim.graph_2_reward.items():
        hash_fo.write(str(k) + '#' + str(v) + '\n')
    hash_fo.close()


def save_one_in_hash_file(topology, efficiency, hash_file_name="reward_hash.txt"):
    hash_fo = open(hash_file_name, "a")
    hash_fo.write(str(topology) + '$' + str(efficiency) + '\n')
    hash_fo.close()


def hash_str_encoding(graph_str):
    m = hashlib.md5()
    m.update(graph_str.encode('utf-8'))
    return m.hexdigest()


def read_reward_hash_list(hash_file_name="reward_hash.txt"):
    graph_2_reward_tmp = []
    fo_conf = open(hash_file_name, "r")
    # line = fo_conf.readline()
    while True:
        line = fo_conf.readline()
        # print(line)
        if not line:
            break
        key_value = line.split('$')
        topo = key_value[0]

        str_list = key_value[1][1:-2]
        # print(str_list)
        value_str_list = str_list.split(',')
        hash_values = []
        for every_value in value_str_list:
            # print(every_value)
            hash_values.append(float(every_value))
        graph_2_reward_tmp.append([topo, hash_values])
    fo_conf.close()
    return graph_2_reward_tmp


def read_reward_hash(hash_file_name="reward_hash.txt", md5_trans=False):
    graph_2_reward_tmp = {}
    # hash_file_name = "reward_hash.txt"
    fo_conf = open(hash_file_name, "r")
    line = fo_conf.readline()
    while True:
        line = fo_conf.readline()
        # print(line)
        if not line:
            break
        key_value = line.split('$')
        topo = key_value[0]
        if md5_trans:
            md5_topo = hash_str_encoding(topo)
        str_list = key_value[1][1:-2]
        # print(str_list)
        value_str_list = str_list.split(',')
        hash_values = []
        for every_value in value_str_list:
            # print(every_value)
            hash_values.append(float(every_value))
        if md5_trans:
            graph_2_reward_tmp[md5_topo] = hash_values
        else:
            graph_2_reward_tmp[topo] = hash_values
    fo_conf.close()
    return graph_2_reward_tmp


def get_topology_from_hash(topo_str):
    conns = topo_str.split("],")
    # print(conns)
    graph = collections.defaultdict(set)
    for conn in conns:
        conn = conn.split(":")
        # print(conn[0])
        try:
            start_port = int(conn[0])
            end_port_list = conn[1][1:]
            # print(end_port_list)
            end_ports = end_port_list.split(',')
            for end_port in end_ports:
                pass
                graph[start_port].add(int(end_port))
                graph[int(end_port)].add(start_port)
        except:
            continue
    return graph


def get_steps_traj(total_traj, total_step, fix_traj, multi, fix_step_traj=False):
    steps_traj = []
    if fix_step_traj:
        traj = total_traj / total_step
        for i in range(total_step):
            steps_traj.append(traj)
        print(steps_traj)
        return steps_traj
    fix_step_number = 2
    for i in range(fix_step_number):
        steps_traj.insert(0, fix_traj)
    decrease_traj = total_traj - fix_step_number * fix_traj
    last_traj = decrease_traj * (multi - 1) / (math.pow(multi, total_step - fix_step_number) - 1)
    last_traj = int(last_traj)
    remain_traj = decrease_traj - last_traj * (math.pow(multi, total_step - fix_step_number) - 1) / (multi - 1)
    print(remain_traj)
    # if last_traj < fix_traj:
    # 	print(last_traj, fix_traj, "Error of get steps traj(last traj < fix_traj)")
    # 	return -1
    traj = last_traj
    for i in range(total_step - fix_step_number):
        steps_traj.insert(0, traj)
        traj *= 2
    steps_traj[0] += remain_traj
    print(steps_traj)
    print(sum(steps_traj), total_traj)

    return steps_traj


def Merge(dict1, dict2):
    dict2.update(dict1)
    return dict2


# get_steps_traj(9*80,9,6,2,False)
# read_approve_path()
# joint_prob = read_joint_component_prob(3)
# allowed_actions = []
# action = sample_component(['FET-A-0'], joint_prob, allowed_actions)
# print(sort_components([]))


component_condition_prob = read_joint_component_prob(3)
for k, v in component_condition_prob.items():
    print(k, ":", v)

# component_condition_prob = read_joint_component_prob(3, None, '../node_joint_probs_FET.json')
# for k, v in component_condition_prob.items():
#     print(k, ":", v)

#
# a = random.choices([1, 2, 3], weights=[0.1, 0.2, 0.3], k=1)
# print(a)
# action_weights = [3,2,3]
# action_weights = [1 for _ in action_weights]
# print(action_weights)
