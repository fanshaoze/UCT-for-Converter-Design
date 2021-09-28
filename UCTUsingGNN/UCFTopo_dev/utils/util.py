import collections
import os
import shutil
import random
import hashlib

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
	sim_configs = {"sys_os": configs["sys_os"], "output": configs["output"],
	               "freq": configs["freq"], "vin": configs["vin"], "D": configs["D"]}
	return sim_configs


def get_args(args_file_name, configs):
	fo_conf = open(args_file_name, "r")

	text = fo_conf.readline()
	text_list = text.split("=")
	configs["max_episode_length"] = int(text_list[1])

	text = fo_conf.readline()
	text_list = text.split("=")
	text_list[1] = text_list[1][:-1]
	configs["deterministic"] = text_list[1]

	text = fo_conf.readline()
	text_list = text.split("=")
	configs["ucb_scalar"] = float(text_list[1])

	text = fo_conf.readline()
	text_list = text.split("=")
	configs["gamma"] = float(text_list[1])

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

def hash_str(graph_str):
    m = hashlib.md5()
    m.update(graph_str.encode('utf-8'))
    return m.hexdigest()


def read_reward_hash(md5_trans=False):
	graph_2_reward_tmp = {}
	hash_file_name = "reward_hash.txt"
	fo_conf = open(hash_file_name, "r")
	while True:
		line = fo_conf.readline()
		# print(line)
		if not line:
			break
		key_value = line.split('#')
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
