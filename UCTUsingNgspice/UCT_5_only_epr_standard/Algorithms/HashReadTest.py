import _thread
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
import numpy as np
import datetime

from utils.util import init_position, generate_depth_list, del_all_files, mkdir, get_sim_configs, \
	read_reward_hash, save_reward_hash
import copy


def merge_act_nodes(dest_act_node, act_node):
	dest_act_node.avg_return_ = dest_act_node.avg_return_ * dest_act_node.num_visits_ + \
	                            act_node.avg_return_ * act_node.num_visits_
	dest_act_node.num_visits_ += act_node.num_visits_
	dest_act_node.avg_return_ = dest_act_node.avg_return_ / dest_act_node.num_visits_


def get_action_from_trees(uct_tree_list, uct_tree, tree_num=4):
	contain_action_flag = 0
	uct_tree.root_.act_vect_ = []
	for i in range(tree_num):
		for j in range(len(uct_tree_list[i].node_vect_)):
			contain_action_flag = 0
			for x in range(len(uct_tree.root_.act_vect_)):
				if uct_tree.root_.act_vect_[x].equal(uct_tree_list[i].act_vect_[j]):
					contain_action_flag = 1
					break
			if contain_action_flag == 1:
				if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
					merge_act_nodes(uct_tree.root_.node_vect_[x], uct_tree_list[i].node_vect_[j])
			else:
				if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
					uct_tree.root_.act_vect_.append(uct_tree_list[i].act_vect_[j].duplicate())
					uct_tree.root_.node_vect_.append(uct_tree_list[i].node_vect_[j])
	act_node = uct_tree.get_action()
	return act_node


def get_action_from_planners(uct_planner_list, uct_tree, tree_num=4):
	contain_action_flag = 0
	uct_tree.root_.act_vect_ = []
	uct_tree.root_.node_vect_ = []
	for i in range(tree_num):
		for j in range(len(uct_planner_list[i].root_.node_vect_)):
			contain_action_flag = 0
			for x in range(len(uct_tree.root_.act_vect_)):
				if uct_tree.root_.act_vect_[x].equal(uct_planner_list[i].root_.act_vect_[j]):
					contain_action_flag = 1
					break
			if contain_action_flag == 1:
				if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
					j] is not None:
					merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
			else:
				if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
					j] is not None:
					uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
					uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
	act_node = uct_tree.get_action()
	return act_node


def get_action_from_trees_vote(uct_planner_list, uct_tree, tree_num=4):
	action_nodes = []
	counts = {}
	for i in range(tree_num):
		action_nodes.append(uct_planner_list[i].get_action())
	for i in range(len(action_nodes)):
		tmp_count = 0
		if counts.get(action_nodes[i]) is None:
			for j in range(len(action_nodes)):
				if action_nodes[j].equal(action_nodes[i]):
					tmp_count += 1
			counts[action_nodes[i]] = tmp_count
	for action, tmp_count in counts.items():
		if tmp_count == max(counts.values()):
			selected_action = action
	return selected_action


def hash_read_test(depth_list, trajectory, configs, date_str):
	out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
	figure_folder = "figures/" + date_str + "/"
	mkdir(figure_folder)
	sim_configs = get_sim_configs(configs)
	uct_tree_list = []
	cumulate_reward_list = []
	start_time = datetime.datetime.now()

	fo = open(out_file_name, "w")
	fo.write("max_depth,num_runs,avg_step\n")
	avg_step_list = []
	init_nums = 1
	for max_depth in depth_list:
		for num_runs in trajectory:
			print("max depth is", max_depth, ",trajectory is", num_runs, "every thread has ",
			      int(num_runs / configs["tree_num"]), " trajectories")
			avg_steps = 0

			for j in range(0, init_nums):
				print()
				cumulate_reward_list = []
				fo.write("----------------------------------------------------------------------" + "\n")
				uct_simulators = []
				for i in range(0, int(configs["game_num"] / init_nums)):
					steps = 0
					avg_cumulate_reward = 0
					cumulate_plan_time = 0
					final_reward = 0
					r = 0
					tree_size = 0

					sim = TopoPlanner.TopoGenSimulator(sim_configs, configs['num_component'])
					sim.graph_2_reward = read_reward_hash()
					print(sim.graph_2_reward)
					uct_tree = uct.UCTPlanner(sim, max_depth, num_runs, configs["ucb_scalar"], configs["gamma"],
					                          configs["leaf_value"], configs["end_episode_value"],
					                          configs["deterministic"])
					uct_simulators.clear()
					uct_tree_list.clear()
					for n in range(configs["tree_num"]):
						uct_simulators.append(TopoPlanner.TopoGenSimulator(sim_configs, configs['num_component']))
						uct_simulators[n].graph_2_reward = read_reward_hash()
						uct_tree_list.append(
							uct.UCTPlanner(uct_simulators[n], max_depth, int(num_runs / configs["tree_num"]),
							               configs["ucb_scalar"],
							               configs["gamma"],
							               configs["leaf_value"], configs["end_episode_value"],
							               configs["deterministic"]))

					# For fixed commponent type
					init_nodes = [0, 3, 1]
					for e in init_nodes:
						action = TopoPlanner.TopoGenAction('node', e)
						sim.act(action)
						# 0:[3],1:[8],2:[6],3:[0],4:[7],5:[8],6:[2, 7],7:[4, 6],8:[1, 5],
					edges = [[0, 3], [1, 8], [2, 6], [4, 7], [5, 8], [4, 6], [6, 7]]

					# edges = []
					for edge in edges:
						action = TopoPlanner.TopoGenAction('edge', edge)
						sim.act(action, False)
						print("!!!!!!!!!!!!!!!!!!!!!!!!!!!", sim.current.graph,
						      TopoPlanner.sort_dict_string(sim.current.graph) in sim.graph_2_reward)
					time.sleep(3)
					print(sim.get_reward())
					return
