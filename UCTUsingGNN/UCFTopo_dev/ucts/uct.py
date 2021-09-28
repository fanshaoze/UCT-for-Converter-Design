"""The UCT model.

This file include the part of UCT, including the definitions and common methods.
"""

import sys
import math
import random
import os


class State(object):
	"""The definiton of State.

    This is the definiton of State in UCT, the attributes need to be defined
    seperately in different game models
    """

	def equal(self, state):
		"""Find out whether two states are equal.

        Args:
          None.

        Returns:
          True if two states are equal.
        """

		pass

	def duplicate(self):
		"""Return a deep copy of itself.
        """

		pass

	def print(self):
		pass

	def __del__(self):
		pass


class SimAction(object):
	"""The definition of action.

    This is the definition of State in UCT, the attributes need to be defined
    separately in different game models
    """

	def equal(self, act):
		"""Find out whether two actions are equal.

        Args:
          None.

        Returns:
          True if two actions are equal.
        """

		pass

	def duplicate(self):
		pass

	def print_state(self):
		pass

	def __del__(self):
		pass


class Simulator(object):
	"""state.

    Defined according to the instances
    Longer class information....
    """

	def set_state(self, state):  # Done
		pass

	def get_state(self):
		pass

	def act(self, action):  # equal(SimAction* act) = 0;#Done
		pass

	def get_actions(self):  # Done
		pass

	def is_terminal(self):  # Done
		pass

	def reset(self):  # Done
		pass


class StateNode(object):

	def __init__(self, _parent_act, _state, _act_vect, _reward, _is_terminal):  # Done
		self.parent_act_ = _parent_act

		self.state_ = _state.duplicate()
		self.reward_ = _reward
		self.is_terminal_ = _is_terminal
		self.num_visits_ = 0
		self.act_ptr_ = 0
		self.first_MC_ = 0
		self.act_vect_ = []
		_size = len(_act_vect)

		for i in range(0, _size):
			self.act_vect_.append(_act_vect[i].duplicate())
		random.shuffle(self.act_vect_)
		# dictionary
		self.node_vect_ = []  # vector<ActionNode*> node_vect_;

	def __del__(self):  # Done
		del self.state_
		self.act_vect_.clear()

	def is_full(self):  # Done
		return self.act_ptr_ == len(self.act_vect_)

	def add_action_node(self):  # Done
		assert self.act_ptr_ < len(self.act_vect_)
		self.node_vect_.append(ActionNode(self))
		self.act_ptr_ += 1
		return self.act_ptr_ - 1

	# #TODO hash function improve
	# m = hashlib.md5()
	# m.update(world_str.encode('utf-8'))
	# return m.hexdigest()


class ActionNode(object):

	def __init__(self, _parent_state):

		self.parent_state_ = _parent_state
		self.avg_return_ = 0
		self.num_visits_ = 0
		self.state_vect_ = []

	def contain_next_state(self, state):
		size = len(self.state_vect_)
		for i in range(0, size):
			if state.equal(self.state_vect_[i].state_):
				return True
		return False

	def get_next_state_node(self, state):
		size = len(self.state_vect_)
		for i in range(0, size):
			if state.equal(self.state_vect_[i].state_):
				return self.state_vect_[i]
		return None

	def add_state_node(self, _state, _act_vect, _reward, _is_terminal):  # Done
		index = len(self.state_vect_)
		self.state_vect_.append(StateNode(self, _state, _act_vect, _reward, _is_terminal))
		return self.state_vect_[index]

	def __del__(self):  # Done
		pass


class UCTPlanner(object):

	def __init__(self, _sim, _max_depth, _num_runs, _ucb_scalar,
	             _gamma, _leaf_value, _end_episode_value, _deterministic):
		self.sim_ = _sim
		self.max_depth_ = _max_depth
		self.num_runs_ = _num_runs
		self.ucb_scalar_ = _ucb_scalar
		self.gamma_ = _gamma
		self.leaf_value_ = _leaf_value
		self.end_episode_value_ = _end_episode_value
		self.deterministic_ = _deterministic

		self.root_ = None

		_leaf_value = 0
		_end_episode_value = 0

		# keep a list of nodes
		# for easier implementation, not including root (which is empty anyway)
		self.node_list = []

	def __del__(self):
		pass

	def get_all_states(self):
		all_states = []
		for node in self.node_list:
			state = node.state_
			if not any(state.equal(s_) for s_ in all_states):
				all_states.append(state)

		return all_states

	def set_root_node(self, _state, _act_vect, _reward, _is_terminal):
		if self.root_ is not None:
			self.clear_tree()
		self.root_ = StateNode(None, _state, _act_vect, _reward, _is_terminal)
		# save a pointer to the root
		self.original_root_ = self.root_

	def clear_tree(self):
		self.root_ = None
		pass

	def set_and_plan(self, conn):
		while True:
			action = conn.recv()
			new_root = conn.recv()
			self.sim_.graph_2_reward = conn.recv()
			if action == -1:
				self.set_root_node(new_root.state_, new_root.act_vect_, new_root.reward_, new_root.is_terminal_)
			else:
				self.update_root_node(action, new_root)
			tree_size, tmp_planner, depth = self.plan()
			conn.send(tree_size)
			conn.send(tmp_planner)
			conn.send(depth)

	def add_all_nodes_2_list(self, root, node_list):
		for action_node in root.node_vect_:
			for state_node in action_node.state_vect_:
				node_list.append(state_node)
				self.add_all_nodes_2_list(state_node, node_list)

	def plan(self, get_viz=False, step_count=0, thread_id=9, return_dict={}):
		"""
        The MCTS process
        currently the parameters are not used
        """

		# fo = open(str(os.getpid()) + "-scalar_test.txt", "a")
		update_list = []
		size = len(self.root_.act_vect_)
		update_sim_lists = [[] for i in range(size)]
		if get_viz:
			node_list = [self.root_]  # for viz
			self.add_all_nodes_2_list(self.root_, node_list)
		current = self.root_
		self.sim_.set_state(current.state_)
		root_topo_reward = self.sim_.get_reward()

		tree_size = 1
		assert self.root_ is not None
		root_offset = 0
		if root_offset == 0:
			self.root_.num_visits_ += 1
			root_offset += 1
		num_runs = self.num_runs_
		for trajectory in range(root_offset, num_runs):
			current = self.root_
			mc_return = self.leaf_value_
			depth = 0
			tmp_uct_branch = -1
			while True:
				depth += 1
				self.sim_.set_state(current.state_)

				# is current is terminal, we need to stop
				if self.sim_.is_terminal():
					mc_return = self.end_episode_value_
					break
				if current.is_full():
					if current == self.root_:
						uct_branch = self.get_UCT_root_index(current)
						# fo.write("round:" + str(tree_size) + "\n")
						det = math.log(float(current.num_visits_))
						# fo.write("Ln(n):" + str(det) + "\n")
						# fo.write("graph:" + str(current.state_.graph)+"\n")
						maximizer = []  # maximizer.clear()
						size = len(current.node_vect_)
						for i in range(0, size):
							val = current.node_vect_[i].avg_return_
							val += self.ucb_scalar_ * math.sqrt(det / float(current.node_vect_[i].num_visits_))
							maximizer.append(val)
							# fo.write(str(i) + " action:" + str(current.act_vect_[i].value) +
							#          " n' of" + ":" + str(float(current.node_vect_[i].num_visits_)) +
							#          " sqrt:" + str(math.sqrt(det / float(current.node_vect_[i].num_visits_))) +
							#          " Qv':" + str(
							# 	float(current.node_vect_[i].num_visits_) * current.node_vect_[i].avg_return_) +
							#          " avg:" + str(current.node_vect_[i].avg_return_) + "\n")

							update_list.append(current.node_vect_[i].avg_return_)
						# fo.write("+++++++++++++++++++++++++++++++++++++++++++" + "\n")
						tmp_uct_branch = uct_branch
					else:
						uct_branch = self.get_UCT_branch_index(current)
					if self.deterministic_ is True:
						current = current.node_vect_[uct_branch].state_vect_[0]
						continue
					else:
						self.sim_.set_state(current.state_)
						r = self.sim_.act(current.act_vect_[uct_branch])
						next_state = self.sim_.get_state()
						if current.node_vect_[uct_branch].contain_next_state(next_state):
							# follow path
							current = current.node_vect_[uct_branch].get_next_state_node(next_state)
							continue
						else:
							next_node = current.node_vect_[uct_branch].add_state_node(next_state,
							                                                          self.sim_.get_actions(),
							                                                          r,
							                                                          self.sim_.is_terminal())
							tree_size += 1
							if -1 == self.max_depth_:
								mc_return = self.MC_sampling_terminal(next_node)
							else:
								mc_return = self.MC_sampling_depth(next_node, self.max_depth_ - depth)
							current = next_node
							if get_viz:
								node_list.append(current)  # for viz
							break
				else:

					act_ID = current.add_action_node()
					self.sim_.set_state(current.state_)

					r = self.sim_.act(current.act_vect_[act_ID])
					next_node = current.node_vect_[act_ID].add_state_node(self.sim_.get_state(),
					                                                      self.sim_.get_actions(),
					                                                      r, self.sim_.is_terminal())
					# keep track of node added here
					self.node_list.append(next_node)
					tree_size += 1
					if -1 == self.max_depth_:
						mc_return = self.MC_sampling_terminal(next_node)
					else:
						mc_return = self.MC_sampling_depth(next_node, self.max_depth_ - depth)
					current = next_node
					if get_viz:
						node_list.append(current)  # for viz
					if tmp_uct_branch == -1:
						tmp_uct_branch = act_ID
					break
			root_sub_return = self.update_values(current, mc_return)
			update_sim_lists[tmp_uct_branch].append(root_sub_return)
		# if update_list:
		# 	fo.write("diff max min:"+str(max(update_list)-min(update_list))+" "+str(max(update_list))+" "+str(min(update_list))+"\n")
		#
		# for i in range(len(update_sim_lists)):
		# 	fo.write(str(i)+' ')
		# 	for j in range(len(update_sim_lists[i])):
		# 		fo.write(str(update_sim_lists[i][j]))
		# 		fo.write(' ')
		# 	fo.write('\n')
		#
		# fo.close()
		if get_viz:
			return tree_size, self, depth, node_list
		return tree_size, self, depth

	def get_action(self):
		return self.root_.act_vect_[self.get_greedy_branch_index()]

	def get_most_visited_branch_index(self):
		assert self.root_ is not None
		maximizer = []
		size = len(self.root_.node_vect_)
		for i in range(0, size):
			maximizer.append(self.root_.node_vect_[i].num_visits_)
		return maximizer.index(max(maximizer))

	def get_greedy_branch_index(self):
		assert self.root_ is not None
		maximizer = []
		size = len(self.root_.node_vect_)
		for i in range(0, size):
			maximizer.append(self.root_.node_vect_[i].avg_return_)
		return maximizer.index(max(maximizer))

	def get_UCT_root_index(self, node):
		det = math.log(float(node.num_visits_))
		maximizer = []  # maximizer.clear()
		size = len(node.node_vect_)
		for i in range(0, size):
			val = node.node_vect_[i].avg_return_
			val += self.ucb_scalar_ * math.sqrt(det / float(node.node_vect_[i].num_visits_))
			maximizer.append(val)
		return maximizer.index(max(maximizer))

	def get_UCT_branch_index(self, node):
		"""
        Use the UCB to select the next branch
        """
		det = math.log(float(node.num_visits_))
		maximizer = []
		size = len(node.node_vect_)
		for i in range(0, size):
			val = node.node_vect_[i].avg_return_
			val += self.ucb_scalar_ * math.sqrt(det / float(node.node_vect_[i].num_visits_))

			maximizer.append(val)

		return maximizer.index(max(maximizer))

	def update_values(self, node, mc_return):
		total_return = mc_return
		if node.num_visits_ == 0:
			node.first_MC_ = total_return
		node.num_visits_ += 1
		# back until root is reached, the parent of root is None
		while node.parent_act_ is not None:
			parent_act = node.parent_act_
			parent_act.num_visits_ += 1
			# print(total_return)
			# print(self.gamma_)
			total_return *= self.gamma_
			total_return += self.modify_reward(node.reward_)
			# avg = (total+avg0(n-1))/n
			# avg = avg0+(total-avg0)/n
			# incremental method, re-calculate the average reward
			parent_act.avg_return_ += (total_return - parent_act.avg_return_) / parent_act.num_visits_
			node = parent_act.parent_state_
			if node.parent_act_ is None:
				root_sub_return = total_return
			node.num_visits_ += 1
		return root_sub_return

	def MC_sampling_depth(self, node, depth):
		"""
        playing out process, with a limited depth
        """
		mc_return = self.leaf_value_
		self.sim_.set_state(node.state_)
		discount = 1
		final_return = None
		for i in range(0, depth):
			if self.sim_.is_terminal():
				mc_return += self.end_episode_value_
				break
			actions = self.sim_.get_actions()
			act_ID = int(random.random() * len(actions))
			r = self.sim_.act(actions[act_ID])
			# if r > mc_return:
			#     mc_return = discnt * self.modify_reward(r)
			mc_return += discount * self.modify_reward(r)
			if not final_return or (final_return < mc_return):
				final_return = mc_return
			discount *= self.gamma_
		self.sim_.get_state()
		if not final_return:
			final_return = 0
		# return mc_return
		return final_return

	def MC_sampling_terminal(self, node):
		"""
        playing out process, until reach a terminal state
        """
		mc_return = self.end_episode_value_
		self.sim_.set_state(node.state_)
		origin_reward = self.sim_.get_reward()
		origin_graph = node.state_.graph
		reward_list = []
		discount = 1
		final_return = None
		while not self.sim_.is_terminal():
			actions = self.sim_.get_actions()
			act_ID = int(random.random() * len(actions))
			r = self.sim_.act(actions[act_ID])
			# reward_list.append(r)
			# print("diff test:----------------", origin_reward, sum(reward_list), self.sim_.get_reward())
			# if origin_reward + sum(reward_list) != self.sim_.get_reward():
			# 	self.sim_.get_reward()
			# 	print("!!!!", reward_list)
			# 	print(origin_graph)
			# 	print(self.sim_.current.graph)

			# if r > mc_return:
			#     mc_return = discount * self.modify_reward(r)
			mc_return += discount * self.modify_reward(r)
			if not final_return or (final_return < mc_return):
				final_return = mc_return
			discount *= self.gamma_
		self.sim_.get_state()
		# return mc_return
		if not final_return:
			final_return = 0
		# print("diff test-------finish:--------")
		return final_return

	def modify_reward(self, orig):
		return orig

	def print_root_values(self):
		size = len(self.root_.node_vect_)
		for i in range(0, size):
			val = self.root_.node_vect_[i].avg_return_
			num_visit = self.root_.node_vect_[i].avg_return_
			print("(", self.root_.act_vect_.print(), ",", val, ",", num_visit, ") ")
		print(self.root_.is_terminal_)

	# def clear_tree(self):
	#     if self.root_ is not None:
	#         self.prune_state(self.root_)
	#     self.root_ = None

	def update_root_node(self, act, new_state, keep_uct_tree=False):
		"""
        Update the root to be one of its children after taking an environment action.
        :param act: the action taken
        :param new_state: the observed next state (necessary for stochastic transition)
        """
		flag = 0
		for act_ptr in range(len(self.root_.act_vect_)):
			if act.equal(self.root_.act_vect_[act_ptr]):
				flag = 1
				break
		if flag == 0:
			print("can not find the exited action")
			exit(0)
		# act_ptr = self.root_.act_vect_.index(act)

		# if we're going to rebuild the uct tree, we can remove pointer to unreferenced subtrees
		if not keep_uct_tree:
			for i in range(len(self.root_.node_vect_)):
				if i != act_ptr:
					self.root_.node_vect_[i] = None

		action_node = self.root_.node_vect_[act_ptr]

		if action_node.state_vect_ is None:
			depth = 1
			current = self.root_
			self.sim_.set_state(current.state_)
			r = self.sim_.act(action_node)
			next_node = current.node_vect_[act_ptr].add_state_node(self.sim_.get_state(),
			                                                       self.sim_.get_actions(),
			                                                       r, self.sim_.is_terminal())
			if -1 == self.max_depth_:
				mc_return = self.MC_sampling_terminal(next_node)
			else:
				mc_return = self.MC_sampling_depth(next_node, self.max_depth_ - depth)
			current = next_node
			self.update_values(current, mc_return)
			self.root_ = current
			action_node = None
			return
		else:
			for s_node in action_node.state_vect_:
				if s_node.state_.equal(new_state):
					self.root_ = s_node
					action_node = None
					return
		return None

	def clear_tree(self):
		self.root_ = None
		pass

	def terminal_root(self):
		return self.root_.is_terminal_

	def prune(self, _action):
		next_root = None
		size = len(self.root_.node_vect_)
		for i in range(0, size):
			if _action.equal(self.root_.act_vect_[i]):
				assert len(self.root_.node_vect_[i].state_vect_) == 1
				next_root = self.root_.node_vect_[i].state_vect_[0]
				del self.root_.node_vect_[i]
			else:
				tmp = self.root_.node_vect_[i]
				self.prune_action(tmp)

		assert next_root is not None
		self.root_ = next_root
		self.root_.parent_act_ = None

	def prune_state(self, _state):
		size_node = len(_state.node_vect_)
		for i in range(0, size_node):
			tmp = _state.node_vect_[i]
			self.prune_action(tmp)

		_state.node_vect_ = []
		del _state

	def prune_action(self, _action):
		size_node = len(_action.state_vect_)
		for i in range(0, size_node):
			tmp = _action.state_vect_[i]
			self.prune_state(tmp)
		_action.state_vect_ = []
		del _action

	def test_root(self, _state, _reward, _is_terminal):
		return self.root_ is not None \
		       and (self.root_.reward_ == _reward) \
		       and (self.root_.is_terminal_ == _is_terminal) \
		       and self.root_.state_.equal(_state)

	def test_deterministic_property(self):
		if self.test_deterministic_property_state(self.root_):
			print("Deterministic Property Test passed!")
		else:
			print("Error in Deterministic Property  Test!")
			sys.exit(0)

	def test_deterministic_property_state(self, _state):
		act_size = len(_state.node_vect_)
		# we test all the actions under a _state
		for i in range(0, act_size):
			if not self.test_tree_structure_Action(_state.node_vect_[i]):
				return False
		return True

	def test_deterministic_property_action(self, action):
		state_size = len(action.state_vect_)
		# under a deterministic property, a on s can only generate one specific s'
		if state_size != 1:
			print("Error in Deterministic Property Test!")
			return False
		# test every state under an action
		for i in range(0, state_size):
			if not self.test_tree_structure_state(action.state_vect_[i]):
				# print("action test: False")
				return False
		# print("action test:True")
		return True

	def test_tree_structure(self):
		if self.test_tree_structure_state(self.root_):
			print("Tree Structure Test passed!")
		else:
			print("Error in Tree Structure Test!")
			sys.exit(1)

	def test_tree_structure_state(self, _state):
		act_visit_counter = 0
		act_size = len(_state.node_vect_)
		for i in range(0, act_size):
			act_visit_counter += _state.node_vect_[i].num_visits_
		# find out that whether the total number of _state' visit is
		# equal to the number of the visit of their parent action
		if (act_visit_counter + 1 != _state.num_visits_) and (not _state.is_terminal_):
			print("n(s) = sum_{a} n(s,a) + 1 failed ! \n Diff: ",
			      act_visit_counter + 1 - _state.num_visits_,
			      "\nact: ", act_visit_counter + 1, "\nstate: ",
			      _state.num_visits_, "\nTerm: ",
			      _state.is_terminal_, "\nstate: ")
			_state.state_.print()
			print("")
			return False

		for i in range(0, act_size):
			if not self.test_tree_structure_Action(_state.node_vect_[i]):
				return False

		return True

	def test_tree_structure_Action(self, action):
		state_visit_counter = 0
		state_size = len(action.state_vect_)
		for i in range(0, state_size):
			state_visit_counter += action.state_vect_[i].num_visits_

		if state_visit_counter != action.num_visits_:
			print("n(s,a) = sum n(s') failed !")
			return False
		# avg
		# Q(s,a) = E {r(s') + gamma * sum pi(a') Q(s',a')}
		# Q(s,a) = sum_{s'} n(s') / n(s,a) * ( r(s')
		# + gamma * sum_{a'} (n (s',a') * Q(s',a') + first) / n(s'))
		value = 0
		for i in range(0, state_size):
			next_state = action.state_vect_[i]
			w = next_state.num_visits_ / float(action.num_visits_)
			next_value = next_state.first_MC_
			next_act_size = len(next_state.node_vect_)
			for j in range(0, next_act_size):
				next_value += next_state.node_vect_[j].num_visits_ * next_state.node_vect_[j].avg_return_
			next_value = next_value / next_state.num_visits_ * self.gamma_
			next_value += next_state.reward_
			value += w * next_value

		if (action.avg_return_ - value) * (action.avg_return_ - value) > 1e-10:
			print("value constraint failed !",
			      "avgReturn=", action.avg_return_, " value=", value)
			return False

		for i in range(0, state_size):
			if not self.test_tree_structure_state(action.state_vect_[i]):
				return False
		# print("test_tree_structure_Action pass")
		return True
