"""
Version:
UCT with DP and prohibit path
5 components
analytical evaluation

Feature:
updated DP with configurable basic weight for not in approved paths
"""
import datetime
import os

from utils.util import mkdir, get_args, generate_depth_list
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.test_analytic_simulator import anay_read_test
from Algorithms.PlannerTest import Planner_test
from Algorithms.BruteForce import brute_force
from Algorithms.RandomSearch import random_search
from Algorithms.GeneticSearch import genetic_search
from Viz.VIZ import viz_test


def generate_traj_lists(trajectories, test_number):
    traj_lists = []
    for traj in trajectories:
        # TODO for multi size test
        # if traj < 800:
        if traj < 0:
            try_times = int(2 * test_number)
        else:
            try_times = test_number
        traj_list = []
        for _ in range(try_times):
            traj_list.append(traj)
        traj_lists.append(traj_list)
    return traj_lists


def main(name):
    # target_lists = []
    target_vout_list = [-200, 50, 200]
    target_vout_list = [50]

    trajectories = [3]
    test_number = 1
    # trajectories = [600, 500, 400, 300, 200, 100]
    # trajectories = [10,15]
    # test_number = 4
    # trajectories = [10000]
    # test_number = 1
# [[600,600,600,600],[500,500,500,500]]
    traj_lists = generate_traj_lists(trajectories, test_number)

    for try_target_vout in target_vout_list:
        for traj_list in traj_lists:
            # for try_traj in traj_list:
            mkdir("figures")
            mkdir("Results")
            # simulation_root_folder = "simu_analy/" + date_str +'-'+str(os.getpid())
            # mkdir(simulation_root_folder)
            configs = {}
            args_file_name = "config"
            get_args(args_file_name, configs)

            # try_target_vout = 50
            configs["target_vout"] = try_target_vout
            configs["fix_paras"] = None

            configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
            configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
            depth_list = generate_depth_list(configs["dep_start"], configs["dep_end"], configs["dep_step_len"])
            # traj_list = generate_traj_List(configs["traj_start"], configs["traj_end"], configs["traj_step_len"])
            date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if configs["algorithm"] == "UCF" or configs["algorithm"] == "UCT":
                serial_UCF_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "GeneticSearch" or configs["algorithm"] == "GS":
                # print(traj_list)
                # print('!!!!!!!!!!!!!!!!')
                genetic_search(traj_list, configs, date_str)
            elif configs["algorithm"] == "VIZ":
                mkdir("Viz/TreeStructures")
                viz_test(depth_list, traj_list, configs, date_str)
            # elif configs["algorithm"] == "TEST":
            #     hash_read_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "ANA_TEST":
                anay_read_test(configs)
            elif configs["algorithm"] == "TEST":
                Planner_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "RS":
                random_search(traj_list, configs, date_str)
            elif configs["algorithm"] == "BF":
                brute_force(depth_list, traj_list, configs, date_str)

    os.system("rm *-analytic*")
    os.system("rm PCC-*.cki")
    os.system("rm PCC-*.simu")
    return


if __name__ == '__main__':
    # from SimulatorAnalysis.sim_result import conduction_analytics_result
    # conduction_analytics_result('./', 'Rb1')
    # read_result('mutitest_50-2021-04-17-16-29-41-37526.txt')
    main('PyCharm')
