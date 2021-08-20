"""
Version:
UCT with DP and prohibit path
5 components
analytical evaluation

Feature:
updated DP with configurable basic weight for not in approved paths
"""
import csv
import datetime
import os

from utils.util import mkdir, generate_depth_list, remove_tmp_files, save_results_to_csv
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.GeneticSearch import genetic_search
from Algorithms.BruteForce import brute_force
from Algorithms.RandomSampling import random_sampling, read_result
from config import configs


def generate_traj_lists(trajectories, test_number):
    traj_lists = []
    for traj in trajectories:
        try_times = test_number
        traj_list = []
        for _ in range(try_times):
            traj_list.append(traj)
        traj_lists.append(traj_list)
    return traj_lists


def main(name):
    # candidate target vouts: -200, 50, 200
    # TODO restruct may not be 18 total steps
    # trajectories = [50,30,15]

    traj_lists = generate_traj_lists(configs['trajectories'], configs['test_number'])
    anal_output_results = [[] for i in range(configs['test_number'] + 1)]
    simu_output_results = [[] for i in range(configs['test_number'] + 1)]
    print(traj_lists)

    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pid_str = str(os.getpid())
    result_folder = date_str + '_' + pid_str
    for traj in configs['trajectories']:
        mkdir("figures")
        mkdir("Results")

        configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
        configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])

        if configs["algorithm"] == "UCF" or configs["algorithm"] == "UCT":
            anal_results, simu_results = serial_UCF_test(traj, configs['test_number'], configs, result_folder)
            for i in range(configs['test_number'] + 1):
                anal_output_results[i].extend(anal_results[i])
                simu_output_results[i].extend(simu_results[i])
        else:
            print('alg does not exist')

    anal_out_file_name = "Results/BF-" + str(configs["target_vout"]) + "-anal-" + date_str + "-" + str(
        os.getpid()) + ".csv"
    save_results_to_csv(anal_out_file_name, anal_output_results)
    simu_out_file_name = "Results/BF-" + str(configs["target_vout"]) + "-simu-" + date_str + "-" + str(
        os.getpid()) + ".csv"
    save_results_to_csv(simu_out_file_name, simu_output_results)
    remove_tmp_files()

    return


if __name__ == '__main__':
    main('PyCharm')
