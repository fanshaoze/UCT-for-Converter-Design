import datetime
from utils.util import mkdir, get_args, init_position, generate_depth_list, generate_traj_List
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.GeneticSearch import genetic_search
from Algorithms.test_analytic_simulator import anay_read_test
from Viz.VIZ import viz_test


def generate_traj_lists(trajectories, test_number):
    traj_lists = []
    for traj in trajectories:
        traj_list = []
        for _ in range(test_number):
            traj_list.append(traj)
        traj_lists.append(traj_list)
    return traj_lists


def main(name):
    # target_lists = []
    target_vout_list = [-200, -50, 50, 200]
    target_vout_list = [50]

    trajectories = [128]
    test_number = 1

    traj_lists = generate_traj_lists(trajectories, test_number)

    for try_target_vout in target_vout_list:
        for traj_list in traj_lists:
            mkdir("figures")
            mkdir("Results")
            configs = {}
            args_file_name = "config"
            get_args(args_file_name, configs)

            # try_target_vout = 50
            configs["target_vout"] = try_target_vout

            configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
            configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
            depth_list = generate_depth_list(configs["dep_start"], configs["dep_end"], configs["dep_step_len"])
            # traj_list = generate_traj_List(configs["traj_start"], configs["traj_end"], configs["traj_step_len"])
            date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if configs["algorithm"] == "UCF":
                serial_UCF_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "GeneticSearch" or configs["algorithm"] == "GS":
                genetic_search(configs, date_str)
            elif configs["algorithm"] == "VIZ":
                mkdir("Viz/TreeStructures")
                viz_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "ANA_TEST":
                anay_read_test(configs)

    return


if __name__ == '__main__':
    main('PyCharm')
