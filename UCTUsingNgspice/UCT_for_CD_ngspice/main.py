import datetime
from utils.util import mkdir,get_args, init_position, generate_depth_list, generate_traj_List
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.ParallelUCF import parallel_UCF_test
from Algorithms.GeneticSearch import genetic_search
from Viz.VIZ import viz_test


def main(name):
    beta_list = [0.01, 0.005, 0.001]
    beta_list = [0.5]
    rave_k_list = [0.45]
    for rave_k in rave_k_list:
        traj_lists = []
        # traj_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        #              1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
        # traj_lists.append(traj_list)
        # traj_list = [768, 768, 768, 768, 768, 768, 768, 768, 768, 768,
        #              768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
        # traj_lists.append(traj_list)
        traj_list = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                     512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                     512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                     512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        traj_lists.append(traj_list)
        traj_list = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                     400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                     400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                     400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
        traj_lists.append(traj_list)
        traj_list = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                     256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                     256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                     256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        traj_lists.append(traj_list)
        traj_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                     200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                     200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                     200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        traj_lists.append(traj_list)
        traj_list = [160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
                     160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
                     160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
                     160, 160, 160, 160, 160, 160, 160, 160, 160, 160]
        traj_lists.append(traj_list)

        traj_list = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
        traj_lists.append(traj_list)
        traj_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                     100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                     100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                     100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        traj_lists.append(traj_list)

        # traj_list = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        #              60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
        # traj_lists.append(traj_list)
        # traj_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        #              50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        # traj_lists.append(traj_list)
        # traj_list = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        #              30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        # traj_lists.append(traj_list)

        # traj_list = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        #              60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        #              60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
        # traj_lists.append(traj_list)
        # traj_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        #              100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        #              100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        # traj_lists.append(traj_list)


        # traj_list = [200,200,200,200,200,200,200,200,200,200]
        # traj_lists.append(traj_list)
        # traj_list = [300,300,300,300,300,300,300,300,300,300]
        # traj_lists.append(traj_list)

        # traj_lists = [[256]]

        for traj_list in traj_lists:
            mkdir("figures")
            mkdir("Results")
            configs = {}
            args_file_name = "config"
            get_args(args_file_name, configs)
            configs["rave_k"] = rave_k
            depth_list = generate_depth_list(configs["dep_start"], configs["dep_end"], configs["dep_step_len"])

            date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if configs["algorithm"] == "UCF":
                if configs["root_parallel"] is False:
                    serial_UCF_test(depth_list, traj_list, configs, date_str)
                else:
                    parallel_UCF_test(depth_list, traj_list, configs, date_str)
            elif configs["algorithm"] == "GeneticSearch" or configs["algorithm"] == "GS":
                genetic_search(configs, date_str)
            elif configs["algorithm"] == "VIZ":
                mkdir("Viz/TreeStructures")
                viz_test(depth_list, traj_list, configs, date_str)

    return


if __name__ == '__main__':
    main('PyCharm')
