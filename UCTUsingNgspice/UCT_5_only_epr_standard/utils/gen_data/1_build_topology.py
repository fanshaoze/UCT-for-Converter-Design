# topology construction: 
# 1. select k components
# 2. construct an initial set of join points, consisting of 2 * k points for the left and right join point for the components, and three special join points (VIN, VOUT, GND) [joint point array]
# 3. for each current point p1 in the (2*k + 3) points:
#         select another point p2 from the [joint point array], 
#                       satisfying  (1) it is not a point of the same component or its union set (i.e., the left and right of the same device), 
#                                               (2) if the current point p1 is VIN, VOUT, or GND, then p2 cannot be the VIN, VOUT or GND or its union set, and 
#                                               (3) p1 and p2 cannot be already in the same union set.
#         union the selected point (p1) and the current point (p2) in the join point array
# new valid condition:
#     (1) there is a path from VIN to VOUT, and (2) there is a path from VIN to GND
import collections
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import random
import time
import json
import os
import shutil


def union(x, y, parent):
    f_x = find(x,parent)
    f_y = find(y,parent)

    if f_x == f_y:
        return False

    parent[f_x] = f_y

    return True
    
def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x],parent)

    return parent[x]

def initial(n):
    # number of basic component in topology

    component_pool = ["GND", 'VIN', 'VOUT']

    port_pool = ["GND", 'VIN', 'VOUT']

    basic_compoments = ["FET-A", "FET-B", "capacitor", "inductor"]

    count_map = {"FET-A":0, "FET-B":0, "capacitor":0, "inductor":0}

    comp2port_mapping = {0:[0], 1:[1], 2:[2]} #key is the idx in component pool, value is idx in port pool

    port2comp_mapping = {0:0, 1:1, 2:2}

    index = range(len(basic_compoments))

    port_2_idx = {"GND":0 , 'VIN':1, 'VOUT':2}

    idx_2_port = {0:'GND', 1:'VIN', 2:"VOUT"}

    same_device_mapping = {}

    graph = collections.defaultdict(set)
        

    for i in range(n):

        idx = random.choice(index)

        count = str(count_map[basic_compoments[idx]])

        count_map[basic_compoments[idx]] += 1

        component = basic_compoments[idx] +  '-' + count

        component_pool.append(component)

        idx_component_in_pool = len(component_pool) - 1

        port_pool.append(component + '-left')

        port_pool.append(component + '-right')

        port_2_idx[component + '-left'] = len(port_2_idx)

        port_2_idx[component + '-right'] = len(port_2_idx)

        comp2port_mapping[idx_component_in_pool] = [port_2_idx[component + '-left'], port_2_idx[component + '-right']]

        port2comp_mapping[port_2_idx[component + '-left']] = idx_component_in_pool

        port2comp_mapping[port_2_idx[component + '-right']] = idx_component_in_pool

        idx_2_port[len(idx_2_port)] = component + '-left'

        idx_2_port[len(idx_2_port)] = component + '-right'

        same_device_mapping[port_2_idx[component + '-left']] = port_2_idx[component + '-right']

        same_device_mapping[port_2_idx[component + '-right']] = port_2_idx[component + '-left']

        # parent = [-1] * len(port_pool)
    parent = list(range(len(port_pool)))


    # print("port_2_idx",port_2_idx)
    # print("idx_2_port",idx_2_port) 
    # print("port_pool",port_pool)
    # print("component_pool",component_pool)
    # print("same_device_mapping",same_device_mapping)
    # print("comp2port_mapping",comp2port_mapping)
    # print("port2comp_mapping",port2comp_mapping)

    return component_pool, port_pool, count_map, comp2port_mapping,port2comp_mapping, port_2_idx, idx_2_port, same_device_mapping, graph, parent


def convert_to_netlist(graph, component_pool, port_pool, parent, comp2port_mapping):
    #for one component, find the two port
    # if one port is GND/VIN/VOUT, leave it don't need to find the root
    # if one port is normal port, then find the root, if the port equal to root then leave it, if the port is not same as root, change the port to root
    list_of_node = set()

    list_of_edge = set()

    list_of_node_2 = set()

    list_of_edge_2 = set()

    # netlist = []
    for idx ,comp in enumerate(component_pool):
        # cur = []
        # cur.append(comp)
        list_of_node.add(comp)
        for port in comp2port_mapping[idx]:
            port_joint_set_root = find(port,parent)
            # cur.append(port_pool[port_joint_set_root])
            if port_joint_set_root in [0,1,2]:
                list_of_node.add(port_pool[port_joint_set_root])
                # list_of_edge.add((comp, port_pool[port_root]))
                list_of_node.add(port_joint_set_root)
                list_of_edge.add((comp, port_joint_set_root))
                list_of_edge.add((port_pool[port_joint_set_root], port_joint_set_root))
            else:
                list_of_node.add(port_joint_set_root)
                list_of_edge.add((comp, port_joint_set_root))


    for idx ,comp in enumerate(component_pool):
        # cur = []
        # cur.append(comp)
     ###print("\nlist of node and edges")
     ###print("idx and comp:", idx, comp)
        list_of_node_2.add(comp)
     ###print("list of node:", list_of_node_2)
        odd_even=1
        for port in comp2port_mapping[idx]:
            port_joint_set_root = find(port,parent)
         ###print("port_joint_set_root:", port_joint_set_root)
            # cur.append(port_pool[port_joint_set_root])
            if port_joint_set_root in [0,1,2]:
                list_of_node_2.add(port_pool[port_joint_set_root])
                # list_of_edge.add((comp, port_pool[port_root]))
                list_of_node_2.add(port_joint_set_root)
                list_of_edge_2.add((comp, port_joint_set_root))
                list_of_edge_2.add((port_pool[port_joint_set_root], port_joint_set_root))
            else:
                list_of_node_2.add(port_joint_set_root)
                if comp not in ["VIN","VOUT","GND"]:
                  if odd_even:
                      list_of_edge_2.add((comp+"-left", port_joint_set_root))
                      list_of_node_2.add(comp+"-left")
                      odd_even=0
                  else:
                      list_of_edge_2.add((comp+"-right", port_joint_set_root))
                      list_of_node_2.add(comp+"-right")
                      odd_even=1
                else:
                     list_of_edge_2.add((comp, port_joint_set_root))

            if comp not in ["VIN","VOUT","GND"]:
                       list_of_edge_2.add((comp+"-left", comp))
                       list_of_edge_2.add((comp+"-right", comp))
         ###print("after add node and edge:", list_of_node_2, list_of_edge_2)


    netlist = []
    joint_list = set()

    for idx ,comp in enumerate(component_pool):
     ###print("\nidx and comp",idx, comp)
        if comp in ['VIN', 'VOUT', 'GND']:
                continue
        cur = []

        cur.append(comp)
        for port in comp2port_mapping[idx]:
         ###print("parent:", parent)
         ###print("port and comp2port_map:",port,comp2port_mapping[idx])
            # print(port_joint_set_root)
            port_joint_set_root = find(port,parent)
         ###print("port_joint_set_root:",port_joint_set_root)
            root_0 = find(0, parent)
            root_1 = find(1, parent)
            root_2 = find(2, parent)
            if port_joint_set_root == root_0:
                cur.append("0")
            elif port_joint_set_root == root_1:
                cur.append("IN")

            elif port_joint_set_root == root_2:
                cur.append("OUT")
                    # cur.append(port_pool[port_joint_set_root])
            # else:
            else:
                joint_list.add(str(port_joint_set_root))
                cur.append(str(port_joint_set_root))
         ###print("cur:", cur)

        netlist.append(cur)
     ###print("netlist:", netlist)

 ###print(joint_list)

    return list(list_of_node), list(list_of_edge), netlist, list(joint_list),list(list_of_node_2), list(list_of_edge_2)



def convert_graph(graph, comp2port_mapping, port2comp_mapping, idx_2_port, parent, component_pool,same_device_mapping, port_pool):
    list_of_node = set()

    list_of_edge = set()

    has_short_cut = False

    for node in comp2port_mapping:
        if len(comp2port_mapping[node]) == 2:
            list_of_node.add(comp2port_mapping[node][0])
            list_of_node.add(comp2port_mapping[node][1])
            list_of_edge.add((comp2port_mapping[node][1], comp2port_mapping[node][0]))

    for node in graph:
        root_node = find(node, parent)
        list_of_node.add(node)
        list_of_node.add(root_node)
        
        # cur_node_the_other_port_root = find(cur_node_the_other_port, parent)

        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                has_short_cut = True

        if root_node != node:
            list_of_edge.add((node, root_node))

        for nei in graph[node]:
            list_of_node.add(nei)
            if nei != root_node:
                list_of_edge.add((nei, root_node))

                    
    return list(list_of_node), list(list_of_edge), has_short_cut

def save_ngspice_file(netlist, path, args):

    path += '.cki'
    file = open(path, 'w')
    print(path)

 ###print("netlist:",netlist)

    capacitor_paras = args.cap_paras
    inductor_paras = args.ind_paras
    
    
    selected_cap = random.choice(capacitor_paras)
    selected_ind = random.choice(inductor_paras)


   ###############################
   ##### Reform device ###########
   ###############################

   # device parameter in data.json = param_list #
    param_list={};
    param_list_2={}
   # .PARAM string = param_str #
    param_str=" ";
   # circuit string = cki_str #
    cki_str="";
    Kcap=0;
    Kind=0;
    net_extra=100;
    L_param=8

    empty_param=[0]*L_param
    nodes_param=empty_param[:]
    nodes_param[L_param-1]=1

    param_list["VIN"]=empty_param[:]
    param_list["VIN"][0]=1;
    param_list["VOUT"]=empty_param[:]
    param_list["VOUT"][1]=1;
    param_list["GND"]=empty_param[:]
    param_list["GND"][2]=1;

    param_list_2["VIN"]=empty_param[:]
    param_list_2["VIN"][0]=1;
    param_list_2["VOUT"]=empty_param[:]
    param_list_2["VOUT"][1]=1;
    param_list_2["GND"]=empty_param[:]
    param_list_2["GND"][2]=1;



    print(netlist)

    for x in netlist:
        tmp_name=x[0]
        x[0] = x[0].replace("-","")
        if "capacitor" in x[0]:
            selected_cap = random.choice(capacitor_paras)
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][3]=selected_cap/200
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][3]=selected_cap/200
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param 
            param_str=param_str+"C"+str(Kcap)+"=%su " % selected_cap
            cki_str=cki_str+x[0]+" "+x[1]+" "+x[2]+" {C"+str(Kcap)+"}\n"
            cki_str=cki_str+"rcap"+str(Kcap)+" "+x[1]+" "+x[2]+" 1Meg\n"
            Kcap=Kcap+1;
        elif "inductor" in x[0]:
            selected_ind = random.choice(inductor_paras)
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][4]=selected_ind/200
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][4]=selected_ind/200
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param 
            param_str=param_str+"L"+str(Kind)+"=%su " % selected_ind
            cki_str=cki_str+"L"+x[0]+" "+x[1]+" "+str(net_extra)+" {L"+str(Kind)+"}\n"
            cki_str=cki_str+"rind"+str(Kind)+" "+str(net_extra)+" "+x[2]+" 0.001\n"
            Kind=Kind+1;
            net_extra=net_extra+1;
        elif "FETA" in x[0]:#nmos
            selected_wn = 10000
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][5]=selected_wn/100000
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][5]=selected_wn/100000
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param 
            cki_str=cki_str+"M"+x[0]+" "+x[1]+" "+"gaten"+" "+x[2]+" 0 MOSN L=1U W=%sU\n" % selected_wn
        elif "FETB" in x[0]:
            selected_wp = 25000
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][6]=selected_wp/100000
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][6]=selected_wp/100000
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param 
            cki_str=cki_str+"M"+x[0]+" "+x[1]+" "+"gatep"+" "+x[2]+" IN MOSP L=1U W=%sU\n" % selected_wp
        elif "Diode" in x[0]:
            x.append("DMOD")
 
         


    prefix = [      
                ".title buck.cki",
                ".model MOSN NMOS level=8 version=3.3.0", 
                ".model MOSP PMOS level=8 version=3.3.0",
                ".model DMOD D (bv=200 is=1e-13 n=1.05)",
                ".PARAM " + args.ngspice_para + param_str,
                "",
                ".SUBCKT DC_source 1",
                "VIN 2 0 {vin}",
                "ROUT 2 1 {rvinout}",
                ".ENDS",
                "",           
                "\n",
                "*input*",
                "Vclock1 gaten 0 PULSE (0 {vin}  {abs(2*D-1)/2/freq} 1n 1n {D/freq} {1/freq})",
                "Vclock2 gatep 0 PULSE (0 {vin} 0 1n 1n {(1-D)/freq} {1/freq})",
                #"Rgate_n gaten_r gaten 0.001",
                #"Rgate_p gatep_r gatep 0.001",
                "XVIN IN_ext DC_source",
                "RINsense IN_ext IN 0.001",
                "ROUTsense OUT OUT_ext 0.001",
                "\n"]

    sufix = [   "\n",
                "COUT OUT 0 10u",
                "RLOAD OUT_ext 0 {rload}",
                ".save all",
                #".save i(vind)",
                ".control",
                "tran 0.1u 4000u",
                "print V(OUT)",
                #".print tran V(IN_ext,IN)",
                #".print tran V(OUT,OUT_ext)",
                "print V(IN_ext,IN)",
                "print V(OUT,OUT_ext)",
                ".endc",
                ".end",
                ]

    file.write("\n".join(prefix) + '\n')
    file.write("*topology*" + '\n')

    file.write(cki_str)
    file.write("\n".join(sufix))
    file.close()
    return param_list, param_list_2


def save_ngspice_file_with_parameters(netlist, path, parameters):

    path += '.cki'
    file = open(path, 'w')
    print(path)
   ###############################
   ##### Reform device ###########
   ###############################

   # device parameter in data.json = param_list #
    param_list={};
    param_list_2={}
   # .PARAM string = param_str #
    param_str=" ";
   # circuit string = cki_str #
    cki_str="";
    Kcap=0;
    Kind=0;
    net_extra=100;
    L_param=8

    empty_param=[0]*L_param
    nodes_param=empty_param[:]
    nodes_param[L_param-1]=1

    param_list["VIN"]=empty_param[:]
    param_list["VIN"][0]=1;
    param_list["VOUT"]=empty_param[:]
    param_list["VOUT"][1]=1;
    param_list["GND"]=empty_param[:]
    param_list["GND"][2]=1;

    param_list_2["VIN"]=empty_param[:]
    param_list_2["VIN"][0]=1;
    param_list_2["VOUT"]=empty_param[:]
    param_list_2["VOUT"][1]=1;
    param_list_2["GND"]=empty_param[:]
    param_list_2["GND"][2]=1;

    print(netlist)

    for x in netlist:
        tmp_name=x[0]
        x[0] = x[0].replace("-","")
        if "capacitor" in x[0]:
            selected_cap = parameters[tmp_name][3]*200
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][3]=selected_cap/200
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][3]=selected_cap/200
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param
            param_str=param_str+"C"+str(Kcap)+"=%su " % selected_cap
            cki_str=cki_str+x[0]+" "+x[1]+" "+x[2]+" {C"+str(Kcap)+"}\n"
            cki_str=cki_str+"rcap"+str(Kcap)+" "+x[1]+" "+x[2]+" 1Meg\n"
            Kcap=Kcap+1;
        elif "inductor" in x[0]:
            selected_ind = parameters[tmp_name][4]*200
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][4]=selected_ind/200
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][4]=selected_ind/200
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param
            param_str=param_str+"L"+str(Kind)+"=%su " % selected_ind
            cki_str=cki_str+"L"+x[0]+" "+x[1]+" "+str(net_extra)+" {L"+str(Kind)+"}\n"
            cki_str=cki_str+"rind"+str(Kind)+" "+str(net_extra)+" "+x[2]+" 0.001\n"
            Kind=Kind+1;
            net_extra=net_extra+1;
        elif "FETA" in x[0]:#nmos
            selected_wn = parameters[tmp_name][5]*100000
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][5]=selected_wn/100000
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][5]=selected_wn/100000
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param
            cki_str=cki_str+"M"+x[0]+" "+x[1]+" "+"gaten"+" "+x[2]+" 0 MOSN L=1U W=%sU\n" % selected_wn
        elif "FETB" in x[0]:
            selected_wp = parameters[tmp_name][6]*100000
            param_list[tmp_name]=empty_param[:]
            param_list[tmp_name][6]=selected_wp/100000
            param_list_2[tmp_name]=empty_param[:]
            param_list_2[tmp_name][6]=selected_wp/100000
            param_list_2[tmp_name+"-left"]=nodes_param
            param_list_2[tmp_name+"-right"]=nodes_param
            cki_str=cki_str+"M"+x[0]+" "+x[1]+" "+"gatep"+" "+x[2]+" IN MOSP L=1U W=%sU\n" % selected_wp
        elif "Diode" in x[0]:
            x.append("DMOD")




    prefix = [
                ".title buck.cki",
                ".model MOSN NMOS level=8 version=3.3.0",
                ".model MOSP PMOS level=8 version=3.3.0",
                ".model DMOD D (bv=200 is=1e-13 n=1.05)",
                ".PARAM " + args.ngspice_para + param_str,
                "",
                ".SUBCKT DC_source 1",
                "VIN 2 0 {vin}",
                "ROUT 2 1 {rvinout}",
                ".ENDS",
                "",
                "\n",
                "*input*",
                "Vclock1 gaten 0 PULSE (0 {vin}  {abs(2*D-1)/2/freq} 1n 1n {D/freq} {1/freq})",
                "Vclock2 gatep 0 PULSE (0 {vin} 0 1n 1n {(1-D)/freq} {1/freq})",
                #"Rgate_n gaten_r gaten 0.001",
                #"Rgate_p gatep_r gatep 0.001",
                "XVIN IN_ext DC_source",
                "RINsense IN_ext IN 0.001",
                "ROUTsense OUT OUT_ext 0.001",
                "\n"]

    sufix = [   "\n",
                "COUT OUT 0 10u",
                "RLOAD OUT_ext 0 {rload}",
                ".save all",
                #".save i(vind)",
                ".control",
                "tran 0.1u 4000u",
                "print V(OUT)",
                #".print tran V(IN_ext,IN)",
                #".print tran V(OUT,OUT_ext)",
                "print V(IN_ext,IN)",
                "print V(OUT,OUT_ext)",
                ".endc",
                ".end",
                ]

    file.write("\n".join(prefix) + '\n')
    file.write("*topology*" + '\n')

    file.write(cki_str)
    file.write("\n".join(sufix))
    file.close()
    return param_list, param_list_2


# 


def nets_to_ngspice_files(topologies, configs, n_components=4,
                          output=False, output_folder="component_data_random"):
    ngspice_para = "freq=%s vin=%s rload=50 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=%s" % \
                   (configs["freq"], configs["vin"], configs["D"])
    sys_os = configs['sys_os']

    directory_path = str(n_components) + output_folder

    if sys_os == "windows":
        # windows command delete prior analysis
        os.system("del /Q %s\*" % directory_path)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    for i in range(0, len(topologies)):
        port_2_idx = topologies[i].port_2_idx
        idx_2_port = topologies[i].idx_2_port
        component_pool = topologies[i].component_pool
        same_device_mapping = topologies[i].same_device_mapping
        port_pool = topologies[i].port_pool
        component_pool = topologies[i].component_pool
        parent = topologies[i].parent
        comp2port_mapping = topologies[i].comp2port_mapping
        port2comp_mapping = {}
        for key in comp2port_mapping.keys():
            for v in comp2port_mapping[key]:
                port2comp_mapping[v] = key
        data = {}
        list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(None, component_pool,
                                                                             port_pool, parent, comp2port_mapping)

        T = nx.Graph()
        T.add_nodes_from(list_of_node)
        T.add_edges_from(list_of_edge)

        name = "PCC-" + str(os.getpid())+"-" + format(i, '06d')
        file_path = directory_path + '/' + name
        T.clear()

        save_ngspice_file_with_parameters(netlist, file_path, topologies[i].parameters)
        data[name] = {
                        "port_2_idx":port_2_idx,
                        "idx_2_port":idx_2_port,
                        "port_pool":port_pool,
                        "component_pool":component_pool,
                        "same_device_mapping":same_device_mapping,
                        "comp2port_mapping":comp2port_mapping,
                        "port2comp_mapping":port2comp_mapping,
                        "list_of_node":list_of_node,
                        "list_of_edge":list_of_edge,
                        "netlist":netlist,
                        "joint_list":joint_list,
                        "parameter_list":param_list
                     }

        data_2[name] = {
                        "port_2_idx":port_2_idx,
                        "idx_2_port":idx_2_port,
                        "port_pool":port_pool,
                        "component_pool":component_pool,
                        "same_device_mapping":same_device_mapping,
                        "comp2port_mapping":comp2port_mapping,
                        "port2comp_mapping":port2comp_mapping,
                        "list_of_node":list_of_node_2,
                        "list_of_edge":list_of_edge_2,
                        "netlist":netlist,
                        "joint_list":joint_list,
                        "parameter_list":param_list_2
                }
        with open(directory_path+'/data.json', 'w') as outfile:
                json.dump(data, outfile)
        outfile.close()
        with open(directory_path+'/data_2.json', 'w') as outfile:
                json.dump(data_2, outfile)
        outfile.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_components', type=int, default=4, help='specify the number of component')
    parser.add_argument('-n_topology', type=int, default=5,
                        help='specify the number of topology you want to generate')
    parser.add_argument('-output_folder', type=str, default="components_data_random",
                        help='specify the output folder path')
    parser.add_argument('-ngspice_para', required=True,type=str,
                        help='specify the parameters in ngspice like: -ngpisce_para="freq=200k vin=5 rload=10"')
    parser.add_argument('-cap_paras',required=True,type=str,
                        help='specify the parameters list for capacitor like: -cap_paras=10,20')
    parser.add_argument('-ind_paras', required=True, type=str,
                        help='specify the parameters list for inductor like -ind_paras=10,20')
    parser.add_argument('-os', required=True, type=str,
                        help='operating system')



    args = parser.parse_args()
    args.cap_paras = [float(x) for x in args.cap_paras.split(',')]
    args.ind_paras = [float(x) for x in args.ind_paras.split(',')]
#    directory_path = str(args.n_components) + args.output_folder
    directory_path = args.output_folder


    if (args.os=="windows"):
       #windows command delete prior analysis
       os.system("del /Q %s\*" % directory_path)

    if os.path.exists(directory_path):
        os.system("rm -r "+directory_path)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    n_components = args.n_components #excluding from VIN VOUT GND
    
    k = args.n_topology # number of topology will be generated

    num_topology = 0

    # from datetime import timedelta
    start = time.time()

    data = {}

    data_2 = {}

    while k > 0:
            # print("graph",k)
        component_pool, port_pool, count_map, comp2port_mapping,port2comp_mapping, port_2_idx, idx_2_port, same_device_mapping, graph, parent = initial(n_components)#key is the idx in component pool, value is idx in port pool

        
        
        p1_pool = list(range(len(port_pool)))

        # random.shuffle(p1_pool)

        for cur_point in p1_pool:
                
            p2_pool = list(range(len(port_pool)))

            random.shuffle(p2_pool)
            
            if len(graph[cur_point]) > 0 and random.uniform(0, 1) > 0.2:
                # print("not select")
                continue
            for point_2_connect in p2_pool:
                    

                if cur_point == point_2_connect:
                    continue

                if point_2_connect in same_device_mapping and cur_point == same_device_mapping[point_2_connect]:
                    continue# check ports don't come from same component

                if (cur_point in graph and point_2_connect in graph[cur_point]) or ( point_2_connect in graph and cur_point in graph[point_2_connect]):
                    break# check 2 port are already connected

                # if point_2_connect in graph and (cur_point == 0 or cur_point == 1 or cur_point == 2) and (0 in graph[point_2_connect] or 1 in graph[point_2_connect] or 2 in graph[point_2_connect]):
                #       continue# check 2 port are vout-vin, vin-gnd, ....

                # if cur_point in graph and (point_2_connect == 0 or point_2_connect == 1 or point_2_connect == 2) and (0 in graph[cur_point] or 1 in graph[cur_point] or 2 in graph[cur_point]):
                #       continue# check 2 port are vout-vin, vin-gnd, ....

                root_cur = find(cur_point, parent)
                root_next = find(point_2_connect, parent)
                root0 = find(0, parent)
                root1 = find(1, parent)
                root2 = find(2, parent)

                if root_cur == root_next:
                    continue

                #check short cut
                # if cur_point in same_device_mapping:
                #       cur_point_the_other_port = same_device_mapping[cur_point]
                #       cur_point_the_other_port_root = find(cur_point_the_other_port, parent)

                #       if cur_point_the_other_port_root == root_cur or cur_point_the_other_port_root == root_next:
                #               print("there is short cut in cur point")
                #               continue

                # if point_2_connect in same_device_mapping:
                #       point_2_connect_the_other_port = same_device_mapping[point_2_connect]
                #       point_2_connect_the_other_port_root = find(point_2_connect_the_other_port, parent)

                #       if point_2_connect_the_other_port_root == root_cur or point_2_connect_the_other_port_root == root_next:
                #               print("there is short cut in point_2_connect")
                #               continue
                #root_cur, root_next, root0, root1, root2, 
                
                # if (cur_point == 0 or cur_point == 1 or cur_point == 2):
                #       if root0 == root_next or root1 == root_next or root2 == root_next:
                #               continue # check if cur port is 0,1,2, the other port is joint set with 0,1,2

                if root_cur == root0 or root_cur == root1 or root_cur == root2:
                    if root0 == root_next or root1 == root_next or root2 == root_next:
                        # print("0,1,2 should not be in the same joint set")
                        continue# check if cur port is joint set with 0, 1, 2, the other port is also joint set with 0, 1, 2


                #
                # if (point_2_connect == 0 or point_2_connect == 1 or point_2_connect == 2):
                #       if root0 == root_cur or root1 == root_cur or root2 == root_cur:
                #               continue

                # if root_next == root0 or root_next == root1 or root_next == root2:
                #       if root0 == root_cur or root1 == root_cur or root2 == root_cur:
                #               continue
                # if root_next == root_next:
                #       continue

                graph[cur_point].add(point_2_connect)
                graph[point_2_connect].add(cur_point)
                union(cur_point, point_2_connect, parent)
                # print("found valid")
                break

        # if check_valid(graph):
        if 1 not in graph or 2 not in graph or 0 not in graph:
                # print("there is no 0,1,2")
            continue

        # if len(graph) != 11:
        #       print("graph is not fully connected")
        #       continue

        # for x, y in [(3,4), (5,6), (7,8), (9,10)]:
        #       root_x = find(x, parent)
        #       root_y = find(y, parent)

        #       if root_x == root_y:
        #               continue
        list_of_node, list_of_edge, has_short_cut = convert_graph(graph, comp2port_mapping,port2comp_mapping, idx_2_port, parent, component_pool, same_device_mapping,port_pool)

        G=nx.Graph()
        G.add_nodes_from((list_of_node))
        G.add_edges_from(list_of_edge)


        # draw_graph(G, "topology_"+str(k))
        # draw_graph(G, "topology_"+str(k))
        # if has_short_cut:
        #       print("has_short_cut")
        # print("here")
        if nx.is_connected(G) and not has_short_cut:
            # print(graph)
            # print(parent)
            # draw_graph(G, "topology_new_"+str(k))
           
            list_of_node, list_of_edge, netlist, joint_list, list_of_node_2, list_of_edge_2 = convert_to_netlist(graph, component_pool, port_pool, parent, comp2port_mapping)


            G.clear()
            # print(netlist, joint_list)
            T=nx.Graph()
            T.add_nodes_from((list_of_node_2))
            T.add_edges_from(list_of_edge_2)
            # plt.figure(1)
            # nx.draw(G, with_labels=True)
            plt.figure()
            nx.draw(T, with_labels=True)

            name = "PCC-" + format(num_topology, '06d') + "-2"
            file_path = directory_path + '/' + name
            # print(graph_name)
            plt.savefig(file_path) # save as png
            # plt.show()
            T.clear()
            plt.close()


            G.clear()
            # print(netlist, joint_list)
            T=nx.Graph()
            T.add_nodes_from((list_of_node))
            T.add_edges_from(list_of_edge)
            # plt.figure(1)
            # nx.draw(G, with_labels=True)
            plt.figure()
            nx.draw(T, with_labels=True)

            name = "PCC-" + format(num_topology, '06d')
            file_path = directory_path + '/' + name
            # print(graph_name)
            plt.savefig(file_path) # save as png
            # plt.show()
            T.clear()
            plt.close()


            k -= 1

            param_list, param_list_2=save_ngspice_file(netlist, file_path, args)
 
            data[name] = {
                            "port_2_idx":port_2_idx,
                            "idx_2_port":idx_2_port,
                            "port_pool":port_pool,
                            "component_pool":component_pool,
                            "same_device_mapping":same_device_mapping,
                            "comp2port_mapping":comp2port_mapping,
                            "port2comp_mapping":port2comp_mapping,
                            "list_of_node":list_of_node,
                            "list_of_edge":list_of_edge,
                            "netlist":netlist,
                            "joint_list":joint_list,
                            "parameter_list":param_list
                         }
           
            data_2[name] = {
                            "port_2_idx":port_2_idx,
                            "idx_2_port":idx_2_port,
                            "port_pool":port_pool,
                            "component_pool":component_pool,
                            "same_device_mapping":same_device_mapping,
                            "comp2port_mapping":comp2port_mapping,
                            "port2comp_mapping":port2comp_mapping,
                            "list_of_node":list_of_node_2,
                            "list_of_edge":list_of_edge_2,
                            "netlist":netlist,
                            "joint_list":joint_list,
                            "parameter_list":param_list_2 
                    }

            num_topology += 1



    elapsed_time_secs = time.time() - start

    msg = 'AVG generation took in secs: ', (elapsed_time_secs*0.1/num_topology)

    print(msg) 
    print()

    with open(directory_path+'/data.json', 'w') as outfile:
            json.dump(data, outfile)

    outfile.close()

    with open(directory_path+'/data_2.json', 'w') as outfile:
            json.dump(data_2, outfile)

    outfile.close()













