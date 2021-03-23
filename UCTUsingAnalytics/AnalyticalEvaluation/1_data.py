from gen_topo import *


if __name__ == '__main__':

    json_file = json.load(open("./components_data_random/data.json"))

    data_folder='component_data_random'
    target_folder='database'

    circuit_dic={}

    data={}

    for fn in json_file:


        tmp=json_file

        key = key_circuit(fn,tmp)
        
        
        if key in circuit_dic:
            circuit_dic[key].append(fn)
        else:
            circuit_dic[key]=[]
            circuit_dic[key].append(fn)


    count=0;

    json_file = json.load(open("./components_data_random/data.json"))
    
    for key in circuit_dic:
            filename=circuit_dic[key][0]

            list_of_node=json_file[filename]['list_of_node']
            list_of_edge=json_file[filename]['list_of_edge']
            netlist=json_file[filename]['netlist']

            print(netlist)
            
            name='Topo-'+ format(count, '04d')
            topo_file=target_folder+'/topo/'+ name + '.png'

            save_topo(list_of_node, list_of_edge,topo_file)

            count=count+1

            data[name] = {
                            "key": key,
                            "list_of_node":list_of_node,
                            "list_of_edge":list_of_edge,
                            "netlist":netlist
                         }

    
    with open(target_folder+'/data.json', 'w') as outfile:
            json.dump(data, outfile)
    outfile.close()
            
    print(len(data))


                
                    

                
                


