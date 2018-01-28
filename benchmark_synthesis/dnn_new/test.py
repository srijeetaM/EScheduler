from fusion_template import *
from dnn import *

if __name__ == "__main__":

    v = [
        [4096, 4096],
        [4096, 2048],
        [2048, 1024],
        [1024, 512],
        [512, 16],
    ]
	
    total_nodes=len(v)
    for start_node in range(total_nodes): 
        dnn = []
        for i in range(1, total_nodes - start_node + 1):
            uid = ""
            tid = ""
            variables=[]
            for j in range(start_node, start_node+i):
                if j < start_node+i-1:
                    uid += str(j) + "_"
                    tid += str(j) + "-"
                else:
                    uid += str(j)
                    tid += str(j)
                variables.append(v[j][0])
                variables.append(v[j][1])
            dag = 0
            depth = i
            ipbuffsize = 2 * depth + 1
            opbuffsize = 1 * depth
            varsize = 2 * depth
            buf_index = 0
            var_index = 0
            datatype = "float"

            dnn.append(
                DNN(
                    uid,
                    dag,
                    depth,
                    ipbuffsize,
                    opbuffsize,
                    varsize,
                    buf_index,
                    var_index,
                    datatype,
                )
            )
            dnn[-1].get_variables(variables)
            dnn[-1].initialise_kernel_info()
            dnn[-1].generate_fused_source_code()
            src_file = "./DNN/src/dnn_" + uid + ".cl"
            file = open(src_file, "w")
            file.writelines(dnn[-1].source_code)
            file.close()
            tinfoname = "./DNN/tinfo/node_" + str(start_node) + ":" + uid
            print tinfoname
            dimension = 1
            src_file_loc ="/DAG_0/dnn_" + uid + ".cl"
            kernel_name = "dnn_" + uid
            dnn[-1].dump_tinfo(kernel_name, src_file_loc, dimension, tinfoname)

