from fusion_template import *
from cnn import *

if __name__ == "__main__":
    configuration = []
    configuration.append(('C',((3,16,16),(32,3,3,3),(32),(32,16,16))))
    configuration.append(('P',((32,16,16),(32,8,8))))
    configuration.append(('C',((32,8,8),(64,3,3,32),(64),(64,8,8))))
    configuration.append(('P',((64,8,8),(64,4,4))))
    configuration.append(('L',((1024),(1024,512),(512))))
    configuration.append(('L',((512),(512,32),(32))))
    # configuration.append(('C',((3,32,32),(16,3,3,3),(16),(16,32,32))))
    # configuration.append(('P',((16,32,32),(16,16,16))))
    # configuration.append(('C',((16,16,16),(32,3,3,16),(32),(32,16,16))))
    # configuration.append(('P',((32,16,16),(32,8,8))))
    # configuration.append(('C',((32,8,8),(64,3,3,32),(64),(64,8,8))))
    # configuration.append(('P',((64,8,8),(64,4,4))))
    # configuration.append(('L',((1024),(1024,512),(512))))
    # configuration.append(('L',((512),(512,128),(128))))
    # configuration.append(('L',((128),(128,16),(16))))
    # configuration.append(('S',((16),(16))))

    total_nodes=len(configuration)
    for start_node in range(total_nodes): 
        cnn = []
        for i in range(1, total_nodes - start_node + 1):
            uid = ""
            tid = ""
            # variables=[]
            cfg=[]
            for j in range(start_node, start_node+i):
                if j < start_node+i-1:
                    uid += str(j) + "_"
                    tid += str(j) + "_"
                else:
                    uid += str(j)
                    tid += str(j)
                cfg.append(configuration[j])
                # variables.append(v[j][1])
            # print cfg
            dag = 1
            depth = i
            ipbuffsize = 0
            opbuffsize = 0

            # ipbuffsize = 2 * depth + 1
            # opbuffsize = 1 * depth
            varsize = 2 * depth
            buf_index = 0
            var_index = 0
            datatype = "float"
            cnn.append(
                CNN(
                    uid,
                    dag,
                    start_node,
                    depth,
                    configuration,
                    buf_index,
                    var_index,
                    datatype,
                )
            )
            
            # dnn[-1].get_variables(variables)
            cnn[-1].initialise_kernel_info()
            cnn[-1].generate_fused_source_code()
            src_file = "./CNN/src/cnn_" + uid + ".cl"
            file = open(src_file, "w")
            file.writelines(cnn[-1].source_code)
            file.close()
            tinfoname = "./CNN/tinfo/node_" + str(start_node) + ":" + tid
            print tinfoname
            dimension = 1
            src_file_loc = "/DAG_1/" + "cnn_" + uid +".cl"
            kernel_name = "cnn_" + uid
            num_args = cnn[-1].num_input_buffers + cnn[-1].num_output_buffers
            print "NUM ARGS ",num_args
            cnn[-1].dump_tinfo(kernel_name, src_file_loc, dimension, tinfoname,num_args)
