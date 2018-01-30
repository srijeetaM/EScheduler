import csv
import os
import shutil
import subprocess
import sys
if __name__ == '__main__':
    job_id = sys.argv[1];# 0 and 1 dnn cnn
    start_node_id = 1;# we can get it in the file name
    fused_noid_id = 1;# we can get it in the file name
    isDependent = -1
    # platform_id =   #change between 0 and 1
    device_id = 0 # 0 is cpu 0,1 is GPU
    deadline = 9999.999
    arrivaltime = 0.0
    safetytime = 9999.999
    isTerminal = 1
    global_dagid = 0
    instanceid = 0
    frequency = -1
    freq_list = {}
    flag = 0 # 1 cpu or gpu 0
    fields = []
    rows = []
    path = "tinfo/DAG_"+str(job_id)
    trace_id_file_map = {}
    trace_id = 0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        start_node, fused_kernel = file[5:].split(":")
        for platform_id in [0,1]:
            curr_params = [job_id,start_node,fused_kernel,isDependent,platform_id,device_id,deadline,arrivaltime,safetytime,isTerminal,global_dagid,instanceid,frequency]
            if job_id == 0:
                benchmark = "dnn"
            else:
                benchmark = "cnn"
            dump_file = "trace/profile_"+benchmark+"_history_"+str(trace_id)+".stats"
            # f= open("trace/profile_dispatch_history_0.stats","w+")
            
            params = str(job_id) + "," +str(start_node) + "," + str(fused_kernel) +  "," + str(isDependent)  + "," + str(platform_id) + "," + str(device_id) + "," + str(deadline) + "," + str(arrivaltime) + "," + str(safetytime) + "," + str(isTerminal) + "," + str(global_dagid) + "," + str(instanceid) + "," + str(frequency)
            print "Dumping to file",dump_file,"the string",params
            f=open(dump_file,'w')
            f.write(params)
            f.close()
            trace_id_file_map[trace_id]=file
            trace_id +=1
    print trace_id_file_map