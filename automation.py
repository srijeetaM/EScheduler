
import csv 
import os
import sys
import shutil
job_id = sys.argv[1];
start_node = sys.argv[2]
fused_node_id = sys.argv[3]
tinfo_filename= "node_" + start_node+ ":" + fused_node_id
platform_id = sys.argv[4]
device_id = sys.argv[5]
isDependent = '-1'
l_deadline = '999.0'
arrivaltime = '0.0'
deadline = '999.0'
isTerminal = '1'
global_dagid = '0'
instanceid = '0'
frequency = '-1'
mico_kernel = '1'
if platform_id == '1':
    mico_kernel = '0'

for job in ['0','1']:
    path = "tinfo/DAG_"+ job
    files_p = os.listdir(path)
    for file in files_p:
        os.remove(os.path.join(path, file))


path = "tinfo_full/DAG_"+ job_id
path1 = "tinfo/DAG_"+ job_id
files = os.listdir(path)
for file in files:
    if(file == tinfo_filename):
        # print(file)
        shutil.copy(path+'/'+file, path1)

daghistory_file_contents = open("dag_history/dag_history_0.stats",'w')
daghistory_file_contents.write("0 "+job_id+" "+arrivaltime+" " +deadline)
daghistory_file_contents.close()

trace_file_contents = open("trace/dispatch_history_0.stats",'w')
trace=(job_id+","+start_node+","+fused_node_id+","+isDependent+","+ platform_id+","+device_id+","+l_deadline+","+arrivaltime+"," +deadline+","+isTerminal+","+global_dagid+","+instanceid+","+frequency)
# print trace
trace_file_contents.write(trace)
trace_file_contents.close()

lines = open('configure_input.txt').read().splitlines()
lines[20] = 'micro_kernel_device='+mico_kernel
open('configure_input.txt','w').write('\n'.join(lines))

