import csv
import os
import shutil
import subprocess
import sys
if __name__ == '__main__':
    trace_tinfo_map = {0: 'node_2:2_3', 1: 'node_2:2_3', 2: 'node_0:0', 3: 'node_0:0', 4: 'node_1:1', 5: 'node_1:1', 6: 'node_0:0_1_2', 7: 'node_0:0_1_2', 8: 'node_0:0_1_2_3', 9: 'node_0:0_1_2_3', 10: 'node_0:0_1', 11: 'node_0:0_1', 12: 'node_0:0_1_2_3_4', 13: 'node_0:0_1_2_3_4', 14: 'node_2:2', 15: 'node_2:2', 16: 'node_1:1_2_3', 17: 'node_1:1_2_3', 18: 'node_1:1_2_3_4', 19: 'node_1:1_2_3_4', 20: 'node_3:3', 21: 'node_3:3', 22: 'node_4:4', 23: 'node_4:4', 24: 'node_1:1_2', 25: 'node_1:1_2', 26: 'node_3:3_4', 27: 'node_3:3_4', 28: 'node_2:2_3_4', 29: 'node_2:2_3_4'}
    for key in range(0, 29):
        trace_file= open("./trace/profile_dispatch_history_"+str(key)+".stats").readlines()
        micro_kernel_device = -1
        trace = trace_file[0].split(",")
        if int(trace[4]) == 0:
            micro_kernel_device = 1
        else:
            micro_kernel_device = 0
        command = "taskset -c 3-7 ./run_kernel trace/profile_dispatch_history_"+str(key)+".stats" + " ./tinfo/DAG_0/"+trace_tinfo_map[key]+" " + str(key) +" " +str(micro_kernel_device) + " " + "./dag_history/dag_history_0.stats"
        print command
        # os.system(command)