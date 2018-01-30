import csv
import os
import shutil
import subprocess
import sys
if __name__ == '__main__':
    trace_tinfo_map_dnn = {0: 'node_2:2_3', 1: 'node_2:2_3', 2: 'node_0:0', 3: 'node_0:0', 4: 'node_1:1', 5: 'node_1:1', 6: 'node_0:0_1_2', 7: 'node_0:0_1_2', 8: 'node_0:0_1_2_3', 9: 'node_0:0_1_2_3', 10: 'node_0:0_1', 11: 'node_0:0_1', 12: 'node_0:0_1_2_3_4', 13: 'node_0:0_1_2_3_4', 14: 'node_2:2', 15: 'node_2:2', 16: 'node_1:1_2_3', 17: 'node_1:1_2_3', 18: 'node_1:1_2_3_4', 19: 'node_1:1_2_3_4', 20: 'node_3:3', 21: 'node_3:3', 22: 'node_4:4', 23: 'node_4:4', 24: 'node_1:1_2', 25: 'node_1:1_2', 26: 'node_3:3_4', 27: 'node_3:3_4', 28: 'node_2:2_3_4', 29: 'node_2:2_3_4'}
    trace_tinfo_map_cnn = {0: 'node_2:2_3', 1: 'node_2:2_3', 2: 'node_0:0', 3: 'node_0:0', 4: 'node_1:1', 5: 'node_1:1', 6: 'node_0:0_1_2', 7: 'node_0:0_1_2', 8: 'node_0:0_1_2_3', 9: 'node_0:0_1_2_3', 10: 'node_0:0_1', 11: 'node_0:0_1', 12: 'node_0:0_1_2_3_4', 13: 'node_0:0_1_2_3_4', 14: 'node_2:2', 15: 'node_2:2', 16: 'node_1:1_2_3', 17: 'node_1:1_2_3', 18: 'node_1:1_2_3_4', 19: 'node_1:1_2_3_4', 20: 'node_2:2_3_4_5', 21: 'node_2:2_3_4_5', 22: 'node_3:3_4_5', 23: 'node_3:3_4_5', 24: 'node_1:1_2_3_4_5', 25: 'node_1:1_2_3_4_5', 26: 'node_3:3', 27: 'node_3:3', 28: 'node_5:5', 29: 'node_5:5', 30: 'node_4:4', 31: 'node_4:4', 32: 'node_1:1_2', 33: 'node_1:1_2', 34: 'node_3:3_4', 35: 'node_3:3_4', 36: 'node_2:2_3_4', 37: 'node_2:2_3_4', 38: 'node_4:4_5', 39: 'node_4:4_5', 40: 'node_0:0_1_2_3_4_5', 41: 'node_0:0_1_2_3_4_5'}
    benchmark = ""
    trace_tinfo_map = None
    if sys.argv[1] == "0":
        benchmark = "dnn"
        trace_tinfo_map = trace_tinfo_map_dnn    
    else:
        benchmark = "cnn"
        trace_tinfo_map = trace_tinfo_map_cnn
    
    for key in range(0,len(trace_tinfo_map.keys())):
        trace_file= open("./trace/profile_"+benchmark+"_history_"+str(key)+".stats").readlines()
        micro_kernel_device = -1
        trace = trace_file[0].split(",")
        if int(trace[4]) == 0:
            micro_kernel_device = 1
        else:
            micro_kernel_device = 0
        command = "taskset -c 3-7 ./run_kernel trace/profile_"+ benchmark + "_history_"+str(key)+".stats" + " ./tinfo/DAG_"+sys.argv[1]+"/"+trace_tinfo_map[key]+" " + str(key) +" " +str(micro_kernel_device) + " " + "./dag_history/dag_history_0.stats"
        print command
        # os.system(command)