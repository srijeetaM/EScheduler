import os
import sys
from collections import defaultdict
import numpy as np
if __name__ == '__main__':
	input_arg = sys.argv[1]
	filename = "profile_statistics/"+input_arg+"/"+input_arg
	profile_map = defaultdict(lambda: defaultdict(list))
	for i in range(0,20):
		f = open(filename+"_"+str(i)+".stats",'r')
		profile_contents = f.readlines()
		for line in profile_contents:
			values = line.strip("\n").split("\t\t")
			dag_id = int(values[0])
			task_id = int(values[1])
			profile_map[(dag_id,task_id)]["write_delay"].append(float(values[2]))
			profile_map[(dag_id,task_id)]["write"].append(float(values[3]))
			profile_map[(dag_id,task_id)]["execute_delay"].append(float(values[4]))
			profile_map[(dag_id,task_id)]["execute"].append(float(values[5]))
			profile_map[(dag_id,task_id)]["read_delay"].append(float(values[6]))
			profile_map[(dag_id,task_id)]["read"].append(float(values[7]))
			profile_map[(dag_id,task_id)]["device_execution"].append(float(values[8]))
			# profile_map[(dag_id,task_id)]["host_execution"].append(float(values[9]))
			profile_map[(dag_id,task_id)]["host_overhead"].append(float(values[10]))
			profile_map[(dag_id,task_id)]["callback_time"].append(float(values[11]))
			# profile_map[(dag_id,task_id)]["callback_overhead"].append(float(values[12]))

	for id_values in profile_map.keys():
		print id_values
		print "---------------------------------------"
		for key in profile_map[id_values].keys():
			mean = np.mean(profile_map[id_values][key])
			std = np.std(profile_map[id_values][key])
			cv = std/mean
			print key,mean/1000,std/1000,cv
		print "---------------------------------------"




