import os
import sys

if __name__ == '__main__':
	filename = sys.argv[1]
	run_command = "taskset -c 3-7 ./scheduler trace/dispatch_history_0.stats profile_statistics/"+filename+"/"
	make_directory = "mkdir profile_statistics/"+filename+"/"
	os.system(make_directory)
	print make_directory
	for i in range(20):
		command = run_command+filename+"_"+str(i)+".stats"
		print command
		os.system(command)
		