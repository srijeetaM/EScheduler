import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    
    input_filename = sys.argv[1]
    dag_history_filename = sys.argv[2]
    output_filename = sys.argv[3]
    dag_history = open(dag_history_filename,"r").readlines()
    output_file_contents = open(output_filename,'w')
    job_instance_counter_map = {0:0,1:0,2:0,3:0}
    job_dag_instance_id_map = {}
    job_dag_arrival_map = {}
    input_file_contents=open(input_filename,'r').readlines()
    for line in dag_history:
        line_contents=line.strip("\n").split(" ")
        job_id = int(line_contents[1])
        dag_id = int(line_contents[0])
        job_dag_arrival_map[(job_id,dag_id)] = float(line_contents[2])
    for line in input_file_contents:
        #print line
        line=line.replace("0-1-2-3-4-5", "0_1_2_3_4_5")
        line=line.replace("0-1-2-3-4", "0_1_2_3_4")
        line=line.replace("0-1-2-3", "0_1_2_3")
        line=line.replace("0-1-2", "0_1_2")
        line=line.replace("0-1", "0_1")
        line=line.replace("1-2-3-4-5", "1_2_3_4_5")
        line=line.replace("1-2-3-4", "1_2_3_4")
        line=line.replace("1-2-3", "1_2_3")
        line=line.replace("1-2", "1_2")
        line=line.replace("2-3-4-5", "2_3_4_5")
        line=line.replace("2-3-4", "2_3_4")
        line=line.replace("2-3", "2_3")
        line=line.replace("3-4-5", "3_4_5")
        line=line.replace("3-4", "3_4")
        line=line.replace("4-5", "4_5")
        print line
        modified_line_contents = []
        line_contents =line.strip("\n").split(",")
        job_id = int(line_contents[0])
        dag_id = int(line_contents[-1])
        pair = (job_id,dag_id)	
        if pair not in job_dag_instance_id_map:
            job_dag_instance_id_map[pair]=job_instance_counter_map[job_id]
            job_instance_counter_map[job_id] += 1
        # modified_line = line.strip("\n")+"," +str(job_dag_instance_id_map[pair]) +",-1\n"
        modified_line_contents = line_contents[0:7]
        #print job_dag_arrival_map
        modified_line_contents.append(str(job_dag_arrival_map[pair]))
        modified_line_contents.extend(line_contents[7:10])
        modified_line_contents.append(str(job_dag_instance_id_map[pair]))
        modified_line_contents.append("-1")
        modified_line = ",".join(modified_line_contents)+"\n"
        output_file_contents.write(modified_line)
    #print job_dag_instance_id_map    
    output_file_contents.close()
