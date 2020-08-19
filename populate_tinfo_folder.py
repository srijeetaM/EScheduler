import csv
import os
import sys
if __name__ == '__main__':
    
    
    path = "./trace"
    for r,d,f in os.walk(path):
        for file in f:
            data = []
            with open(os.path.join(r, file), 'r') as file_content:
                data = file_content.readlines()
            for line in data:
                p_list = line.split(",")
                job_id = p_list[0]
                start_node_id = p_list[1]
                fused_node_id = p_list[2]
                tinfofile_src="./tinfo_full/DAG_"+job_id+"/node_"+start_node_id+":"+fused_node_id
                tinfofile_dst="./tinfo/DAG_"+job_id+"/"
                #print tinfofile_src,tinfofile_dst
                command = "cp "+ tinfofile_src + " " + tinfofile_dst
                # print command
                os.system(command)
    