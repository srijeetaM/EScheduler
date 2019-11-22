import os

path = "./tinfo_old/"

files = []
# r=root, d=directories, f = files
for r,d,f in os.walk(path):
    for file in f:
        data = []
        with open(os.path.join(r, file), 'r') as file_content:
            data = file_content.readlines()

        #input buffer    
        ip_buf=data[4].rstrip()
        name,parameters=ip_buf.split("=")  
        p_list = parameters.split(",")
        p_string=name+"="
        i=0
        for p in p_list:
            if i==2:
                p_string=p_string+p+",0,"
            elif i%3==2:         
                p_string=p_string+p+",1,"
            else:
                p_string=p_string+p+","
            i=i+1        
        p_string= p_string[:-1]+"\n"
        data[4]=p_string

        #outbuffer change
        op_buf=data[5].rstrip()
        name,parameters=op_buf.split("=")  
        p_list = parameters.split(",")
        p_string=name+"="
        i=0
        for p in p_list:
            if i==len(p_list)-1:
                p_string=p_string+p+",0,"
            elif i%3==2:         
                p_string=p_string+p+",1,"
            else:
                p_string=p_string+p+","
            i=i+1        
        p_string= p_string[:-1]+"\n"
        data[5]=p_string

        with open(os.path.join("./tinfo/", file), 'w') as file_content:
            file_content.writelines(data)

