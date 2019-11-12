import os

file = open("cnn_gpu_profiling", "w+")
path = "./time/"
file2 = open("cnn_gpu_profiling_avg", "w+")
dict = {}

for filename in os.listdir(path):
    with open(path+filename, "r") as f:
        for line in f.readlines():
            line_contents = line.split(" ")
            fields = line_contents[0].split("/")
            name = fields[3]
            time = line_contents[3]
            #print time
            file.write(name + " " + time + "\n")
            t = int(time,10)
            #print t
            if name in dict:
        		dict[name] = dict.get(name)+t
            else:
                dict[name] = t
file.close()
for x in sorted(dict.keys()):
    print x
    dict[x]=dict.get(x)/20
    file2.write(x + " " + str(dict.get(x)) + "\n")
file2.close()






# import os

# file = open("gpu_profiling_cnn", "w+")

# path = "./time/"


# for filename in os.listdir(path):
#     with open(path+filename, "r") as f:
#         for line in f.readlines():
#             line_contents = line.split(" ")
#             fields = line_contents[0].split("/")
#             name = fields[3]
#             time = line_contents[3]
#             #print time
#             file.write(name + " " + time + "\n")

# file.close()

