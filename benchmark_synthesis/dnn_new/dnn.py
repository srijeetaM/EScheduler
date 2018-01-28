from fusion_template import *


class DNN(Kernel):
    def __init__(
        self,
        uid,
        dag_id,
        depth,
        num_input_buffers,
        num_output_buffers,
        num_variables,
        buffer_index,
        variable_index,
        datatype,
        
    ):

        super(DNN, self).__init__(
            uid=uid,
            dag_id=dag_id,
            depth=depth,
            num_input_buffers=num_input_buffers,
            num_output_buffers=num_output_buffers,
            num_variables=num_variables,
            buffer_index=buffer_index,
            variable_index=variable_index,
            
            
        )
        # self.header = "#if defined(cl_khr_fp64) \n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#elif defined(cl_amd_fp64)  // AMD extension available?\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#endif\n"
        self.name = "dnn_" + str(uid)
        self.datatype="float"

    def get_variables(self,variables):
        for j in range(len(variables)):
            self.variable_values.append(variables[j])

    def initialise_kernel_info(self):
        self.set_kernel_info()
        worksize= max(self.variable_values)
        self.global_work_size = [worksize, 1, 1]   

    def get_input_buffer_sizes(self):
        ipbuff_size=[]
        # print self.depth
        for i in range(self.depth):
            if i==0:
                ipbuff_size.append(self.variable_values[i*2+0])
                ipbuff_size.append(self.variable_values[i*2+0] * self.variable_values[i*2+1])
                ipbuff_size.append(self.variable_values[i*2+1])
            else:
                ipbuff_size.append(self.variable_values[i*2+0] * self.variable_values[i*2+1] )
                ipbuff_size.append(self.variable_values[i*2+1])
        # print ipbuff_size
        return ipbuff_size

    def get_output_buffer_sizes(self):
        opbuff_size=[]
        for i in range(self.depth):
            opbuff_size.append(self.variable_values[i*2+1])
        return opbuff_size

    def set_kernel_info(self):
        arg_index = 0
        buffer_sizes = {}
        buffer_sizes["input"] = self.get_input_buffer_sizes()
        buffer_sizes["output"] = self.get_output_buffer_sizes()

        # print buffer_sizes, self.num_input_buffers, self.num_output_buffers
        
        counter = 0
        for i in range(self.num_input_buffers):
            # print counter , buffer_sizes['input'][counter]
            self.buffer_info["input"].append(
                (
                    self.datatype,
                    buffer_sizes["input"][counter],
                    arg_index,
                    "b" + str(self.buffer_index),
                )
            )
            arg_index += 1
            self.buffer_index += 1
            counter += 1
            # print counter
        counter = 0
        for i in range(self.num_output_buffers):
            # print counter , buffer_sizes['output'][counter]
            self.buffer_info["output"].append(
                (
                    self.datatype,
                    buffer_sizes["output"][counter],
                    arg_index,
                    "b" + str(self.buffer_index),
                )
            )
            arg_index += 1
            self.buffer_index += 1
            counter += 1
            # print counter

    def define_variables(self):
        define_var = ""
        counter = 0
        # print self.variable_values, self.num_variables
        for i in range(self.num_variables):
            self.variable_info.append(
                ("int", self.variable_values[counter], "v" + str(self.variable_index))
            )
            define_var += (
                "#define v"
                + str(self.variable_index)
                + " "
                + str(self.variable_values[i])
                + "\n"
            )
            self.variable_index += 1
            counter += 1

        define_var += "\n"
        return define_var

    def function_signature(self):
        bi = []
        bo = []
        v = []
        function_sig = ""
        for i in range(self.num_input_buffers):
            bi.append(self.buffer_info["input"][i][-1])
        for i in range(self.num_output_buffers):
            bo.append(self.buffer_info["output"][i][-1])
        for i in range(self.num_variables):
            v.append(self.variable_info[i][-1])
        datatype = self.datatype
        function_sig += "__kernel void " + self.name + "("
        function_sig += " __global const " + datatype + "* " + bi[0]
        for i in range(len(bi) - 1):
            function_sig += ", __global const " + datatype + "* " + bi[i + 1]
        for i in range(len(bo)):
            function_sig += ", __global " + datatype + "* " + bo[i]
        # for i in range(len(v)):
        # 	String1+= " ,int " + v[i]
        function_sig += ")\n{\n"
        # print function_sig
        return function_sig

    def global_ids(self):
        String1 = ""
        String1 += "\tint globalId = get_global_id(0);\n\n"
        # print String1
        return String1



    def init_private_variables(self):
        pv = ""
        pv += "\ttypedef float4 floatX;\n\tfloatX wt,temp;\n\tfloat dotProduct;\n\n"
        return pv

    def load(self):
        pass

    def load_compute_store(self,depth):
        bi = []
        bo = []
        v = []
        for i in range(self.num_input_buffers):
            bi.append(self.buffer_info["input"][i][-1])
        for i in range(self.num_output_buffers):
            bo.append(self.buffer_info["output"][i][-1])
        code = ""
        code += "\n\tdotProduct=0.0;\n"
        code += "\tif(globalId < " + self.variable_info[depth*2+1][-1] + ")\n\t{\n\t#pragma unroll\n"
        code += "\t\tfor(int x=0; x<"+ self.variable_info[depth*2+0][-1] + "/4; x++)\n\t\t{\n"
        if depth==0:  
            code += "\t\t\ttemp= vload4(0,(__global const float *)" + bi[0] + "+(4*x));\n"
            code += (
                "\t\t\twt= vload4(0,(__global const float *)"
                + bi[1]
                + "+(globalId*"
                + self.variable_info[depth*2+0][-1]
                + "+4*x));\n"
            )
            code += ("\t\t\tdotProduct += dot(wt,temp);\n\t\t}\n\t\t"
                + bo[0]
                + "[globalId] = dotProduct+"
                + bi[2]
                + "[globalId];\n\t}\n"
            )
        else:
            code += "\t\t\ttemp= vload4(0,(__global float *)" + bo[depth-1] + "+(4*x));\n"    
            code += (
                "\t\t\twt= vload4(0,(__global const float *)"
                + bi[depth*2+1]
                + "+(globalId*"
                + self.variable_info[depth*2+0][-1]
                + "+4*x));\n"
            )
            code += ("\t\t\tdotProduct += dot(wt,temp);\n\t\t}\n\t\t"
                + bo[depth]
                + "[globalId] = dotProduct+"
                + bi[depth*2+2]
                + "[globalId];\n\t}\n"
            )
        return code

    def synchronization(self):
        code = ""
        code += "\n\tbarrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n"
        return code

    def store(self):
        pass

    def generate_source_code(self):
        # self.source_code += self.header
        self.source_code += self.define_variables()
        self.source_code += self.function_signature()
        self.source_code += self.global_ids()
        self.source_code += self.init_private_variables()
        self.source_code += self.load_compute_store()
        self.source_code += "\n}"

    def generate_fused_source_code(self):
        self.source_code += self.define_variables()
        self.source_code += self.function_signature()
        self.source_code += self.global_ids()
        self.source_code += self.init_private_variables()
        for i in range(self.depth):
            self.source_code += self.load_compute_store(i)
            if i < self.depth -1:
                self.source_code += self.synchronization()
        self.source_code += "\n}"


