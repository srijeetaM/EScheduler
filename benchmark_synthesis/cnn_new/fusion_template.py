import networkx as nx


class DNN(object):
    def __init__(self, dag_id, input_size, cfg):
        self.dag_id = dag_id
        self.tasks, self.skeleton = self.get_dag_stats(input_size, cfg)
        self.task_components = dict()
        self.finished_tasks = list()
        self.free_task_components = list()
        self.processing_tasks = list()

        self.task_component_mappings = dict()
        self.task_component_id_map = dict()
        self.num_nodes = nx.number_of_nodes(self.skeleton)
        self.G = nx.relabel_nodes(
            self.skeleton, lambda s: FusedKernel(self.tasks[s]), copy=True
        )
        for task_component in self.G.nodes():
            for kid in task_component.get_kernel_ids():
                self.task_components[kid] = task_component
            if not self.get_task_component_parents(task_component):
                self.free_task_components.append(task_component)
        for task_component in self.G.nodes():
            self.task_component_id_map[task_component.id] = task_component

    def get_dag_stats(self, input_size, cfg):
        kernel_id = 0
        task_dict = {}

        for layer_type, layer_properties in cfg:
            if layer_type == "L":
                datatype, num_in, num_out = layer_properties
                task_dict[kernel_id] = Gemm(
                    kernel_id,
                    self.dag_id,
                    2,
                    1,
                    3,
                    input_size,
                    num_in,
                    num_out,
                    0,
                    0,
                    datatype,
                )
                input_size = num_out
                kernel_id += 1
        dag = nx.path_graph(len(cfg), create_using=nx.DiGraph())
        return task_dict, dag

    def get_kernel_parent_ids(self, kid):
        """
        Should return a list of kernel ids that are predecessors to given kernel.
        """
        return self.skeleton.predecessors(kid)

    def get_kernel_children_ids(self, kid):
        """
        Should return a list of kernel ids that are successors to given kernel.
        """
        return self.skeleton.successors(kid)

    def get_skeleton_subgraph(self, kernel_ids):
        return self.skeleton.subgraph(kernel_ids)

    def get_task_component_parents(self, task_component):
        return self.G.predecessors(task_component)

    def get_task_component_children(self, task_component):
        return self.G.successors(task_component)

    def update_dependencies(self, task_component):
        """
        Updates task dependencies. Call this whenever a task is modified. Adds or remove edges to task dag based on
        skeleton kernel dag for the given task.
        :param task:
        :return:
        """
        p, c = (
            set(self.get_task_component_parents(task_component)),
            set(self.get_task_component_children(task_component)),
        )
        pt, ct = set(), set()
        for kid in task_component.get_kernel_ids():
            for pkid in self.get_kernel_parent_ids(kid):
                pt.add(self.task_components[pkid])
            for ckid in self.get_kernel_children_ids(kid):
                ct.add(self.task_components[ckid])
        pt -= set([task_component])
        ct -= set([task_component])
        for t in pt - p:
            self.G.add_edge(t, task_component)
        for t in ct - c:
            self.G.add_edge(task_component, t)
        for t in p - pt:
            self.G.remove_edge(t, task_component)
        for t in c - ct:
            self.G.remove_edge(task_component, t)

    def merge_task_components(self, t1, t2):
        dependencies = set().union(
            *[set(self.get_kernel_parent_ids(kid)) for kid in t2.get_kernel_ids()]
        )

        if set(t1.get_kernel_ids()) >= dependencies:
            t1.add_kernels_from_task(t2)
        else:
            raise Exception("Some dependent kernels are not part of this task.")
        for kid in t2.get_kernel_ids():
            self.task_components[kid] = t1
        self.update_dependencies(t1)
        self.G.remove_node(t2)
        self.task_component_id_map.pop(t2.id)

    def merge_independent_task_components(self, t1, t2):
        # print t2.get_kernel_ids(), t2
        # for node in self.G.nodes():
        #     print node.get_kernel_ids(),node
        t1.add_kernels_from_task(t2)
        for kid in t2.get_kernel_ids():
            self.task_components[kid] = t1
        self.update_dependencies(t1)
        # print "Removing ",t2.get_kernel_ids(), t2
        self.G.remove_node(t2)
        self.task_component_id_map.pop(t2.id)

    def merge_task_list(self, t):
        # print "Tasks to be merged: ",
        # for task_component in t:
        #     print task_component.get_kernel_names(),
        t1 = t[0]
        for t2 in t[1:]:
            # print "T1 ",
            # print t1.get_kernel_names()
            # print "T2 ",
            # print t2.get_kernel_names()
            self.merge_independent_task_components(t1, t2)
        return t1

    def init_gemm(self, index):
        fused_kernel = self.task_components[index]
        kernel1 = self.tasks[index]
        kernel2 = self.tasks[index + 1]
        fused_kernel.buffer_index = 0
        fused_kernel.variable_index = 0
        fused_kernel.num_input_buffers = 3
        fused_kernel.num_output_buffers = 2
        fused_kernel.num_variables = 5
        fused_kernel.input_buffer_sizes = kernel1.get_input_buffer_sizes()
        fused_kernel.output_buffer_sizes = kernel1.get_output_buffer_sizes()
        fused_kernel.input_buffer_sizes.extend(kernel2.get_input_buffer_sizes())
        fused_kernel.output_buffer_sizes.extend(kernel2.get_output_buffer_sizes())
        fused_kernel.variable_values = kernel1.get_variable_values()
        fused_kernel.variable_values.extend(kernel2.get_variable_values()[1:])

    def kernel_fusion(self, node_ids):
        for index in range(len(node_ids) - 1):
            if (
                self.G.tasks[index].name == "gemm"
                and self.G.tasks[index + 1].name == "gemm"
            ):
                if self.task_components[index].is_supertask():
                    pass
                else:
                    self.init_gemm(index)


class Kernel(object):
    def __init__(
        self, uid, dag_id, start_node, depth, cfg, buffer_index, variable_index
    ):
        self.id = uid
        self.dag_id = dag_id
        self.start_node = start_node
        self.num_input_buffers = 0
        self.num_output_buffers = 0
        self.cfg = cfg
        # self.num_variables = num_variables
        self.source_code = ""
        self.depth = depth
        self.input_buffer_names = []
        self.output_buffer_names = []
        self.ipbuffer_info = {}
        self.opbuffer_info = {}
        self.variable_info = []
        self.variable_names = []
        self.variable_values = []
        self.buffer_index = buffer_index
        self.variable_index = variable_index
        self.header = "#if defined(cl_khr_fp64) \n #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n #elif defined(cl_amd_fp64)  // AMD extension available?\n #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n #endif\n #define TS 4\n #define pool_size 2\n"
        self.global_work_size = []
        self.local_work_size = []
        self.work_dimension = 1

    def thread_ids(self, kernel_dimension):
        if kernel_dimension == 1:
            return " int tx = get_local_id(0);\n int bx = get_group_id(0);\n"
        if kernel_dimension == 2:
            return " int tx = get_local_id(0); \n int ty = get_local_id(1); \n	int bx = get_group_id(0);\n	int by = get_group_id(1)\n;"

    def load(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def dump_tinfo(self, name, source, dimension, filename,num_args=-1):
        kernelName = "KernelName=" + name + "\n"
        kernelSource = "KernelSource=" + source + "\n"
        workDimension = "workDimension=" + str(dimension) + "\n"
        globalWorkSize = (
            "globalWorkSize=" + ",".join([str(g) for g in self.global_work_size]) + "\n"
        )
        localWorkSize = (
            "localWorkSize=" + ",".join([str(g) for g in self.local_work_size]) + "\n"
        )

        inputBuffers = ""
        # print self.start_node, self.start_node, self.depth
        # print self.ipbuffer_info
        for l in range(self.start_node, self.start_node + self.depth):
            for input_buffer in self.ipbuffer_info[l]:
                inputBuffers = inputBuffers + ",".join(
                    [str(b) for b in input_buffer[:-1]]
                )
                if input_buffer[:-1][2]==0:
                    inputBuffers += ",0"
                else:
                    inputBuffers += ",1"
            
                inputBuffers += ","
        inputBuffers = "inputBuffers=" + inputBuffers[:-1] + "\n"

        outputBuffers = ""
        for l in range(self.start_node, self.start_node + self.depth):
            for output_buffer in self.opbuffer_info[l]:
                outputBuffers = outputBuffers + ",".join(
                    [str(b) for b in output_buffer[:-1]]
                )
                print self.start_node, self.depth
                if self.depth == 1:
                    outputBuffers += ",0"
                else:
                    print "ARGUMENT Position", output_buffer[:-1], num_args-1
                    if output_buffer[:-1][2]==num_args-1:
                        outputBuffers += ",0"
                    else:
                        outputBuffers += ",1"
        
                        
                outputBuffers += ","
        
        outputBuffers = "outputBuffers=" + outputBuffers[:-1] + "\n"

        varArguments = ""
        for variables in self.variable_info:
            varArguments = varArguments + ",".join([str(v) for v in variables[:-1]])
            varArguments += ","
        varArguments = "varArguments=" + varArguments[:-1] + "\n"
        # print self.opbuffer_info
        # print self.depth - 1
        outflow = ""
        outflow += (
            "data_outflow="
            + str(int(self.name[-1:], 10) + 1)
            + ",0,"
            + str(self.opbuffer_info[self.start_node + self.depth - 1][-1][-2])
        )
        # print outflow

        f = open(filename, "w")
        f.write(kernelName)
        f.write(kernelSource)
        f.write(workDimension)
        f.write(globalWorkSize)
        # f.write(localWorkSize)
        f.write(inputBuffers)
        f.write(outputBuffers)
        # f.write(varArguments)
        if int(self.name[-1:], 10) + 1 < 6:
            f.write(outflow)
        f.close()


class FusedKernel(object):
    def __init__(self, kernel):
        import uuid

        self.id = str(uuid.uuid1())
        self.kernels = set()
        self.sorted_kernels = list()
        self.kernels.add(kernel)
        self.dag_id = kernel.dag_id
        self.num_input_buffers = -1
        self.num_output_buffers = -1
        self.num_variables = -1
        self.source_code = ""
        self.input_buffer_names = []
        self.output_buffer_names = []
        self.ipbuffer_info = {}
        self.opbuffer_info = {}
        self.variable_info = []
        self.variable_names = []
        self.buffer_index = -1
        self.variable_index = -1
        self.header = "#if defined(cl_khr_fp64) \n #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n #elif defined(cl_amd_fp64)  // AMD extension available?\n #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n #endif\n #define TS 4\n #define pool_size 2\n"
        self.global_work_size = []
        self.local_work_size = []
        self.input_buffer_sizes = []
        self.output_buffer_sizes = []
        self.variable_values = []

    def set_kernel_info(self):

        arg_index = 0
        buffer_sizes = {}
        buffer_sizes["input"] = self.input_buffer_sizes
        buffer_sizes["output"] = self.output_buffer_sizes
        variable_values = self.variable_values
        counter = 0
        for i in range(self.num_input_buffers):
            self.buffer_info["input"].append(
                (
                    buffer_sizes["input"][counter],
                    arg_index,
                    self.datatype,
                    "b" + str(self.buffer_index),
                )
            )
            arg_index += 1
            self.buffer_index += 1
            counter += 1
        counter = 0
        for i in range(self.num_output_buffers):
            self.buffer_info["output"].append(
                (
                    buffer_sizes["output"][counter],
                    arg_index,
                    self.datatype,
                    "b" + str(self.buffer_index),
                )
            )
            arg_index += 1
            self.buffer_index += 1
            counter += 1
        counter = 0
        for i in range(self.num_variables):
            self.variable_info.append(
                (
                    variable_values[counter],
                    arg_index,
                    "int",
                    "v" + str(self.variable_index),
                )
            )
            arg_index += 1
            self.variable_index += 1
            counter += 1

    def get_first_kernel(self):
        """
        Returns the first kernel in the fused kernel (indegree zero)
        :return:
        :rtype:
        """
        return list(self.kernels)[0]

    def get_kernels(self):
        """
        Returns all kernels in fused kernel (Kernel Objects )
        :return:
        :rtype:
        """
        return self.kernels

    def get_kernel_ids(self):
        """
        Return list of ids pertaining to each kernel 
        :return:
        :rtype:
        """
        return map(lambda k: k.id, self.get_kernels())


class Gemm(Kernel):
    def __init__(
        self,
        uid,
        dag_id,
        num_input_buffers,
        num_output_buffers,
        num_variables,
        input_size,
        num_in,
        num_out,
        buffer_index,
        variable_index,
        datatype,
    ):

        super(Gemm, self).__init__(
            uid=uid,
            dag_id=dag_id,
            num_input_buffers=num_input_buffers,
            num_output_buffers=num_output_buffers,
            num_variables=num_variables,
            buffer_index=buffer_index,
            variable_index=variable_index,
        )
        self.num_in = num_in
        self.num_out = num_out
        self.input_size = input_size
        self.datatype = "float"
        self.name = "gemm"
        self.set_kernel_info()
        self.global_work_size = [input_size, num_out, 1]

    def get_input_buffer_sizes(self):
        if self.datatype == "float":
            return [self.input_size * self.num_in * 4, self.num_in * self.num_out * 4]

    def get_output_buffer_sizes(self):
        if self.datatype == "float":
            return [self.input_size * self.num_out * 4]

    def get_variable_values(self):
        return [self.input_size, self.num_in, self.num_out]

    def set_kernel_info(self):

        arg_index = 0
        buffer_sizes = {}
        buffer_sizes["input"] = self.get_input_buffer_sizes()
        buffer_sizes["output"] = self.get_output_buffer_sizes()
        variable_values = self.get_variable_values()
        counter = 0
        for i in range(self.num_input_buffers):
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
        counter = 0
        for i in range(self.num_output_buffers):
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
        counter = 0
        for i in range(self.num_variables):
            self.variable_info.append(
                (
                    "int",
                    variable_values[counter],
                    arg_index,
                    "v" + str(self.variable_index),
                )
            )
            arg_index += 1
            self.variable_index += 1
            counter += 1

    def function_signature(self):

        b1 = self.buffer_info["input"][0][-1]
        b2 = self.buffer_info["input"][1][-1]
        b3 = self.buffer_info["output"][0][-1]
        v1 = self.variable_info[0][-1]
        v2 = self.variable_info[1][-1]
        v3 = self.variable_info[2][-1]
        datatype = self.datatype
        return (
            "__kernel void "
            + self.name
            + "( __global"
            + datatype
            + "* "
            + b1
            + ", __global "
            + datatype
            + "* "
            + b2
            + ", __global "
            + datatype
            + "* "
            + b3
            + " ,int "
            + v1
            + ", int "
            + v2
            + ", int "
            + v3
            + ") {\n"
        )

    def init_matrix_coordinates(self):
        return "\t int Row = get_global_id(1) * 4;\n\t int Col = get_global_id(0);\n"

    def init_private_variables(self):
        return (
            "\t typedef float4 floatX;\n\t floatX wt;\n\t floatX tmp0,tmp1,tmp2,tmp3;\n"
        )

    def init_shared_memory(self):
        return "\t__local float sub1[TS][TS];\n\t__local float sub2[TS][TS];\n\t int numTiles=0\n"

    def set_num_tiles(self, v1):
        pass

    def load(self):
        pass

    def load_compute_store(self):  # M,N,K
        b1 = self.buffer_info["input"][0][-1]
        b2 = self.buffer_info["input"][1][-1]
        b3 = self.buffer_info["output"][0][-1]
        v1 = self.variable_info[0][-1]
        v2 = self.variable_info[1][-1]
        v3 = self.variable_info[2][-1]
        code = ""
        code += (
            "\t numTiles = (TS+"
            + v1
            + "-1)/TS;\n\t for (int t=0; t<numTiles; t++) {\n\t\t for (int k=0; k<4; k++) { \n\t\t\t  if (t*TS + tx < "
            + v2
            + "&& Row < "
            + v1
            + ") \n\t\t\t\t  sub1[ty+k][tx] = "
            + b1
            + "[(Row+k)*N + (t*TS + tx)]; \n\t\t\t  else\n\t\t\t\t sub1[ty+k][tx] = 0.0;\n\t\t\t  if (t*TS + ty <"
            + v2
            + "&& Col < "
            + v3
            + ")\n\t\t\t\t sub2[tx][ty+k] = "
            + b2
            + "[Col + (t*TS + ty+k)*"
            + v3
            + "];\n\t\t\t else\n\t\t\t\t sub2[tx][ty+k] = 0.0;\n\t\t }"
        )
        code += "\n\t\t barrier(CLK_LOCAL_MEM_FENCE);\n\t\t wt = vload4(0,(__local float *)sub2+(4*tx));\n\t\t tmp0 = vload4(0,(__local float *)sub1);\n\t\t tmp1 = vload4(0,(__local float *)sub1+4);\n\t\t tmp2 = vload4(0,(__local float *)sub1+8);\n\t\t tmp3 = vload4(0,(__local float *)sub1+12);\n\t\t a ??cc0.x +=dot(wt,tmp0);\n\t\t acc0.y +=dot(wt,tmp1); \n\t\t acc0.z +=dot(wt,tmp2);\n\t\t acc0.w +=dot(wt,tmp3);\n\t\t barrier(CLK_LOCAL_MEM_FENCE);\n\t }\n"
        code += (
            "\n\tif (Row < "
            + v1
            + " && Col < "
            + v3
            + "){\n\t\t "
            + b3
            + "[Row*"
            + v3
            + "+ Col] = acc0.x;\n\t\t "
            + b3
            + "[(Row+1)*"
            + v3
            + "+ Col] = acc0.y;\n\t\t "
            + b3
            + "[(Row+2)*"
            + v3
            + "+ Col] = acc0.z;\n\t\t "
            + b3
            + "[(Row+3)*"
            + v3
            + "+ Col] = acc0.w; \n\t}\n"
        )
        return code

    def store(self):
        pass

    def generate_source_code(self):
        self.source_code += self.header
        self.source_code += self.function_signature()
        self.source_code += self.thread_ids(2)
        self.source_code += self.init_matrix_coordinates()
        self.source_code += self.init_private_variables()
        self.source_code += self.init_shared_memory()
        self.source_code += self.load_compute_store()
        self.source_code += "\n}"

