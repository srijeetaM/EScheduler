
#ifndef __FUNCTIONALITIES_H  // Control inclusion of header files
#define __FUNCTIONALITIES_H 

// System includes
#include <stdio.h> 
#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <cmath> 
#include <queue>
#include <map>
#include <unistd.h>
#include <ctime>
#include <time.h>
#include <thread> 
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include<sys/resource.h>
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <mutex>  



// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// #define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
// #undef CL_VERSION_1_2
// #include <CL/cl.hpp>
#include <CL/cl_ext.h>
// Custom includes
#include "string.h"

int STR_LENGTH; 
int NumOfJobs;
int NumCoresPerDevice;
int NumofCPUs;
int NumOfSensors;
int NumOfTempBand;
int PLATFORM_GPU;
int PLATFORM_CPU;
int G_BIG;
int G_LITTLE;
int C_BIG;
int NumNodes_0;
int NumNodes_1;
int TempInterval;
int POLE;
int MODE;
int numOfHyperperiod;
int LOG_LEVEL;
int LOG_SCHEDULER;
int LOG_PROFILE;
int micro_kernel_device;
int time_buffer;
int FACTOR;
int SAFE;
int isProfileMode;
int monitorTemp;
int controlerTemp;
int generatePlot;
int numOfDAGs;
int raceToIdle;
int no_micro_kernel=0;
int rblock=0;
int nlock=0;
int devlock=0;
int nKernels=0;
int nTasks=0;
int SchedulerFinish=0;
int stop_scheduler=0;
unsigned long long int safe_duration=0;
unsigned long long int notify_cb_buffer_c=0;
unsigned long long int notify_cb_buffer_g=0; 
unsigned long long int START_TIME;
unsigned long long int hyper_period;
std::mutex mtx_rblock;
std::mutex mtx_nlock;
std::mutex mtx_devlock;
std::string outputbuffer;
/**************************** Helper functions ****************************************/

//OpenCL Error Logs

void print_profile_event_status(cl_int status)
{
    if(status==CL_SUCCESS)
        printf("CL_SUCCESS\n");
    else if(status==CL_PROFILING_INFO_NOT_AVAILABLE)
        printf("CL_PROFILING_INFO_NOT_AVAILABLE \n");
    else if(status==CL_INVALID_VALUE)
        printf("CL_INVALID_VALUE \n");
    else if(status==CL_INVALID_EVENT)
        printf("CL_INVALID_EVENT \n");
    else if(status==CL_OUT_OF_RESOURCES)
        printf("CL_OUT_OF_RESOURCES \n");
    else if(status==CL_OUT_OF_HOST_MEMORY)
        printf("CL_OUT_OF_HOST_MEMORY \n");
}

void check(cl_int status, const char* str) {
    if(status != CL_SUCCESS) {
        // fprintf(fp,"Failed : %s. Error %d\n", str, status);
        printf("Failed %s Error %d\n", str,status);
        exit(EXIT_FAILURE);
    }
}

//FILE IO

inline bool file_exists(const std::string& name) 
{
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

int write_file(const char *name, const unsigned char *content, size_t size) 
{
  FILE *fp = fopen(name, "wb+");
  if (!fp) 
    return -1;
  
  fwrite(content, size, 1, fp);
  fclose(fp);
  return 0;
}

int read_file(unsigned char **output, size_t *size, const char *name) 
{
  FILE* fp = fopen(name, "rb");
  if (!fp) 
    return -1;
  

  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(*size);
  if (!*output) 
  {
    fclose(fp);
    return -1;
  }

  fread(*output, *size, 1, fp);
  fclose(fp);
  return 0;
}

cl_int write_binaries(std::string filename,cl_program program, unsigned num_devices,
                      cl_uint platform_idx) {
  unsigned i;
  cl_int err = CL_SUCCESS;
  size_t *binaries_size = NULL;
  unsigned char **binaries_ptr = NULL;
  printf("Generating binaries for %d devices of platform %d\n",num_devices,platform_idx);
  // Read the binaries size
  size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
  binaries_size = (size_t *)malloc(binaries_size_alloc_size);
  if (!binaries_size) {
    err = CL_OUT_OF_HOST_MEMORY;
   
  }

  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                         binaries_size_alloc_size, binaries_size, NULL);
  if (err != CL_SUCCESS) {
    
  }

  // Read the binaries
  size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
  binaries_ptr = (unsigned char **)malloc(binaries_ptr_alloc_size);
  if (!binaries_ptr) {
    err = CL_OUT_OF_HOST_MEMORY;
    
  }
  memset(binaries_ptr, 0, binaries_ptr_alloc_size);
  for (i = 0; i < num_devices; ++i) {
    binaries_ptr[i] = (unsigned char *)malloc(binaries_size[i]);
    if (!binaries_ptr[i]) {
      err = CL_OUT_OF_HOST_MEMORY;
      
    }
  }

  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_ptr_alloc_size,
                         binaries_ptr, NULL);


  // Write the binaries to file
  for (i = 0; i < num_devices; ++i) {
    // Create output file name
 
    // Write the binary to the output file
    write_file(filename.c_str(), binaries_ptr[i], binaries_size[i]);
  }

  return err;
}


/**************************** Helper functions ****************************************/

typedef struct _jobinfo
{  
    int jobID;
    int numOfNodes;
    int numOfEdges;
    std::vector <std::vector <int>> dependency;
    std::vector <std::vector <int>> successors;
}JobInfo;


typedef struct _daginfo
{  
    int globalDAGID;
    int jobID;
    int instanceID;
    
}DAGInfo;

typedef struct _kernelinfo
{   
    std::string KernelName;
    std::string kernelSource;
    unsigned int noInputBuffers;
    unsigned int noOutputBuffers;
    unsigned int noIOBuffers;
    std::vector <unsigned int> bufferInputID;
    std::vector <unsigned int> bufferOutputID;  
    cl_kernel kernelObject;
    std::vector<cl_kernel> kernelObjects;
    int workDimension;
    size_t globalWorkSize[3];
    size_t localWorkSize[3];
    // Data Type, Size, Position,Persistence
    std::vector < std::tuple <std::string, unsigned int, unsigned int, unsigned int> > inputBuffers;
    std::vector < std::tuple <std::string, unsigned int, unsigned int, unsigned int> > outputBuffers;
    std::vector < std::tuple <std::string, unsigned int, unsigned int, unsigned int> > ioBuffers;
    std::vector < std::tuple <std::string, unsigned int, unsigned int> > varArguments;
    std::vector < std::tuple <std::string, unsigned int, unsigned int> > localArguments;
    std::vector < unsigned long long int> nonPartition;  
    std::vector < std::tuple <unsigned int, unsigned int, unsigned int> >data_outflow;
    std::string options;
    std::vector <float> localSizeFactor;
    unsigned long long int chunkSizeTiming;
    int configured = 0;

} KernelInfo;


typedef struct _taskinfo
{    
    std::string taskID;
    int traceID;        
    std::vector < int> nodes;
    std::vector <int> dependency;
    std::vector <KernelInfo *>kernels;
    DAGInfo * dagInfo;
    unsigned int task_size;
    float exTime;
    unsigned long long int arrival;
    unsigned long long int deadline;    
    std::vector <float> basespeed;
    //float performance_goal;
    std::vector<void*>  data;
    int isTerminal;
} TaskInfo;


typedef struct _event
{
    cl_event ev;
    std::string type;

    _event(cl_event e,std::string t)
    {
        ev = e;
        type = t;
    }
 
    unsigned long long int get_timestamp(std::string command)
    {
        unsigned long long int timestamp;
        cl_int status;
        if(command == "queued")
        {
            status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timestamp, NULL);
            // print_profile_event_status(status);
            check(status, "Getting event command queued profiling info");
        }
        else if(command == "submit")
        {
            status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timestamp, NULL);
            // print_profile_event_status(status);
            check(status, "Getting event command submit profiling info");
        }
        else if(command == "start")
        {
            status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timestamp, NULL);
            // print_profile_event_status(status);
            check(status, "Getting event command start profiling info");
        }
        else if (command == "end")
        {
            status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timestamp, NULL);
            // print_profile_event_status(status);
            check(status, "Getting event command end profiling info");
        }
        else
        {
            printf("No such event %s exists",command.c_str());
            exit(EXIT_FAILURE);
        }
        return timestamp;
    }

    unsigned long long int get_queue_submit()
    {
        return ( this->get_timestamp("submit") - this->get_timestamp("queued"))/1000;

    }

    unsigned long long int get_submit_start()
    {
        return (this->get_timestamp("start") - this->get_timestamp("submit"))/1000;

    }

    unsigned long long int get_start_end()
    {
        return (this->get_timestamp("end") - this->get_timestamp("start"))/1000;

    }

    void print()
    {
        if(this->type == "write")
            printf("\033[0;32m");
        else if(this->type == "exec")
            printf("\033[1;31m");
        else if(this->type == "read")
            printf("\033[0;34m");
        printf("----------------------------------------------------\n");
        printf("%s Queued -> Submit: %llu\n",this->type.c_str(),this->get_queue_submit());
        printf("%s Submit -> Start: %llu\n",this->type.c_str(),this->get_submit_start());
        printf("%s Start -> End: %llu\n",this->type.c_str(),this->get_start_end());
        printf("----------------------------------------------------\n");
        printf("\033[0m"); 
    }

    void dump()
    {
        if(this->type == "write")
            printf("\033[0;32m");
        else if(this->type == "exec")
            printf("\033[1;31m");
        else if(this->type == "read")
            printf("\033[0;34m");
        printf("%s Queued %llu, ",this->type.c_str(),this->get_timestamp("queued"));
        printf("Submitted %llu, ",this->get_timestamp("submit"));
        printf("Started %llu, ",this->get_timestamp("start"));
        printf("Ended %llu\n",this->get_timestamp("end"));
        printf("\033[0m"); 
    }
} Event;


typedef struct _kernel_events
{
    std::vector<cl_event> write;
    cl_event barrier_write;
    cl_event exec;
    std::vector<cl_event> read;
    cl_event barrier_read;
    int is_profiled;

    void print()
    {
        for(unsigned int i=0;i<write.size();i++)
        {
            std::string type = "write";
            Event *e = new Event(write[i],type);
            e->print();
        }
        std::string type = "execute";
        Event *e = new Event(exec,type);
        e->print();
        for(unsigned int i=0;i<read.size();i++)
        {
            std::string type = "read";
            Event *e = new Event(read[i],type);
            e->print();
        }


    }

    void dump()
    {
        for(unsigned int i=0;i<write.size();i++)
        {
            std::string type = "write";
            Event *e = new Event(write[i],type);
            e->dump();
        }
        std::string type = "execute";
        Event *e = new Event(exec,type);
        e->dump();
        for(unsigned int i=0;i<read.size();i++)
        {
            std::string type = "read";
            Event *e = new Event(read[i],type);
            e->dump();
        }


    }

} KernelEvents;

typedef struct _kernelexecutioninfo{  
    
    unsigned long long int  devStartTime;
    unsigned long long int  devEndTime;
    unsigned long long int  devTotalTime; 
    unsigned long long int write_buffers_start;
    unsigned long long int  rel_start_time; //dispatch start Rtime 
    unsigned long long int  rel_end_time; //notify_callback start Rtime, read end Rtime 
    unsigned long long int  notify_callback_rel_start_time;
    unsigned long long int  notify_callback_rel_end_time;
    unsigned long long int  turnaroundTime; // diff bw rel_end_time and rel_start_time
    unsigned long long int  dispatchTime; // time difference between dispatch start and dispatch end

    unsigned long long int  writeQueued;
    unsigned long long int  writeSubmit;
    unsigned long long int  writeStart;
    unsigned long long int  writeEnd;
    unsigned long long int  writeTime;


    unsigned long long int  ndQueued;
    unsigned long long int  ndSubmit;
    unsigned long long int  ndStart;
    unsigned long long int  ndEnd;
    unsigned long long int  ndTime;


    unsigned long long int  readQueued; 
    unsigned long long int  readSubmit;   
    unsigned long long int  readStart;
    unsigned long long int  readEnd;
    unsigned long long int  readTime; 

    unsigned long long int  write_start_h;
    unsigned long long int  write_start;
    unsigned long long int  write_end;
    unsigned long long int  write_time;
    unsigned long long int  nd_start_h;
    unsigned long long int  nd_start;
    unsigned long long int  nd_end;
    unsigned long long int  nd_time;
    unsigned long long int  read_start_h;
    unsigned long long int  read_start;
    unsigned long long int  read_end;
    unsigned long long int  read_time;

    unsigned long long int frequency_change_time;
    

} KernelExecutionInfo;

typedef struct _kernellaunchinfo{
    TaskInfo *task;
    int kernel_index;
    KernelEvents ke;
    KernelExecutionInfo kex;
    std::vector<cl_mem> io;
    int platform_pos;
    int device_pos;
    int device_index;
    unsigned int not_done_count;
    unsigned int size;
    unsigned int offset;
    int queued;  
    int reset;  
    unsigned int frequency;
    int control_mode;
    int priority;
    unsigned long long int start;
    float expected_speed;
    // float measured_speed;
    float last_speedup;
    
} KernelLaunchInfo;

typedef struct _DAGtime
{
    KernelLaunchInfo* klinfo;
    // int platform;
    // int device;
    unsigned long long int startTime;
    unsigned long long int finishTime;
    unsigned long long int arrivalTime;
    unsigned long long int writeStart;
    unsigned long long int ndStart;
    unsigned long long int readStart;
    float deadline;
    float turnaroundTime;
    float makespan;
    float lateness;
    int safe_mode;
    int deadlineViolated;

}DAGTime;


typedef struct _devConfig{
    
    unsigned int frequency;
    float speedup;
    float powerup;
    
} DeviceConfig;

typedef struct _deviceSpecification{
    
    std::vector<DeviceConfig*> device_config;    
    unsigned int lowFrequencyBound; //= frequencies[0];
    unsigned int highFrequencyBound;
    unsigned int midFrequency; //= frequencies[frequencies.size() - 1]; 
} DeviceSpecification;

typedef struct _temperature{
    
    std::vector<int> sensors;
    int cpuB_0;
    int cpuB_1;
    int cpuB_2;
    int cpuB_3;
    int gpu_4;

} Temperature;

typedef struct _interval{
    unsigned long long int start;
    unsigned long long int end;
    unsigned long long int intervalWidth;
}Interval;


unsigned long long int convert_to_relative_time( unsigned long long int t,unsigned long long int ref);
void dump_profile_event_timing(KernelExecutionInfo kex);
void dump_execution_time_statistics(KernelLaunchInfo *kl,int dag, int task,std::ofstream &ofs);
void create_micro_kernel(int platform);
void strip_ext(char *fname);
std::string getFileName(std::string filePath, bool withExtension, char seperator);
void generate_plot_data(int traceCount);
unsigned int get_buffer_size(KernelInfo* suc_kl,int suc_pos);
std::string get_buffer_type(KernelInfo* suc_kl,int suc_pos);
void trasfer_data_to_ipbuf(KernelLaunchInfo* cur_kl, KernelLaunchInfo* suc_kl,int cur_pos,int suc_pos);
void print_task_map();
void print_job_map();
pair <int, int> dag_to_job_id(int dag);
int job_to_dag_id(int job, int inst);
void populate_global_values(std::map<std::string,std::string>);
void load_config(std::string filename);
void find_max_idle_slot(Interval* intvl_gap,std::vector<std::vector<KernelLaunchInfo*>> &taskList,std::vector<Interval*> &idle_slots);
void choose_task_list(std::vector<std::vector<std::vector<DAGTime>>>  &d_timeMatrix,Interval* global_intvl,std::pair<int, int> max_pair,std::vector<std::vector<std::vector<KernelLaunchInfo*>>> &taskList);
void convert_time_matrix(std::vector<std::vector<DAGTime>> &l_timeMatrix,std::vector<std::vector<std::vector<DAGTime>>> &d_timeMatrix);
void get_idle_slot_list(std::vector<std::vector<DAGTime>> &l_timeMatrix,Interval* global_intvl,std::pair<int,int> max_pair, std::vector<std::vector<Interval*>> &idleSlots);
void shift_task(KernelLaunchInfo* kl);
int find_nearest_config(float s_up, int platform,int device);
int local_controller(KernelLaunchInfo* kl);
float calculate_speed(KernelLaunchInfo* klinfo);
// void choose_task_list(std::vector<std::vector<DAGTime>> &l_timeMatrix,std::vector<Interval*> &global_intvl,std::pair<int, int> max_pair,std::vector<KernelLaunchInfo*> &taskList);
void choose_task_list(std::vector<std::vector<DAGTime>> &l_timeMatrix,Interval* global_intvl,std::pair<int, int> max_pair,std::vector<KernelLaunchInfo*> &taskList);
void get_global_interval(std::vector <std::vector<std::vector<Interval*>>> &intervals_per_tband,std::pair<int, int> max_pair,std::vector<Interval*> &global_intvl);
std::pair<int, int> get_max_band(std::vector <std::vector<std::vector<Interval*>>> &intervals_per_tband);
void get_intervals_per_band(std::map<unsigned long long int,Temperature> &temperatureMap,std::vector <std::vector<std::vector<Interval*>>> &intervals_per_tband);
void choose_intervals();
void reset_launch_info();
void print_DAGTime();
void print_DAGTime(std::vector<std::vector<DAGTime>> &l_dagtime_matrix );
void print_tempMap();
void print_temperature(Temperature* tmp);
void print_temperature(Temperature* tmp);
void monitor_temperature();
void CL_CALLBACK notify_callback_print (cl_event event, cl_int event_command_exec_status, void *user_data);
void childrenToReadyBuffer(KernelLaunchInfo& klinfo);
void build_all_kernel_objects(const char* directory);
void read_dag_structure(const char* directory);
void *taskToReadyB(void *vargp);
const char* parse_file_name(const char* filename);
void read_dag_file(const char* filename);
void populate_task_queue();
void initialise_nodes_matrix();
int task_dev_queue_empty();
int check_dependancy(std::vector <string> deps,int dag);
void create_output_file(const char* op_file);
int read_trace_file(const char* filename);
void *run_scheduler(void *vargp);
void run_scheduler(int taskcount);
int chunk_factor(KernelLaunchInfo* kl);
unsigned long long int get_current_time();
int parse_trace_input(int* index);
void add_task_to_queue(char *ip);
void modify_dag_task_data();
void print_launch_info(KernelLaunchInfo& launchinfo);
void change_frequency(unsigned int frequency, int platform_pos, int device_pos);
void dispatch_from_queue();
int get_dev_index(int platform,int device);
void get_device_specification(char *filename);
//std::vector<DeviceSpecification> get_device_specification(char *filename);
void get_all_devices() ;
cl_uint get_sub_devices(cl_device_id device_id,std::vector< std::vector<cl_device_id> >* all_devices)  ;
void print_device_info(cl_device_id device_id,int i,int j);
void print_all_device_info(std::vector<std::vector<cl_device_id>>& all_devices);
void check(cl_int status, const char* str);
void host_initialize(std::vector<std::vector<cl_device_id>>& all_devices, std::vector<cl_context>& ctxs, std::vector< std::vector<cl_command_queue> >& cmd_qs);
std::vector<cl_command_queue> create_command_queue_for_each(cl_device_id *devs, int num_devs, cl_context ctx);
std::vector<cl_program> build_kernel_from_info(KernelInfo& ki, const char *info_file_name, std::vector<std::vector<cl_device_id>>& all_devices, std::vector<cl_context>& ctxs);
std::vector<cl_program> build_kernel(KernelInfo& ki,  std::vector<std::vector<cl_device_id>>& all_devices, std::vector<cl_context>& ctxs);
cl_program cl_compile_program(const char* kernel_file_name, cl_context ctx,int platform_type);
KernelInfo* assign_kernel_info(const char * info_file_name);
void profile_events(KernelLaunchInfo& kl_info);
int store_variable_kernel_arg(std::string type, std::string value);
void host_array_initialize(KernelInfo& ki, std::vector<void*>& data);
void array_randomize(void* data, std::string type, int size);
void output_initialize(void* data, std::string type, int size);
void* array_allocate(std::string type, int size);
size_t get_sizeof(std::string str);
//void dispatch(KernelLaunchInfo &kl_info,cl_context ctx, cl_command_queue cmd_q, KernelInfo& ki, std::vector<void*>& data, cl_device_id device_id);
cl_event dispatch(KernelLaunchInfo &kl_info);
void update_and_release (KernelLaunchInfo *kl);
unsigned int calculate_ip_buffer_size(unsigned int size,int ip_index,KernelInfo& ki);
unsigned int calculate_ip_buffer_offest(unsigned int dataoffset,int ip_index,KernelInfo& ki);
unsigned int calculate_op_buffer_size(unsigned int size,int op_index,KernelInfo& ki);
unsigned int calculate_op_buffer_offset(unsigned int dataoffset,int op_index,KernelInfo& ki);

void cl_create_buffers(cl_context& ctx, KernelInfo& ki, std::vector<cl_mem>& io, std::vector<void*>& data, unsigned int size,unsigned int dataoffset);
void cl_set_kernel_args(KernelInfo& ki, std::vector<cl_mem>& io, int object,unsigned int datasize) ;
std::vector<cl_event> cl_enqueue_write_buffers(KernelExecutionInfo *di , cl_command_queue cmd_q, KernelInfo& ki, std::vector<cl_mem>& io, std::vector<void*>& data, unsigned int size,unsigned int dataoffset, cl_event dep);
std::vector<cl_event> cl_enqueue_read_buffers(KernelExecutionInfo *di,cl_command_queue cmd_q, KernelInfo& ki, std::vector<cl_mem>& io, std::vector<void*>& data,unsigned int size,unsigned int dataoffset, cl_event dep);
cl_event cl_enqueue_nd_range_kernel(KernelExecutionInfo *di,cl_command_queue cmd_q, KernelInfo& ki, int object,unsigned int size, cl_event dep);
void* get_global_pointer(std::string str, int index);
void CL_CALLBACK notify_callback_update_release (cl_event event, cl_int event_command_exec_status, void *user_data);
void CL_CALLBACK notify_callback_eventstatus (cl_event event, cl_int event_command_exec_status, void *user_data);
void CL_CALLBACK notify_callback_buffer_free (cl_event event, cl_int event_command_exec_status, void *user_data);
void CL_CALLBACK notify_callback_update_exinfo (cl_event event, cl_int event_command_exec_status, void *user_data);
int compare_and_swap(int *word,int testval,int newval);
int test_and_set(int *rqlock,int testval, int newval);
unsigned long long int  gcd(unsigned long long int  a, unsigned long long int  b);
unsigned long long int findlcm(unsigned long long int arr[], int n) ;


void *take_input( void *ptr );
void printEventStatus(cl_event ev);
void printEventStatus(cl_int status, cl_event ev);
// void print_profile_event_status(cl_int status);
void host_synchronize(std::vector< std::vector<cl_command_queue>>& cmd_qs);
void release_host_arrays(std:: vector <void*> &data);
void release_buffers( std::vector<cl_mem>& buffers);
void release_kernel_events(KernelEvents& k);
void release_programs( std::vector<cl_program> all_programs );
void release_everything(std::vector<cl_context>& ctxs, std::vector< std::vector<cl_command_queue> >& cmd_qs);
void reset();
/// Global vectors to store variable kernel argument values.
std::vector<int> gint;
std::vector<uint> guint;
std::vector<short> gshort;
std::vector<ushort> gushort;
std::vector<long> glong;
std::vector<ulong> gulong;
std::vector<bool> gbool;
std::vector<float> gfloat;
std::vector<double> gdouble;
std::vector<char> gchar;





#include "core.inl"

#endif
