#include "functionalities.h"

int main(int argc, char const *argv[])
{	

    load_config("./configure_input.txt");
    const char* tok =parse_file_name(argv[1]);
    char argv2[100];
    strcpy(argv2,"dag_history/dag_history_"); 
    strcat(argv2,tok);
    strcat(argv2,".stats");
    printf("\nInput read from file: %s %s\n",argv[1],argv2);
    read_dag_file(argv2); 
    // printf("Input read_dag_file done\n");
    read_dag_structure("./dag_structure/");
    // printf("Input read_dag_structure done\n");
    /*create Output dump file*/
    create_output_file(argv[1]);
      
      /* Populate all deviceIds in a vector of vector, command queues and context for each platform*/
    get_all_devices();
    host_initialize(all_devices, all_ctxs, all_cmd_qs);
    get_device_specification("gmm512_config.csv"); 

    /* Prit details of all devices*/
    print_all_device_info(all_devices);
    fflush(fp);     

    build_all_kernel_objects("./tinfo/");
    // printf("build_all_kernel_objects \n");

    /*get input file for trace*/
    int traceCount=read_trace_file(argv[1]);
    printf("\ntraceCount %d \n",traceCount);
    populate_task_queue();
    // printf("populate_task_queue DONE\n");
    // print_task_map();  

    if(micro_kernel_device!=-1)
      create_micro_kernel(micro_kernel_device);

    /*Start scheduler*/     
    
    int rc;
    pthread_attr_t attr;
    struct sched_param param;
    rc = pthread_attr_init (&attr);
    rc = pthread_attr_getschedparam (&attr, &param);
    (param.sched_priority)++;
    rc = pthread_attr_setschedparam (&attr, &param);

    pthread_t  thread_controller;

    try
    {
      for(int i=0;i<numOfHyperperiod;i++)    
      {   
          printf("in hyperperiod: %llu\n",hyper_period);

          // if(numOfDAGs!=count_dag_from_file(argv2))
          // {
          //     printf("New task DAG added! \n");
          //     traceCount=modify_dag_task_data(argv2,argv[1],traceCount);
          // }
          
    
          reset_launch_info();
          printf("reset_launch_info\n");
          nTasks=0; 
          SchedulerFinish=0;   
                   
          pthread_t thread_taskQtoReadyB, thread_temperature_monitor,thread_scheduler;
          printf("\nStartTime run_scheduler: %llu\n",get_current_time());     

          struct timeval c_time;
          gettimeofday(&c_time,NULL); 
          START_TIME=(unsigned long long int )(c_time.tv_sec*1000000+c_time.tv_usec); 
          pthread_create(&thread_taskQtoReadyB, NULL, taskToReadyB, &traceCount);        
          if(monitorTemp==1)
            pthread_create(&thread_temperature_monitor, NULL, temperature_monitor, NULL);          
          pthread_create(&thread_scheduler, &attr, run_scheduler, &traceCount);  
          
          pthread_join(thread_taskQtoReadyB, NULL); 
          printf("Thread Join1: %llu\n",get_current_time());
          pthread_join(thread_scheduler, NULL);
          printf("Thread Join2: %llu\n",get_current_time());

          host_synchronize(all_cmd_qs);
          printf("Host Synchronised: %d\n",nKernels);
          SchedulerFinish=1; 
          if(monitorTemp==1)       
          { 
            pthread_join(thread_temperature_monitor, NULL); 
            print_tempMap();
          }
          if(controlerTemp==1)
          {  
            int mode=MODE;
            pthread_create(&thread_controller, NULL, mode_controller,&mode ); 
          }
          if(generatePlot==1)
            generate_plot_data(traceCount);
                    
      }
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception caught : " << e.what() << std::endl;
    }
    
    printf("Exit Hyperperiods\n");    
    
    if(controlerTemp==1)
      pthread_join(thread_controller, NULL);    
        
    // /*Release objects*/    
    // for(int i=0;i<task_queue.size();i++)
    //     release_host_arrays(task_queue[i].data); 
    
    printf("-------------------------Execution Statistics--------------------------------------\n");
    printf("dag \t\t task \t\t w_delay \t\t w \t\t e_delay \t\t e \t\t r_delay \t\t r \t\t k_ex \t\t h_ex \t\t h_over \t\t cb \t\t cb_over\n");
    for (auto itr = taskMap.begin(); itr != taskMap.end(); ++itr) { 
      // cout << "\tDAG ID " << itr->first.first << '\t' << "Kernel ID " << itr->first.second << '\n'; 
      dump_execution_time_statistics(itr->second->kex,itr->first.first,itr->first.second);
    
    } 
    release_everything(all_ctxs, all_cmd_qs);
    printf("released_everything\n");

    fprintf(fp,"\n");
    fclose (fp); 
    fclose (t_result); 
    fclose (tmp_result);
    fclose (m_result);
    printf("\n");
    printf("FINISHED!\n\n");    
	  return 0;
}
