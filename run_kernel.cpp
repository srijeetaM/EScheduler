#include "core.inl"
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
    pthread_t scheduler = pthread_self(); 
    rc = pthread_attr_init (&attr);
    rc = pthread_attr_getschedparam (&attr, &param);
    
    param.sched_priority = sched_get_priority_max(SCHED_RR);
    pthread_setschedparam(scheduler,SCHED_RR,&param);
    rc = pthread_attr_setschedparam (&attr, &param);

    pthread_t  thread_controller;

    try
    {
         
        reset_launch_info();
        printf("reset_launch_info\n");
        nTasks=0; 
        SchedulerFinish=0;   
        pthread_t thread_temperature_monitor,thread_scheduler;
        printf("\nStartTime run_kernel: %llu\n",get_current_time());     

        struct timeval c_time;
        gettimeofday(&c_time,NULL); 
        START_TIME=(unsigned long long int )(c_time.tv_sec*1000000+c_time.tv_usec); 
        cout << "\tTrying for DAG ID " << taskMap.begin()->first.first << '\t' << "Kernel ID " << taskMap.begin()->first.second << "KernelObject: "<<taskMap.begin()->second->kernel_index<<'\n';
        
        if(monitorTemp==1)
          pthread_create(&thread_temperature_monitor, NULL, temperature_monitor, NULL);          
        pthread_create(&thread_scheduler, &attr, run_kernel, &traceCount);  
                
        
        pthread_join(thread_scheduler, NULL);
        printf("Thread Join thread scheduler: %llu\n",get_current_time());
                
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
    catch (std::exception& e)
    {
        std::cerr << "Exception caught : " << e.what() << std::endl;
    }
    

    
    if(controlerTemp==1)
      pthread_join(thread_controller, NULL);    
        
    // /*Release objects*/    
    // for(int i=0;i<task_queue.size();i++)
    //     release_host_arrays(task_queue[i].data); 
    // std::ofstream ofs;
    // ofs.open(argv[2],std::ios_base::app);
    // printf("-------------------------Execution Statistics--------------------------------------\n");
    // // printf("dag \t\t task \t\t w_delay \t\t w \t\t e_delay \t\t e \t\t r_delay \t\t r \t\t k_ex \t\t h_ex \t\t h_over \t\t cb \t\t cb_over\n");
    // for (auto itr = taskMap.begin(); itr != taskMap.end(); ++itr) { 
    //   cout << "\tDAG ID " << itr->first.first << '\t' << "Kernel ID " << itr->first.second << '\n'; 
    //   dump_execution_time_statistics(itr->second,itr->first.first,itr->first.second,ofs);
    
    // }
    // ofs.close(); 
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
