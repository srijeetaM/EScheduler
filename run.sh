#rm -rf ./tinfo
rm -rf ./output/log/*
rm -rf ./output/task_set/*
rm -rf ./output/temperature/*
rm -rf ./output/time/*
rm -rf ./output/timestamp/*

#sshpass -p "17cs92p03" scp -r srijeeta@10.5.20.216:/home/srijeeta/Project_new/Odroid/Scheduler_Profiling/\{configure_input.txt,gmm512_config.csv,src,tinfo,dag_history,dag_structure,trace\} .

if sudo taskset -c 3-7 ./scheduler trace/dispatch_history_0.stats;then
	echo "SUCCESSFULLY EXECUTED !!"; 
#	sshpass -p "17cs92p03" scp -r output srijeeta@10.5.20.216:/home/srijeeta/Project_new/Odroid/Scheduler_Profiling/output_archive/
else 
echo "EXECUTION FAILED !!"; 
fi
