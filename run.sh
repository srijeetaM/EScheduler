# remove previous files
#rm -rf ./tinfo/DAG_*/*
rm -rf ./output/*/*
rm -rf trace,src,tinfo_full,dag_history,dag_structure,trace

# copy and set required files

sshpass -p "17cs92p03" scp -r srijeeta@10.5.20.216:/home/srijeeta/Dropbox/Project/VLSID_extension/\{configure_input.txt,speedup_config.csv,src,tinfo_full,dag_history,dag_structure,trace\} .

#python populate_tinfo_folder.py 
sudo service lightdm stop

# Execute

#trace_file dag_history tinfo_file micro_kernel_device
if sudo taskset -c 3-7 ./scheduler trace/DAG_0/profile_dnn_history_30.stats ./dag_history/dag_history_0.stats ./tinfo/DAG_0/node_0:0 1 0;then
	echo "SUCCESSFULLY EXECUTED !!"; 	
else 
	echo "EXECUTION FAILED !!"; 
fi

sshpass -p "17cs92p03" scp -r output srijeeta@10.5.20.216:/home/srijeeta/Dropbox/Project/VLSID_extension/output_archive/
sshpass -p "17cs92p03" scp -r profile_statistics srijeeta@10.5.20.216:/home/srijeeta/Dropbox/Project/VLSID_extension/output_archive/


