sshpass -p "17cs92p03" scp -r srijeeta@10.5.20.216:/home/srijeeta/Project_new/Odroid/Scheduler_Profiling/\{scheduler.cpp,functionalities.*,string.*\} .

if g++ scheduler.cpp -O3 -lm -lOpenCL -lpthread -I /usr/include/CL_1_2/ -o scheduler; then 
	echo "COMPILATION SUCCEEDED !!";
else 
	echo "COMPILATION FAILURE !!"; 
fi
