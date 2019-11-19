if g++ run_kernel.cpp -O3 -lm -lOpenCL -lpthread -I /usr/include/CL_1_2/ -o run_kernel; then 
	echo "COMPILATION SUCCEEDED !!";
else 
	echo "COMPILATION FAILURE !!"; 
fi
