if g++ scheduler.cpp -O3 -lm -lOpenCL -lpthread -I /usr/include/CL_1_2/ -o scheduler; then 
	echo "COMPILATION SUCCEEDED !!";
else 
	echo "COMPILATION FAILURE !!"; 
fi
