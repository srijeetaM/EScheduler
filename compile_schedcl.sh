if g++ schedcl.cpp -O3 -lm -lOpenCL -lpthread -I /usr/include/CL_1_2/ -o schedcl; then 
	echo "COMPILATION SUCCEEDED !!";
else 
	echo "COMPILATION FAILURE !!"; 
fi
