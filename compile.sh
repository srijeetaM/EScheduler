sshpass -p "17cs92p03" scp -r srijeeta@10.5.20.216:/home/srijeeta/Dropbox/Project/VLSID_extension/\{core.h,core.inl,main.cpp,string.h,string.inl,populate_tinfo_folder.py,compile.sh,run.sh\} .

if g++ main.cpp -O3 -lm -lOpenCL -lpthread -I /usr/include/CL_1_2/ -o scheduler; then 
	echo "COMPILATION SUCCEEDED !!";
else 
	echo "COMPILATION FAILURE !!"; 
fi
