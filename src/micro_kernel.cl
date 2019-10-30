#define var 8192
#define i_max 500
#define j_max 500

__kernel void micro_kernel(__global int* in_data_1, __global int* in_data_2, __global int* out_data)
{
	int tid = get_global_id(0);	
	int data_tmp = 0;
	int data_1,data_2;

	if(tid < var)
	{
		for(int i=0; i<i_max; i++)
		{
			//Step 1: read data (memory related)
	 		data_1 = in_data_1[tid];
			data_2 = in_data_2[tid];
		
		 	//Step 2: do calculation (memory unrelated)
			for(int j=0; j<j_max; j++)
			{
				data_tmp += j;
				data_tmp = data_tmp % 10000;
			}
		
			//Step 3:  write back (memory related)
			out_data[tid] = data_1 + data_2*data_tmp;
		
		}     
	}       
} 