__kernel void cnn_7_8( __global const float* b0, __global const float* b1, __global const float* b3, __global float* b2, __global float* b4)
{
	int globalId = get_global_id(0);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 128)
	{
		for(int x=0; x<512/4; x++)
		{
			temp= vload4(0,(__global const float *)b0+(4*x));
			wt.x= b1[128*(4*x)+globalId];
			wt.y= b1[128*((4*x)+1)+globalId];
			wt.z= b1[128*((4*x)+2)+globalId];
			wt.w= b1[128*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b2[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < 16)
	{
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global float *)b2+(4*x));
			wt.x= b3[16*(4*x)+globalId];
			wt.y= b3[16*((4*x)+1)+globalId];
			wt.z= b3[16*((4*x)+2)+globalId];
			wt.w= b3[16*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b4[globalId] = dotProduct;
		printf("");
	}

}