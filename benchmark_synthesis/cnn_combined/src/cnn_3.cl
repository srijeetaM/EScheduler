__kernel void cnn_3( __global const float* b0, __global const float* b1, __global const float* b3, __global float* b2, __global float* b4)
{
	int globalId = get_global_id(0);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 512)
	{
		for(int x=0; x<1024/4; x++)
		{
			temp= vload4(0,(__global const float *)b0+(4*x));
			wt= vload4(0,(__global const float *)b1+(globalId*1024+4*x));
			dotProduct += dot(wt,temp);
/*			dotProduct += wt.x*temp.x + wt.y*temp.y + wt.z*temp.z + wt.w*temp.w;*/
		}
		b2[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < 128)
	{
		for(int x=0; x<512/4; x++)
		{
			temp= vload4(0,(__global float *)b2+(4*x));
			wt= vload4(0,(__global const float *)b3+(globalId*512+4*x));
			dotProduct += dot(wt,temp);
/*			dotProduct += wt.x*temp.x + wt.y*temp.y + wt.z*temp.z + wt.w*temp.w;*/
		}
		b4[globalId] = dotProduct;
		printf("");
	}

}