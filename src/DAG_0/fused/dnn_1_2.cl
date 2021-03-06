#define v0 4096
#define v1 2048
#define v2 2048
#define v3 1024

__kernel void dnn_1_2( __global const float* b0, __global const float* b1, __global const float* b2, __global const float* b3, __global const float* b4, __global float* b5, __global float* b6)
{
	int globalId = get_global_id(0);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;


	dotProduct=0.0;
	if(globalId < v1)
	{
		for(int x=0; x<v0/4; x++)
		{
			temp= vload4(0,(__global const float *)b0+(4*x));
			wt= vload4(0,(__global const float *)b1+(globalId*v0+4*x));
			dotProduct += dot(wt,temp);
		}
		b5[globalId] = dotProduct+b2[globalId];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < v3)
	{
		for(int x=0; x<v2/4; x++)
		{
			temp= vload4(0,(__global float *)b5+(4*x));
			wt= vload4(0,(__global const float *)b3+(globalId*v2+4*x));
			dotProduct += dot(wt,temp);
		}
		b6[globalId] = dotProduct+b4[globalId];
	}

}