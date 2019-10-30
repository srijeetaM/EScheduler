#define v0 4096
#define v1 4096
#define v2 4096
#define v3 2048
#define v4 2048
#define v5 1024
#define v6 1024
#define v7 512
#define v8 512
#define v9 16

__kernel void dnn_0_1_2_3_4( __global const float* b0, __global const float* b1, __global const float* b2, __global const float* b3, __global const float* b4, __global const float* b5, __global const float* b6, __global const float* b7, __global const float* b8, __global const float* b9, __global const float* b10, __global float* b11, __global float* b12, __global float* b13, __global float* b14, __global float* b15)
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
		b11[globalId] = dotProduct+b2[globalId];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < v3)
	{
		for(int x=0; x<v2/4; x++)
		{
			temp= vload4(0,(__global float *)b11+(4*x));
			wt= vload4(0,(__global const float *)b3+(globalId*v2+4*x));
			dotProduct += dot(wt,temp);
		}
		b12[globalId] = dotProduct+b4[globalId];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < v5)
	{
		for(int x=0; x<v4/4; x++)
		{
			temp= vload4(0,(__global float *)b12+(4*x));
			wt= vload4(0,(__global const float *)b5+(globalId*v4+4*x));
			dotProduct += dot(wt,temp);
		}
		b13[globalId] = dotProduct+b6[globalId];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < v7)
	{
		for(int x=0; x<v6/4; x++)
		{
			temp= vload4(0,(__global float *)b13+(4*x));
			wt= vload4(0,(__global const float *)b7+(globalId*v6+4*x));
			dotProduct += dot(wt,temp);
		}
		b14[globalId] = dotProduct+b8[globalId];
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < v9)
	{
		for(int x=0; x<v8/4; x++)
		{
			temp= vload4(0,(__global float *)b14+(4*x));
			wt= vload4(0,(__global const float *)b9+(globalId*v8+4*x));
			dotProduct += dot(wt,temp);
		}
		b15[globalId] = dotProduct+b10[globalId];
	}

}