__kernel void cnn_5_6_7_8_9( __global const float* b0, __global const float* b2, __global const float* b4, __global const float* b6, __global float* b1, __global float* b3, __global float* b5, __global float* b7, __global float* b8)
{
	int globalId = get_global_id(0);

	typedef float2 floatP;
	floatP data0_P,data1_P;

	int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
	float maxValue;
	bool process;

	if(globalId < 64*4*4)
	{
		localId_P = globalId % (4*4);
		outputRow_P = localId_P / 4;
		outputCol_P = localId_P % 4;
		image2dIdx_P = globalId / (4*4);
		plane_P = image2dIdx_P % 64;
		n_P = image2dIdx_P / 64;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*64 + plane_P)*8*8;
		poolInputOffset = inputImageOffset + inputRow_P * 8 + inputCol_P;
		maxValue = b0[ poolInputOffset ];

		process = (inputRow_P + 1 < 8) && (inputCol_P + 1 < 8);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b0+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b0+poolInputOffset+8);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<4 && outputCol_P<4)
			b1[globalId] = maxValue;

		printf("");
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 512)
	{
		for(int x=0; x<1024/4; x++)
		{
			temp= vload4(0,(__global float *)b1+(4*x));
			wt.x= b2[512*(4*x)+globalId];
			wt.y= b2[512*((4*x)+1)+globalId];
			wt.z= b2[512*((4*x)+2)+globalId];
			wt.w= b2[512*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b3[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < 128)
	{
		for(int x=0; x<512/4; x++)
		{
			temp= vload4(0,(__global float *)b3+(4*x));
			wt.x= b4[128*(4*x)+globalId];
			wt.y= b4[128*((4*x)+1)+globalId];
			wt.z= b4[128*((4*x)+2)+globalId];
			wt.w= b4[128*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b5[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < 16)
	{
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global float *)b5+(4*x));
			wt.x= b6[16*(4*x)+globalId];
			wt.y= b6[16*((4*x)+1)+globalId];
			wt.z= b6[16*((4*x)+2)+globalId];
			wt.w= b6[16*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b7[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int class;
	float sum, max;

	if( globalId==0)
	{
		class=16;
		sum = 0.0;
		max = b7[0];

		for(int i=0;i<class;i++)
			max = (max > b7[i]) ? max : b7[i];

		for(int i=0;i<class;i++)
			b8[i] = exp((b7[i] - max));

		for(int i=0;i<class;i++)
			sum+=b8[i];

		for(int i=0;i<class;i++)
			b8[i] = b8[i]/sum;

		printf("");
	}

}