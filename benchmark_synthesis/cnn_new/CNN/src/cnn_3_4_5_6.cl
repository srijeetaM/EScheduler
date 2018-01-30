__kernel void cnn_3_4_5_6( __global const float* b0, __global const float* b3, __global const float* b5, __global float* b1, __global float* b2, __global float* b4, __global float* b6)
{
	int globalId = get_global_id(0);

	typedef float2 floatP;
	floatP data0_P,data1_P;

	int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
	float maxValue;
	bool process;

	if(globalId < 32*8*8)
	{
		localId_P = globalId % (8*8);
		outputRow_P = localId_P / 8;
		outputCol_P = localId_P % 8;
		image2dIdx_P = globalId / (8*8);
		plane_P = image2dIdx_P % 32;
		n_P = image2dIdx_P / 32;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*32 + plane_P)*16*16;
		poolInputOffset = inputImageOffset + inputRow_P * 16 + inputCol_P;
		maxValue = b0[ poolInputOffset ];

		process = (inputRow_P + 1 < 16) && (inputCol_P + 1 < 16);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b0+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b0+poolInputOffset+16);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<8 && outputCol_P<8)
			b1[globalId] = maxValue;

		printf("");
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if(globalId < 32*8*8)
	{
		localId_P = globalId % (8*8);
		outputRow_P = localId_P / 8;
		outputCol_P = localId_P % 8;
		image2dIdx_P = globalId / (8*8);
		plane_P = image2dIdx_P % 32;
		n_P = image2dIdx_P / 32;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*32 + plane_P)*16*16;
		poolInputOffset = inputImageOffset + inputRow_P * 16 + inputCol_P;
		maxValue = b1[ poolInputOffset ];

		process = (inputRow_P + 1 < 16) && (inputCol_P + 1 < 16);

		if(process)
		{
			data0_P = vload2(0,(__global float *)b1+poolInputOffset);
			data1_P = vload2(0,(__global float *)b1+poolInputOffset+16);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<8 && outputCol_P<8)
			b2[globalId] = maxValue;

		printf("");
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 512)
	{
		for(int x=0; x<2048/4; x++)
		{
			temp= vload4(0,(__global float *)b2+(4*x));
			wt.x= b3[512*(4*x)+globalId];
			wt.y= b3[512*((4*x)+1)+globalId];
			wt.z= b3[512*((4*x)+2)+globalId];
			wt.w= b3[512*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b4[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	dotProduct=0.0;
	if(globalId < 32)
	{
		for(int x=0; x<512/4; x++)
		{
			temp= vload4(0,(__global float *)b4+(4*x));
			wt.x= b5[32*(4*x)+globalId];
			wt.y= b5[32*((4*x)+1)+globalId];
			wt.z= b5[32*((4*x)+2)+globalId];
			wt.w= b5[32*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b6[globalId] = dotProduct;
		printf("");
	}

}