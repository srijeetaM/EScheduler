__kernel void cnn_1( __global const float* b0, __global float* b1)
{
	int globalId = get_global_id(0);

	typedef float2 floatP;
	floatP data0_P,data1_P;

	int localId_P, outputRow_P, outputCol_P, image2dIdx_P, plane_P, n_P, pool_size, inputRow_P, inputCol_P, inputImageOffset, poolInputOffset;
	float maxValue;
	bool process;

	if(globalId < 16*16*16)
	{
		localId_P = globalId % (16*16);
		outputRow_P = localId_P / 16;
		outputCol_P = localId_P % 16;
		image2dIdx_P = globalId / (16*16);
		plane_P = image2dIdx_P % 16;
		n_P = image2dIdx_P / 16;
		if(n_P > 1) 
			return;
		
		pool_size = 2;
		inputRow_P = outputRow_P * pool_size;
		inputCol_P = outputCol_P * pool_size;
		inputImageOffset  = (n_P*16 + plane_P)*32*32;
		poolInputOffset = inputImageOffset + inputRow_P * 32 + inputCol_P;
		maxValue = b0[ poolInputOffset ];

		process = (inputRow_P + 1 < 32) && (inputCol_P + 1 < 32);

		if(process)
		{
			data0_P = vload2(0,(__global const float *)b0+poolInputOffset);
			data1_P = vload2(0,(__global const float *)b0+poolInputOffset+32);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<16 && outputCol_P<16)
			b1[globalId] = maxValue;

		printf("");
	}
}