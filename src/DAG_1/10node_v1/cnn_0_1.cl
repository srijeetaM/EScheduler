__kernel void cnn_0_1( __global const float* b0, __global const float* b1, __global const float* b2, __global float* b3, __global float* b4)
{
	int globalId = get_global_id(0);

	typedef float4 floatC;
	floatC data0_C,data1_C;
	int localId_C, w2 , h2, outputRow_C, outputCol_C, image2dIdx_C, plane_C, n_C, stride, pad, inputRow_C, inputCol_C;
	float dots;

	if(globalId < 16*32*32)
	{
		w2 = 32;
		h2 = 32;

		localId_C = globalId%(w2*h2);
		outputRow_C = localId_C / w2;
		outputCol_C = localId_C % w2;

		image2dIdx_C = globalId / (h2*w2);
		plane_C = image2dIdx_C % 16;
		n_C = image2dIdx_C / 16;
		dots = 0.0;
		stride = 1;
		pad=1;
		inputRow_C = outputRow_C * stride - pad;
		inputCol_C = outputCol_C * stride - pad;

			for(int j=0;j<3;j++)
			{
				for(int i=0;i<3;i++)
				{
					if(inputRow_C+i<0 || inputRow_C+i>=32)
						data0_C.xyz = (float3)(0.0f,0.0f,0.0f);
					else if(inputCol_C<0)
					{
						if(inputRow_C+i==0 && j==0)
						{
							data0_C.y = b0[0];
							data0_C.z = b0[1];
						}
						else
							data0_C.xyz = vload3(0,(__global const float *)b0+(j*32*32)+(inputRow_C+i)*32+inputCol_C);
						data0_C.x = 0;
					}
					else if(inputCol_C+3-1>=32)
					{
						if(inputRow_C+i==32 && j==2)
						{
							data0_C.x = b0[(j*32*32)+(inputRow_C+i)*32+2];
							data0_C.y = b0[(j*32*32)+(inputRow_C+i)*32+3];
						}
						else
						data0_C.xyz = vload3(0,(__global const float *)b0+(j*32*32)+(inputRow_C+i)*32+inputCol_C);
						data0_C.z = 0;
					}
					else
					{
						data0_C.xyz = vload3(0,(__global const float *)b0+(j*32*32)+(inputRow_C+i)*32+inputCol_C);
					}
					data0_C.w = 0.0;
					data1_C.xyz = vload3(0,(__global const float *)b1+(plane_C*3*3*3)+(j*3*3)+i*3);
					dots += dot(data0_C,data1_C);
				}
			}
			if(outputRow_C<32 && outputCol_C<32)
				b3[plane_C*32*32+outputRow_C*32+outputCol_C] = dots + b2[plane_C];
		}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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
		maxValue = b3[ poolInputOffset ];

		process = (inputRow_P + 1 < 32) && (inputCol_P + 1 < 32);

		if(process)
		{
			data0_P = vload2(0,(__global float *)b3+poolInputOffset);
			data1_P = vload2(0,(__global float *)b3+poolInputOffset+32);

			data0_P = fmax(data0_P,data1_P);
			maxValue = fmax(data0_P.s0,data0_P.s1);
		}

		if(outputRow_P<16 && outputCol_P<16)
			b4[globalId] = maxValue;

		printf("");
	}
}