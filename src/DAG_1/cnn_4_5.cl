__kernel void cnn_4_5( __global const float* b0, __global const float* b1, __global float* b2, __global float* b3)
{
	int globalId = get_global_id(0);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 16)
	{
		for(int x=0; x<128/4; x++)
		{
			temp= vload4(0,(__global const float *)b0+(4*x));
			wt= vload4(0,(__global const float *)b1+(globalId*128+4*x));
			dotProduct += dot(wt,temp);
/*			dotProduct += wt.x*temp.x + wt.y*temp.y + wt.z*temp.z + wt.w*temp.w;*/
		}
		b2[globalId] = dotProduct;
		printf("");
	}


	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int class;
	float sum, max;

	if( globalId==0)
	{
		class=16;
		sum = 0.0;
		max = b2[0];

		for(int i=0;i<class;i++)
			max = (max > b2[i]) ? max : b2[i];

		for(int i=0;i<class;i++)
			b3[i] = exp((b2[i] - max));

		for(int i=0;i<class;i++)
			sum+=b3[i];

		for(int i=0;i<class;i++)
			b3[i] = b3[i]/sum;

		printf("");
	}

}