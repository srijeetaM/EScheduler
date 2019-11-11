__kernel void cnn_8_9( __global const float* b0, __global const float* b1, __global float* b2, __global float* b3)
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
			wt.x= b1[16*(4*x)+globalId];
			wt.y= b1[16*((4*x)+1)+globalId];
			wt.z= b1[16*((4*x)+2)+globalId];
			wt.w= b1[16*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
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