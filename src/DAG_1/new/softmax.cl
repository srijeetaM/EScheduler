__kernel void cnn_9( __global const float* b0, __global float* b1)
{
	int globalId = get_global_id(0);

	int class;
	float sum, max;

	class=16;
	sum = 0.0;
	max = b0[0];

	if( globalId==0)
	{
		for(int i=0;i<class;i++)
			max = (max > b0[i]) ? max : b0[i];

		for(int i=0;i<class;i++)
			b1[i] = exp((b0[i] - max));

		for(int i=0;i<class;i++)
			sum+=b1[i];

		for(int i=0;i<class;i++)
			b1[i] = b1[i]/sum;

	}

}