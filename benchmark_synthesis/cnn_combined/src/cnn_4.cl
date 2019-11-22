__kernel void cnn_4( __global const float* b0, __global const float* b1, __global float* b2)
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

}