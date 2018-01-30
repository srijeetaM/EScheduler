__kernel void cnn_4( __global const float* b0, __global const float* b1, __global float* b2)
{
	int globalId = get_global_id(0);

	typedef float4 floatX;
	floatX wt,temp;
	float dotProduct;

	dotProduct=0.0;
	if(globalId < 256)
	{
		for(int x=0; x<512/4; x++)
		{
			temp= vload4(0,(__global const float *)b0+(4*x));
			wt.x= b1[256*(4*x)+globalId];
			wt.y= b1[256*((4*x)+1)+globalId];
			wt.z= b1[256*((4*x)+2)+globalId];
			wt.w= b1[256*((4*x)+3)+globalId];
			dotProduct += dot(wt,temp);
		}
b2[globalId] = dotProduct;
		printf("");
	}

}