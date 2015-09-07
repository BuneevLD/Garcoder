#pragma once

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__device__ void memCopy(unsigned char *destination, unsigned char *source, int size) {
	unsigned char *dest = destination;
	unsigned char *src = source;
	for (int tid = threadIdx.x; tid<size; tid += blockDim.x)
		dest[tid] = src[tid];
}

//Parallel reduce algorithm based on GPU Gems 3 - Chapter 39
__global__ void prefixSum(unsigned char* data, unsigned* oSizes, unsigned const size, unsigned const n, unsigned* overallSize)
{
	//*overallSize = 15;
	extern __shared__ unsigned temp[];  // allocated on invocation  
	int thid = threadIdx.x;
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int offset = 1;

	unsigned* temp2 = &(temp[n]);

	temp2[2 * thid] = temp[2 * thid] = oSizes[2 * thid]; // load input into shared memory  
	temp2[2 * thid + 1] = temp[2 * thid + 1] = oSizes[2 * thid + 1];

	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element 

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	data[2*id] = temp[2*id]; // write results to device memory  
	data[2*id+1] = temp[2*id+1];

	/*for (int i = 0; i < n; ++i)
	{
		//if(thid == 0)
		//printf("%d", temp[i]);
		memCopy(data + temp[i], data + size*i, temp2[i]);
	}*/
	*overallSize = temp[0] = temp[n - 1] + temp2[n - 1];
}