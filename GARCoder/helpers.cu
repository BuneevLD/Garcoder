#pragma once

#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DataModels.cu"
#include "_Constants.h"

__global__ void prefixSum
(unsigned char* data,
unsigned* oSizes,
unsigned const size,
unsigned const n,
unsigned* overallSize);

__global__ void encodeKernel
(unsigned const bit_0_prob,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize);

__global__ void encodeKernelAdaptive
(AdaptiveBinaryModel model,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize);

__global__ void encodeKernelContextAdaptive
(unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize);


//Both oData and iData are on the device already.
//Therefore, several kernels dont copy data between host and device memory.
void launchBasicEncodeKernel
	(unsigned const bit_0_prob,
	unsigned char* iData,
	unsigned const iSize,
	unsigned char* oData,
	unsigned* oChunkSizes,
	unsigned chunksCount,
	unsigned const chunkSize)
{
	// Fast division rounding up
	unsigned blocksCount = (chunksCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	encodeKernel <<<blocksCount, THREADS_PER_BLOCK >>>(bit_0_prob, iData, iSize, oData, oChunkSizes, chunkSize);
}

void launchAdaptiveEncodeKernel
(AdaptiveBinaryModel model,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize)
{
	unsigned blocksCount = (chunksCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	encodeKernelAdaptive <<<blocksCount, THREADS_PER_BLOCK >>>(model, iData, iSize, oData, oChunkSizes, chunkSize);
}

void launchContextAdaptiveEncodeKernel
(unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize)
{
	unsigned blocksCount = (chunksCount + 32 - 1) / 32;
	encodeKernelContextAdaptive <<<blocksCount, 32>>>(iData, iSize, oData, oChunkSizes, chunkSize);
}


//Doesn't work
void launchPrefixSumEncodeKernel
	(unsigned const bit_0_prob,
	unsigned char* iData,
	unsigned const iSize,
	unsigned char* oData,
	unsigned* oChunkSizes,
	unsigned chunksCount,
	unsigned const chunkSize,
	unsigned* overallSize)
{
	encodeKernel <<<chunksCount/8, 8 >>>(bit_0_prob, iData, iSize, oData, oChunkSizes, chunkSize);
	cudaDeviceSynchronize();
	prefixSum <<<1, chunksCount / 2, chunksCount * 2 * sizeof(unsigned)>>>(oData, oChunkSizes, chunkSize + 2, chunksCount, overallSize);
	//prefixSum<<<1, chunksCount/2, chunksCount * 2 * sizeof(unsigned) >>>(oData, oChunkSizes, chunkSize + 2, chunksCount, overallSize);
}