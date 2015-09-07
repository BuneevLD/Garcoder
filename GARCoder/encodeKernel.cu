#pragma once
#include "_Constants.h"
#include "DataModels.cu"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

//const unsigned stepSize = 32;

__global__ void encodeKernel
(unsigned const bit_0_prob,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize)
{
	int chunkId = blockIdx.x*blockDim.x + threadIdx.x;
	iData += chunkId*chunkSize;
	oData += chunkId*(chunkSize + 2);

	unsigned char* ac_pointer = oData + 2;
	unsigned length = AC_MaxLength;
	unsigned base = 0;
	if (chunkId*chunkSize < iSize) {
		unsigned sizeToProcess = min(chunkSize, iSize - chunkId*chunkSize);
		unsigned char* end = ac_pointer + sizeToProcess;

		for (unsigned i = 0; i < sizeToProcess; ++i)
		{
			unsigned char byte = iData[i];
			for (unsigned j = 0; j < 8; ++j)
			{
				unsigned x = bit_0_prob * (length >> BM_LengthShift);
				if ((byte & (1 << j)) == 0)
					length = x;
				else
				{
					unsigned init_base = base;
					base += x;
					length -= x;
					if (init_base > base) //Overflow? Carry.
					{
						unsigned char * p;
						for (p = ac_pointer - 1; *p == 0xFFU; p--)
							*p = 0;
						++*p;
					}
				}
				if (length < AC_MinLength) //Renormalization
				{
					do
					{
						if (ac_pointer >= end)
							break;
						*ac_pointer++ = (unsigned char)(base >> 24);
						base <<= 8;
					} while ((length <<= 8) < AC_MinLength);
				}
			}
		}
		if (ac_pointer < end) {
			unsigned init_base = base;
			if (length > 2 * AC_MinLength) {
				base += AC_MinLength;
				length = AC_MinLength >> 1;
			}
			else {
				base += AC_MinLength >> 1;
				length = AC_MinLength >> 9;
			}

			if (init_base > base)
			{
				unsigned char * p;
				for (p = ac_pointer - 1; *p == 0xFFU; p--)
					*p = 0;
				++*p;
			}
			do
			{
				if (ac_pointer >= end)
					break;
				*ac_pointer++ = (unsigned char)(base >> 24);
				base <<= 8;

			} while ((length <<= 8) < AC_MinLength);
		}

		unsigned codeBytes = 0;
		if (ac_pointer < end) {
			codeBytes = unsigned(ac_pointer - oData);
		}

		oData[0] = codeBytes >> 8;
		oData[1] = codeBytes;
		oChunksSizes[chunkId] = codeBytes;
	}
}

__global__ void encodeKernelAdaptive
(AdaptiveBinaryModel model,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize)
{
	int chunkId = blockIdx.x*blockDim.x + threadIdx.x;
	iData += chunkId*chunkSize;
	oData += chunkId*(chunkSize + 2);

	unsigned char* ac_pointer = oData + 2;
	unsigned length = AC_MaxLength;
	unsigned base = 0;
	if (chunkId*chunkSize < iSize) {
		unsigned sizeToProcess = min(chunkSize, iSize - chunkId*chunkSize);
		unsigned char* end = ac_pointer + sizeToProcess;

		for (unsigned i = 0; i < sizeToProcess; ++i)
		{
			unsigned char byte = iData[i];
			for (unsigned j = 0; j < 8; ++j)
			{
				unsigned x = model.bit0Prob * (length >> BM_LengthShift);
				if ((byte & (1 << j)) == 0) {
					length = x;
					++model.bit0Count;
				}
				else
				{
					unsigned init_base = base;
					base += x;
					length -= x;
					if (init_base > base) //Overflow? Carry.
					{
						unsigned char * p;
						for (p = ac_pointer - 1; *p == 0xFFU; p--)
							*p = 0;
						++*p;
					}
				}
				if (length < AC_MinLength) //Renormalization
				{
					do
					{
						if (ac_pointer >= end)
							break;
						*ac_pointer++ = (unsigned char)(base >> 24);
						base <<= 8;
					} while ((length <<= 8) < AC_MinLength);
				}
				if (--model.bitsUntilUpdate == 0) model.update();
			}
		}
		if (ac_pointer < end) {
			unsigned init_base = base;
			if (length > 2 * AC_MinLength) {
				base += AC_MinLength;
				length = AC_MinLength >> 1;
			}
			else {
				base += AC_MinLength >> 1;
				length = AC_MinLength >> 9;
			}

			if (init_base > base)
			{
				unsigned char * p;
				for (p = ac_pointer - 1; *p == 0xFFU; p--)
					*p = 0;
				++*p;
			}
			do
			{
				if (ac_pointer >= end)
					break;
				*ac_pointer++ = (unsigned char)(base >> 24);
				base <<= 8;

			} while ((length <<= 8) < AC_MinLength);
		}

		unsigned codeBytes = 0;
		if (ac_pointer < end) {
			codeBytes = unsigned(ac_pointer - oData);
		}

		oData[0] = codeBytes >> 8;
		oData[1] = codeBytes;
		oChunksSizes[chunkId] = codeBytes;
	}
}

__global__ void encodeKernelContextAdaptive
(unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunksSizes,
unsigned const chunkSize)
{
	ContextAdaptiveBinaryModel model;

	int chunkId = blockIdx.x*blockDim.x + threadIdx.x;
	iData += chunkId*chunkSize;
	oData += chunkId*(chunkSize + 2);

	unsigned char* ac_pointer = oData + 2;
	unsigned length = AC_MaxLength;
	unsigned base = 0;
	if (chunkId*chunkSize < iSize) {
		unsigned sizeToProcess = min(chunkSize, iSize - chunkId*chunkSize);
		unsigned char* end = ac_pointer + sizeToProcess;

		unsigned char prefix = 0;
		for (unsigned i = 0; i < sizeToProcess; ++i)
		{
			unsigned char byte = iData[i];
			for (unsigned j = 0; j < 8; ++j)
			{
				unsigned x = model.models[prefix].bit0Prob * (length >> BM_LengthShift);
				if ((byte & (1 << j)) == 0) {
					length = x;
					++model.models[prefix].bit0Count;
					if (--model.models[prefix].bitsUntilUpdate == 0) model.models[prefix].update();
					prefix <<= 1;
				}
				else
				{
					unsigned init_base = base;
					base += x;
					length -= x;
					if (init_base > base) //Overflow? Carry.
					{
						unsigned char * p;
						for (p = ac_pointer - 1; *p == 0xFFU; p--)
							*p = 0;
						++*p;
					}
					if (--model.models[prefix].bitsUntilUpdate == 0) model.models[prefix].update();
					prefix <<= 1;
					prefix |= 1;
				}
				if (length < AC_MinLength) //Renormalization
				{
					do
					{
						if (ac_pointer >= end)
							break;
						*ac_pointer++ = (unsigned char)(base >> 24);
						base <<= 8;
					} while ((length <<= 8) < AC_MinLength);
				}
			}
		}
		if (ac_pointer < end) {
			unsigned init_base = base;
			if (length > 2 * AC_MinLength) {
				base += AC_MinLength;
				length = AC_MinLength >> 1;
			}
			else {
				base += AC_MinLength >> 1;
				length = AC_MinLength >> 9;
			}

			if (init_base > base)
			{
				unsigned char * p;
				for (p = ac_pointer - 1; *p == 0xFFU; p--)
					*p = 0;
				++*p;
			}
			do
			{
				if (ac_pointer >= end)
					break;
				*ac_pointer++ = (unsigned char)(base >> 24);
				base <<= 8;

			} while ((length <<= 8) < AC_MinLength);
		}

		unsigned codeBytes = 0;
		if (ac_pointer < end) {
			codeBytes = unsigned(ac_pointer - oData);
		}

		oData[0] = codeBytes >> 8;
		oData[1] = codeBytes;
		oChunksSizes[chunkId] = codeBytes;
	}
}


/*__global__ void encodeKernelCoalesced
(unsigned const bit_0_prob,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned const chunkSize)
{
//32 threads read data for each other in coalesced way
int chunk = blockIdx.x*blockDim.x + threadIdx.x;
iData += blockIdx.x*blockDim.x*chunkSize;
oData += chunk*(chunkSize + 2);

const unsigned stepsCount = chunkSize / stepSize;
const unsigned bufSize = stepSize * blockDim.x;


unsigned char* ac_pointer = oData + 2;
unsigned length = 0xFFFFFFFFU;
unsigned base = 0;
for(unsigned step = 0; step < stepsCount; ++step)
{
//read [stepSize] bytes for every thread in warp
extern __shared__ unsigned char buf[];
for(unsigned i = 0; i < blockDim.x; ++i)
{
for(unsigned j = threadIdx.x; j < stepSize; j += blockDim.x)
{
buf[j] = iData[i*chunkSize + j];
}
}
for(unsigned i = 0; i < stepSize; ++i)
{
unsigned char byte = buf[threadIdx.x*stepSize + i];
*ac_pointer++ = byte;
for(unsigned j = 0; j < 8; ++j)
{
unsigned x = bit_0_prob * (length >> 13);
if ((byte & (1 << j)) == 0)
length  = x;
else
{
unsigned init_base = base;
base   += x;
length -= x;
if (init_base > base) //Overflow? Carry.
{
unsigned char * p;
for (p = ac_pointer - 1; *p == 0xFFU; p--)
*p = 0;
++*p;
}
}
if (length < 0x01000000U) //Renormalization
{
do
{
*ac_pointer++ = (unsigned char)(base >> 24);
base <<= 8;
} while ((length <<= 8) < 0x01000000U);
}
}
}
}
unsigned init_base = base;
if (length > 2 * 0x01000000U) {
base  += 0x01000000U;
length = 0x01000000U >> 1;
}
else {
base  += 0x01000000U >> 1;
length = 0x01000000U >> 9;
}

if (init_base > base)
{
unsigned char * p;
for (p = ac_pointer - 1; *p == 0xFFU; p--)
*p = 0;
++*p;
}
do
{
*ac_pointer++ = (unsigned char)(base >> 24);
base <<= 8;
} while ((length <<= 8) < 0x01000000U);

unsigned codeBytes = unsigned(ac_pointer - oData);
oData[0] = codeBytes >> 8;
oData[1] = codeBytes;
}

*/