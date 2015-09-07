#include "GPUCodec.h"
#include "Exceptions.h"
#include <algorithm>
#include <iostream>
#include <chrono>


void launchBasicEncodeKernel
(unsigned const bit_0_prob,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize);

void launchAdaptiveEncodeKernel
(AdaptiveBinaryModel model,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize);

void launchContextAdaptiveEncodeKernel
(unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize);

void launchPrefixSumEncodeKernel
(unsigned const bit_0_prob,
unsigned char* iData,
unsigned const iSize,
unsigned char* oData,
unsigned* oChunkSizes,
unsigned chunksCount,
unsigned const chunkSize,
unsigned* overallSize);

namespace GARCoder
{
	GPUCodec::GPUCodec() : ArithmeticCodec()
	{
		mode = Mode::Basic;
	}
	GPUCodec::~GPUCodec()
	{
	}

	void GPUCodec::checkCudaStatus(cudaError_t cudaStatus)
	{
		if (cudaStatus != cudaSuccess)
			throw CudaException(cudaStatus);
	}

	uint64 GPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, StaticBinaryModel& model)
	{
		/*if(iDataSize % chunkSize != 0)
		{
		throw GPUCoderWrongInputException();
		}*/
		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char* dev_iData = 0;
		unsigned char* dev_oData = 0;
		unsigned* dev_compressedSizes = 0;
		unsigned* dev_overallCount = 0;

		cudaError_t cudaStatus;
		checkCudaStatus(cudaMalloc((void**)&dev_iData, iDataSize));
		checkCudaStatus(cudaMalloc((void**)&dev_oData, iDataSize * 2));
		checkCudaStatus(cudaMalloc((void**)&dev_compressedSizes, sizeof(unsigned)*chunksCount));
		checkCudaStatus(cudaMalloc(&dev_overallCount, sizeof(unsigned)));
		checkCudaStatus(cudaMemcpy(dev_iData, iData, iDataSize, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		if (mode == Mode::Basic) 
		{
			//auto t1 = std::chrono::high_resolution_clock::now();
			launchBasicEncodeKernel(model.bit0Prob, dev_iData, iDataSize, dev_oData, dev_compressedSizes, chunksCount, chunkSize);
			checkCudaStatus(cudaDeviceSynchronize());
			//auto t2 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tMiniElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

			unsigned oSize = 0;
			unsigned char* temp_oData = new unsigned char[iDataSize * 2];
			unsigned* compressedSizes = new unsigned[chunksCount];

			checkCudaStatus(cudaMemcpy(temp_oData, dev_oData, iDataSize * 2, cudaMemcpyDeviceToHost));
			checkCudaStatus(cudaMemcpy(compressedSizes, dev_compressedSizes, sizeof(unsigned)*chunksCount, cudaMemcpyDeviceToHost));

			int k = 0;
			/*std::cout << "Compressed sizes: ";
			for (int j = 0; j < chunksCount; j++)
			{
				std::cout << compressedSizes[j] << " ";
			}
			std::cout << std::endl;*/
			//int failedChunks = 0;

			//auto t3 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < chunksCount; i++)
			{
				if (compressedSizes[i] == 0) {
					oData[k] = 0;
					++k;
					oData[k] = 0;
					++k;
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
						std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/

					unsigned long long maxSize = std::min((unsigned long long)chunkSize, (iDataSize - (i*chunkSize)));
					memcpy(oData + k, iData + (i*chunkSize), maxSize);

					k += maxSize;
					//failedChunks++;
				}
				else {
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
						std::cout << "Chunk #" << std::hex << chunkNumber << " " << compressedSizes[i] << std::dec << std::endl;*/

					memcpy(oData + k, temp_oData + (i*(chunkSize + 2)), compressedSizes[i]);
					k += compressedSizes[i];
				}
			}
			//auto t4 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tPrefixSumElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms" << std::endl;
			//std::cout << "Failed chunks: " << failedChunks << std::endl;
			cudaFree(dev_iData);
			cudaFree(dev_oData);
			cudaFree(dev_compressedSizes);
			cudaFree(dev_overallCount);
			return k;
		}

		return 0;
	}

	uint64 GPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveBinaryModel& model)
	{
		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char* dev_iData = 0;
		unsigned char* dev_oData = 0;
		unsigned* dev_compressedSizes = 0;
		unsigned* dev_overallCount = 0;

		cudaError_t cudaStatus;
		cudaDeviceSynchronize();
		checkCudaStatus(cudaMalloc((void**)&dev_iData, iDataSize));
		checkCudaStatus(cudaMalloc((void**)&dev_oData, iDataSize + chunksCount * 2));
		checkCudaStatus(cudaMalloc((void**)&dev_compressedSizes, sizeof(unsigned)*chunksCount));
		checkCudaStatus(cudaMalloc(&dev_overallCount, sizeof(unsigned)));
		checkCudaStatus(cudaMemcpy(dev_iData, iData, iDataSize, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		if (mode == Mode::Basic)
		{
			//auto t1 = std::chrono::high_resolution_clock::now();
			model.reset();
			launchAdaptiveEncodeKernel(model, dev_iData, iDataSize, dev_oData, dev_compressedSizes, chunksCount, chunkSize);
			checkCudaStatus(cudaDeviceSynchronize());
			//auto t2 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tMiniElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

			unsigned oSize = 0;
			unsigned char* temp_oData = new unsigned char[iDataSize + chunksCount * 2];
			unsigned* compressedSizes = new unsigned[chunksCount];

			checkCudaStatus(cudaMemcpy(temp_oData, dev_oData, iDataSize + chunksCount * 2, cudaMemcpyDeviceToHost));
			checkCudaStatus(cudaMemcpy(compressedSizes, dev_compressedSizes, sizeof(unsigned)*chunksCount, cudaMemcpyDeviceToHost));

			int k = 0;
			/*std::cout << "Compressed sizes: ";
			for (int j = 0; j < chunksCount; j++)
			{
			std::cout << compressedSizes[j] << " ";
			}
			std::cout << std::endl;*/
			//int failedChunks = 0;

			//auto t3 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < chunksCount; i++)
			{
				if (compressedSizes[i] == 0) {
					oData[k] = 0;
					++k;
					oData[k] = 0;
					++k;
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/

					unsigned long long maxSize = std::min((unsigned long long)chunkSize, (iDataSize - (i*chunkSize)));
					memcpy(oData + k, iData + (i*chunkSize), maxSize);

					k += maxSize;
					//failedChunks++;
				}
				else {
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << " " << compressedSizes[i] << std::dec << std::endl;*/

					memcpy(oData + k, temp_oData + (i*(chunkSize + 2)), compressedSizes[i]);
					k += compressedSizes[i];
				}
			}
			//auto t4 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tPrefixSumElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms" << std::endl;
			//std::cout << "Failed chunks: " << failedChunks << std::endl;
			cudaFree(dev_iData);
			cudaFree(dev_oData);
			cudaFree(dev_compressedSizes);
			cudaFree(dev_overallCount);
			return k;
		}

		return 0;
	}

	uint64 GPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, ContextAdaptiveBinaryModel& model)
	{
		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char* dev_iData = 0;
		unsigned char* dev_oData = 0;
		unsigned* dev_compressedSizes = 0;
		unsigned* dev_overallCount = 0;

		cudaError_t cudaStatus;
		checkCudaStatus(cudaMalloc((void**)&dev_iData, iDataSize));
		checkCudaStatus(cudaMalloc((void**)&dev_oData, iDataSize * 2));
		checkCudaStatus(cudaMalloc((void**)&dev_compressedSizes, sizeof(unsigned)*chunksCount));
		checkCudaStatus(cudaMalloc(&dev_overallCount, sizeof(unsigned)));
		checkCudaStatus(cudaMemcpy(dev_iData, iData, iDataSize, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		if (mode == Mode::Basic)
		{
			//auto t1 = std::chrono::high_resolution_clock::now();
			model.reset();
			launchContextAdaptiveEncodeKernel(dev_iData, iDataSize, dev_oData, dev_compressedSizes, chunksCount, chunkSize);
			checkCudaStatus(cudaDeviceSynchronize());
			//auto t2 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tMiniElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

			unsigned oSize = 0;
			unsigned char* temp_oData = new unsigned char[iDataSize * 2];
			unsigned* compressedSizes = new unsigned[chunksCount];

			checkCudaStatus(cudaMemcpy(temp_oData, dev_oData, iDataSize * 2, cudaMemcpyDeviceToHost));
			checkCudaStatus(cudaMemcpy(compressedSizes, dev_compressedSizes, sizeof(unsigned)*chunksCount, cudaMemcpyDeviceToHost));

			int k = 0;
			/*std::cout << "Compressed sizes: ";
			for (int j = 0; j < chunksCount; j++)
			{
			std::cout << compressedSizes[j] << " ";
			}
			std::cout << std::endl;*/
			//int failedChunks = 0;

			//auto t3 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < chunksCount; i++)
			{
				if (compressedSizes[i] == 0) {
					oData[k] = 0;
					++k;
					oData[k] = 0;
					++k;
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/

					unsigned long long maxSize = std::min((unsigned long long)chunkSize, (iDataSize - (i*chunkSize)));
					memcpy(oData + k, iData + (i*chunkSize), maxSize);

					k += maxSize;
					//failedChunks++;
				}
				else {
					/*unsigned long long chunkNumber = (i*(chunkSize + 2)) + 12;
					if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << " " << compressedSizes[i] << std::dec << std::endl;*/

					memcpy(oData + k, temp_oData + (i*(chunkSize + 2)), compressedSizes[i]);
					k += compressedSizes[i];
				}
			}
			//auto t4 = std::chrono::high_resolution_clock::now();
			//std::cout << "\tPrefixSumElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms" << std::endl;
			//std::cout << "Failed chunks: " << failedChunks << std::endl;
			cudaFree(dev_iData);
			cudaFree(dev_oData);
			cudaFree(dev_compressedSizes);
			cudaFree(dev_overallCount);
			return k;
		}

		return 0;
	}


	//void GPUCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize) {	}
}