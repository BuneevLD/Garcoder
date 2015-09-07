#pragma once
#include "ArithmeticCodec.h"
#include "cuda_runtime.h"

namespace GARCoder
{
	class GPUCodec : public ArithmeticCodec
	{
	public:
		enum Mode { 
			Basic,
			PrefixSum
		};
		Mode mode;
	private:
		void checkCudaStatus(cudaError_t status);
	public:
		GPUCodec();
		virtual ~GPUCodec(void);

		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, StaticBinaryModel& model);
		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveBinaryModel& model);
		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, ContextAdaptiveBinaryModel& model);
		//virtual void DecodeData(byte* iData, byte* oData, uint64 oDataSize);
	};
}
