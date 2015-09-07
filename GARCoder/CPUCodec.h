#pragma once
#include "ArithmeticCodec.h"

namespace GARCoder
{
	class CPUCodec : public ArithmeticCodec
	{
	protected:
		inline void propagate_carry(unsigned char*& ac_pointer);
		inline bool renorm_enc(unsigned char*& ac_pointer, unsigned& base, unsigned& length, unsigned char*& limit);
		inline void renorm_dec(unsigned char*& ac_pointer, unsigned& value, unsigned& length, unsigned char* ac_pointer_max);
	public:
		CPUCodec();
		virtual ~CPUCodec();

		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, StaticBinaryModel& model);
		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveBinaryModel& model);
		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, ContextAdaptiveBinaryModel& model);
		virtual uint64 EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveByteModel& model);
		virtual void DecodeData(byte* iData, byte* oData, uint64 oDataSize, StaticBinaryModel& model);
		virtual void DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveBinaryModel& model);
		virtual void DecodeData(byte* iData, byte* oData, uint64 oDataSize, ContextAdaptiveBinaryModel& model);
		virtual void DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveByteModel& model);
	};
}