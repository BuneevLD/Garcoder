#include "ArithmeticCodec.h"
#include "Exceptions.h"

namespace GARCoder
{
	ArithmeticCodec::ArithmeticCodec()
	{
		chunkSize = 8192;
		chunkSizeBytes = 2;
	}

	ArithmeticCodec::~ArithmeticCodec()
	{
	}

	uint ArithmeticCodec::get_chunkSize()
	{
		return chunkSize;
	}
	void ArithmeticCodec::set_chunkSize(uint value)
	{
		chunkSize = value;
	}
	uint ArithmeticCodec::get_chunkSizeBytes()
	{
		return chunkSizeBytes;
	}
	void ArithmeticCodec::set_chunkSizeBytes(uint value)
	{
		chunkSizeBytes = value;
	}

	uint64 ArithmeticCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, StaticBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement static binary encoding");
	}

	void ArithmeticCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, StaticBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement static binary decoding");
	}
	uint64 ArithmeticCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive binary encoding");
	}

	void ArithmeticCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive binary decoding");
	}

	uint64 ArithmeticCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, ContextAdaptiveBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive binary encoding");
	}

	void ArithmeticCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, ContextAdaptiveBinaryModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive binary decoding");
	}

	uint64 ArithmeticCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveByteModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive context encoding");
	}

	void ArithmeticCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveByteModel& model)
	{
		throw NotImplementedException("Codec used doesn't implement adaptive context decoding");
	}
}