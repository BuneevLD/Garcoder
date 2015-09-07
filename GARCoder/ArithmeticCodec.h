#pragma once
#include "DataModels.cu"
namespace GARCoder
{
	typedef unsigned int uint;
	typedef unsigned long long uint64;
	typedef unsigned char byte;

	class ArithmeticCodec
	{
	protected:
		unsigned int chunkSizeBytes;							// amount of bytes used for storing chunk size
		unsigned int chunkSize;					// chunk size, including {chunkSizeBytes} bytes for storing it
	public:
		virtual uint get_chunkSize();
		virtual void set_chunkSize(uint value);
		virtual uint get_chunkSizeBytes();
		virtual void set_chunkSizeBytes(uint value);
		ArithmeticCodec();
		virtual ~ArithmeticCodec();

		virtual uint64 EncodeData(	// returns size of encoded data byte array
			byte* iData,						// input data byte array
			uint64 iDataSize,					// input data size in bytes
			byte* oData,							// output data byte array pointer, must be allocated and contain at least {iDataSize} bytes
			StaticBinaryModel& model
			);

		virtual void DecodeData(
			byte* iData,						// input data byte array
			byte* oData,						// output data byte array pointer, must be allocated and contain at least {iDataSize} bytes
			uint64 oDataSize,					// output (decoded) data byte array size, logic for retrieving it from compressed data stream should be implemented on higher level
			StaticBinaryModel& model
			);

		virtual uint64 EncodeData(	// returns size of encoded data byte array
			byte* iData,						// input data byte array
			uint64 iDataSize,					// input data size in bytes
			byte* oData,							// output data byte array pointer, must be allocated and contain at least {iDataSize} 
			AdaptiveBinaryModel& model
			);

		virtual void DecodeData(
			byte* iData,						// input data byte array
			byte* oData,						// output data byte array pointer, must be allocated and contain at least {iDataSize} bytes
			uint64 oDataSize,					// output (decoded) data byte array size, logic for retrieving it from compressed data stream should be implemented on higher level
			AdaptiveBinaryModel& model
			);

		virtual uint64 EncodeData(	// returns size of encoded data byte array
			byte* iData,						// input data byte array
			uint64 iDataSize,					// input data size in bytes
			byte* oData,							// output data byte array pointer, must be allocated and contain at least {iDataSize} 
			ContextAdaptiveBinaryModel& model
			);

		virtual void DecodeData(
			byte* iData,						// input data byte array
			byte* oData,						// output data byte array pointer, must be allocated and contain at least {iDataSize} bytes
			uint64 oDataSize,					// output (decoded) data byte array size, logic for retrieving it from compressed data stream should be implemented on higher level
			ContextAdaptiveBinaryModel& model
			);

		virtual uint64 EncodeData(	// returns size of encoded data byte array
			byte* iData,						// input data byte array
			uint64 iDataSize,					// input data size in bytes
			byte* oData,							// output data byte array pointer, must be allocated and contain at least {iDataSize} 
			AdaptiveByteModel& model
			);

		virtual void DecodeData(
			byte* iData,						// input data byte array
			byte* oData,						// output data byte array pointer, must be allocated and contain at least {iDataSize} bytes
			uint64 oDataSize,					// output (decoded) data byte array size, logic for retrieving it from compressed data stream should be implemented on higher level
			AdaptiveByteModel& model
			);
	};
}