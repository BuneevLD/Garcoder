#include <iostream>
#include "CPUCodec.h"
#include "_Constants.h"

namespace GARCoder
{
	inline void CPUCodec::propagate_carry(unsigned char*& ac_pointer)
	{
		unsigned char * p;
		for (p = ac_pointer - 1; *p == 0xFFU; p--)
			*p = 0;
		++*p;
	}

	inline bool CPUCodec::renorm_enc(unsigned char*& ac_pointer, unsigned& base, unsigned& length, unsigned char*& limit)
	{
		do
		{
			if (ac_pointer >= limit)
				return false;
			*ac_pointer++ = (unsigned char)(base >> 24);
			base <<= 8;
		} while ((length <<= 8) < AC_MinLength);
		return true;
	}
	inline void CPUCodec::renorm_dec(unsigned char*& ac_pointer, unsigned& value, unsigned& length, unsigned char* ac_pointer_max)
	{
		do {
			value = (value << 8) | unsigned(*++ac_pointer);
		} while ((length <<= 8) < AC_MinLength);
	}

	CPUCodec::CPUCodec() : ArithmeticCodec()
	{
	}

	CPUCodec::~CPUCodec()
	{
	}
	uint64 CPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, StaticBinaryModel& model)
	{
		unsigned char* oPointer = oData;
		unsigned char* iPointer = iData;
		unsigned base;
		unsigned length;
		unsigned nb;
		unsigned codeBytes = 0;

		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char * superBegin = oData;

		//std::cout << "Chunk sizes: ";

		//int failedChunks = 0;
		do
		{
			//Later in header we will store the size of encoded chunk.
			unsigned char* chunkHeader = oPointer;
			oPointer += 2;

			//Read AC__L2ChunkSize bytes if possible, or all data remain if not.
			nb = ((iDataSize) >= (unsigned long long)chunkSize) ? chunkSize : int(iDataSize);
			unsigned char* chunkEnd = oPointer + nb;

			base = 0;
			length = AC_MaxLength;


			
			//Encode chunk
			for (unsigned i = 0; i < nb; ++i)
			{
				for (unsigned j = 0; j < 8; ++j)
				{
					unsigned x = model.bit0Prob * (length >> BM_LengthShift);
					if ((iPointer[i] & (1 << j)) == 0)
						length = x;
					else
					{
						unsigned init_base = base;
						base += x;
						length -= x;
						if (init_base > base) //Overflow? Carry.
							propagate_carry(oPointer);
					}
					if (length < AC_MinLength) //Renormalization
						if (!renorm_enc(oPointer, base, length, chunkEnd))
						{
							break;
						}
				}
			}
			

			//Write last bits
			if (oPointer < chunkEnd)
			{
				unsigned init_base = base;
				if (length > 2 * AC_MinLength) {
					base += AC_MinLength;
					length = AC_MinLength >> 1;
				}
				else {
					base += AC_MinLength >> 1;
					length = AC_MinLength >> 9;
				}
				if (init_base > base) propagate_carry(oPointer);
				renorm_enc(oPointer, base, length, chunkEnd);
			}


			//Write encoded chunk size (2 bytes) //(15 bits => 32767 max.)
			if (oPointer < chunkEnd)
			{
				unsigned encodedSize = unsigned(oPointer - chunkHeader);
				codeBytes += encodedSize;
				chunkHeader[0] = encodedSize >> 8;
				chunkHeader[1] = encodedSize;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << " " << encodedSize << std::dec << std::endl;*/
			}
			else	//Compression failed - set "0" as chunk size and replace with original data
			{
				chunkHeader[0] = 0;
				chunkHeader[1] = 0;
				memcpy(chunkHeader + 2, iPointer, nb);
				oPointer = chunkHeader + 2 + nb;
				codeBytes += nb + 2;
				//failedChunks++;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
					std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/
				//std::cout << "FAILED CHUNK ";
			}

			iPointer += nb;
			chunksCount++;
			
		} while (iDataSize -= nb);
		//std::cout << std::endl << "Chunks count: " << chunksCount << std::endl;
		//std::cout << "Failed chunks: " << failedChunks << std::endl;
		return codeBytes;
	}
	void CPUCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, StaticBinaryModel& model)
	{
		unsigned nb;
		unsigned char* ac_pointer;
		unsigned char* iDataNext = iData;
		do
		{
			nb = (int(oDataSize) >= chunkSize) ? chunkSize : int(oDataSize);
			unsigned length = AC_MaxLength;
			short cSize = ((iData[0] << 8) | (iData[1]));
			if (cSize == 0)
			{
				iDataNext = iData + chunkSize + 2;
				memcpy(oData, iData + 2, chunkSize);
			}
			else
			{
				iDataNext = iData + cSize;
				unsigned value = (unsigned(iData[2]) << 24)
					| (unsigned(iData[3]) << 16)
					| (unsigned(iData[4]) << 8)
					| unsigned(iData[5]);
				ac_pointer = iData + 5;
				for (unsigned i = 0; i < nb; ++i)
				{
					oData[i] = 0;
					for (unsigned j = 0; j < 8; ++j)
					{
						unsigned x = model.bit0Prob * (length >> BM_LengthShift);
						unsigned bit = (value >= x);
						if (bit == 0)
							length = x;
						else
						{
							oData[i] |= 1 << j;
							value -= x;
							length -= x;
						}
						if (length < AC_MinLength) //Renormalization
							renorm_dec(ac_pointer, value, length, iDataNext);
					}
				}
			}
			oData += nb;
			iData = iDataNext;
		} while (oDataSize -= nb);
	}
	uint64 CPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveBinaryModel& model)
	{
		unsigned char* oPointer = oData;
		unsigned char* iPointer = iData;
		unsigned base;
		unsigned length;
		unsigned nb;
		unsigned codeBytes = 0;

		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char * superBegin = oData;

		//std::cout << "Chunk sizes: ";

		//int failedChunks = 0;
		do
		{
			model.reset();

			//Later in header we will store the size of encoded chunk.
			unsigned char* chunkHeader = oPointer;
			oPointer += 2;

			//Read AC__L2ChunkSize bytes if possible, or all data remain if not.
			nb = ((iDataSize) >= (unsigned long long)chunkSize) ? chunkSize : int(iDataSize);
			unsigned char* chunkEnd = oPointer + nb;

			base = 0;
			length = AC_MaxLength;

			//Encode chunk
			for (unsigned i = 0; i < nb; ++i)
			{
				for (unsigned j = 0; j < 8; ++j)
				{
					unsigned x = model.bit0Prob * (length >> BM_LengthShift);
					if ((iPointer[i] & (1 << j)) == 0) {
						length = x;
						++model.bit0Count;
					}
					else
					{
						unsigned init_base = base;
						base += x;
						length -= x;
						if (init_base > base) //Overflow? Carry.
							propagate_carry(oPointer);
					}
					if (length < AC_MinLength) //Renormalization
						if (!renorm_enc(oPointer, base, length, chunkEnd))
							break;

					if (--model.bitsUntilUpdate == 0) model.update();
				}
			}


			//Write last bits
			if (oPointer < chunkEnd)
			{
				unsigned init_base = base;
				if (length > 2 * AC_MinLength) {
					base += AC_MinLength;
					length = AC_MinLength >> 1;
				}
				else {
					base += AC_MinLength >> 1;
					length = AC_MinLength >> 9;
				}
				if (init_base > base) propagate_carry(oPointer);
				renorm_enc(oPointer, base, length, chunkEnd);
			}


			//Write encoded chunk size (2 bytes) //(15 bits => 32767 max.)
			if (oPointer < chunkEnd)
			{
				unsigned encodedSize = unsigned(oPointer - chunkHeader);
				codeBytes += encodedSize;
				chunkHeader[0] = encodedSize >> 8;
				chunkHeader[1] = encodedSize;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << " " << encodedSize << std::dec << std::endl;*/
			}
			else	//Compression failed - set "0" as chunk size and replace with original data
			{
				chunkHeader[0] = 0;
				chunkHeader[1] = 0;
				memcpy(chunkHeader + 2, iPointer, nb);
				oPointer = chunkHeader + 2 + nb;
				codeBytes += nb + 2;
				//failedChunks++;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/
				//std::cout << "FAILED CHUNK ";
			}

			iPointer += nb;
			chunksCount++;

		} while (iDataSize -= nb);
		//std::cout << std::endl << "Chunks count: " << chunksCount << std::endl;
		//std::cout << "Failed chunks: " << failedChunks << std::endl;
		return codeBytes;
	}
	void CPUCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveBinaryModel& model)
	{
		unsigned nb;
		unsigned char* ac_pointer;
		unsigned char* iDataNext = iData;
		do
		{
			model.reset();

			nb = (int(oDataSize) >= chunkSize) ? chunkSize : int(oDataSize);
			unsigned length = AC_MaxLength;
			short cSize = ((iData[0] << 8) | (iData[1]));
			if (cSize == 0)
			{
				iDataNext = iData + chunkSize + 2;
				memcpy(oData, iData + 2, chunkSize);
			}
			else
			{
				iDataNext = iData + cSize;
				unsigned value = (unsigned(iData[2]) << 24)
					| (unsigned(iData[3]) << 16)
					| (unsigned(iData[4]) << 8)
					| unsigned(iData[5]);
				ac_pointer = iData + 5;
				for (unsigned i = 0; i < nb; ++i)
				{
					oData[i] = 0;
					for (unsigned j = 0; j < 8; ++j)
					{
						unsigned x = model.bit0Prob * (length >> BM_LengthShift);
						unsigned bit = (value >= x);
						if (bit == 0) {
							length = x;
							++model.bit0Count;
						}
						else
						{
							oData[i] |= 1 << j;
							value -= x;
							length -= x;
						}
						if (length < AC_MinLength) //Renormalization
							renorm_dec(ac_pointer, value, length, iDataNext);
						if (--model.bitsUntilUpdate == 0) model.update();
					}
				}
			}
			oData += nb;
			iData = iDataNext;
		} while (oDataSize -= nb);
	}
	uint64 CPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, AdaptiveByteModel& model)
	{
		unsigned char* oPointer = oData;
		unsigned char* iPointer = iData;
		unsigned base;
		unsigned length;
		unsigned nb;
		unsigned codeBytes = 0;

		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char * superBegin = oData;

		//std::cout << "Chunk sizes: ";

		//int failedChunks = 0;
		do
		{
			model.reset();

			//Later in header we will store the size of encoded chunk.
			unsigned char* chunkHeader = oPointer;
			oPointer += 2;

			//Read AC__L2ChunkSize bytes if possible, or all data remain if not.
			nb = ((iDataSize) >= (unsigned long long)chunkSize) ? chunkSize : int(iDataSize);
			unsigned char* chunkEnd = oPointer + nb;

			base = 0;
			length = AC_MaxLength;



			//Encode chunk
			for (unsigned i = 0; i < nb; ++i)
			{
				unsigned data = iPointer[i];
				unsigned x, init_base = base;
				// compute products
				if (data == model.last_symbol) {
					x = model.distribution[data] * (length >> CM_LengthShift);
					base += x;                                            // update interval
					length -= x;                                          // no product needed
				}
				else {
					x = model.distribution[data] * (length >>= CM_LengthShift);
					base += x;                                            // update interval
					length = model.distribution[data + 1] * length - x;
				}

				if (init_base > base) propagate_carry(oPointer);                 // overflow = carry

				if (length < AC_MinLength)
					if (!renorm_enc(oPointer, base, length, chunkEnd))
						break;

				++model.symbol_count[data];
				if (--model.symbols_until_update == 0) model.update(true);  // periodic model update
			}


			//Write last bits
			if (oPointer < chunkEnd)
			{
				unsigned init_base = base;
				if (length > 2 * AC_MinLength) {
					base += AC_MinLength;
					length = AC_MinLength >> 1;
				}
				else {
					base += AC_MinLength >> 1;
					length = AC_MinLength >> 9;
				}
				if (init_base > base) propagate_carry(oPointer);
				renorm_enc(oPointer, base, length, chunkEnd);
			}


			//Write encoded chunk size (2 bytes) //(15 bits => 32767 max.)
			if (oPointer < chunkEnd)
			{
				unsigned encodedSize = unsigned(oPointer - chunkHeader);
				codeBytes += encodedSize;
				chunkHeader[0] = encodedSize >> 8;
				chunkHeader[1] = encodedSize;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << " " << encodedSize << std::dec << std::endl;*/
			}
			else	//Compression failed - set "0" as chunk size and replace with original data
			{
				chunkHeader[0] = 0;
				chunkHeader[1] = 0;
				memcpy(chunkHeader + 2, iPointer, nb);
				oPointer = chunkHeader + 2 + nb;
				codeBytes += nb + 2;
				//failedChunks++;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/
				//std::cout << "FAILED CHUNK ";
			}

			iPointer += nb;
			chunksCount++;

		} while (iDataSize -= nb);
		//std::cout << std::endl << "Chunks count: " << chunksCount << std::endl;
		//std::cout << "Failed chunks: " << failedChunks << std::endl;
		return codeBytes;
	}
	void CPUCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, AdaptiveByteModel& model)
	{
		unsigned nb;
		unsigned char* ac_pointer;
		unsigned char* iDataNext = iData;
		do
		{
			model.reset();

			nb = (int(oDataSize) >= chunkSize) ? chunkSize : int(oDataSize);
			unsigned length = AC_MaxLength;
			short cSize = ((iData[0] << 8) | (iData[1]));
			if (cSize == 0)
			{
				iDataNext = iData + chunkSize + 2;
				memcpy(oData, iData + 2, chunkSize);
			}
			else
			{
				iDataNext = iData + cSize;
				unsigned value = (unsigned(iData[2]) << 24)
					| (unsigned(iData[3]) << 16)
					| (unsigned(iData[4]) << 8)
					| unsigned(iData[5]);
				ac_pointer = iData + 5;
				for (unsigned i = 0; i < nb; ++i)
				{
					oData[i] = 0;
					unsigned n, s, x, y = length;

					if (model.decoder_table) {              // use table look-up for faster decoding

						unsigned dv = value / (length >>= CM_LengthShift);
						unsigned t = dv >> model.table_shift;

						s = model.decoder_table[t];         // initial decision based on table look-up
						n = model.decoder_table[t + 1] + 1;

						while (n > s + 1) {                        // finish with bisection search
							unsigned m = (s + n) >> 1;
							if (model.distribution[m] > dv) n = m; else s = m;
						}
						// compute products
						x = model.distribution[s] * length;
						if (s != model.last_symbol) y = model.distribution[s + 1] * length;
					}


					value -= x;                                               // update interval
					length = y - x;

					if (length < AC_MinLength) renorm_dec(ac_pointer, value, length, iDataNext);        // renormalization

					++model.symbol_count[s];
					if (--model.symbols_until_update == 0) model.update(false);  // periodic model update

					oData[i] = s;
				}
			}
			oData += nb;
			iData = iDataNext;
		} while (oDataSize -= nb);
	}





	uint64 CPUCodec::EncodeData(byte* iData, uint64 iDataSize, byte* oData, ContextAdaptiveBinaryModel& model)
	{
		unsigned char* oPointer = oData;
		unsigned char* iPointer = iData;
		unsigned base;
		unsigned length;
		unsigned nb;
		unsigned codeBytes = 0;

		unsigned chunksCount = iDataSize / chunkSize;
		if (iDataSize % chunkSize > 0)
			chunksCount++;

		unsigned char * superBegin = oData;

		//std::cout << "Chunk sizes: ";

		//int failedChunks = 0;
		do
		{
			model.reset();

			//Later in header we will store the size of encoded chunk.
			unsigned char* chunkHeader = oPointer;
			oPointer += 2;

			//Read AC__L2ChunkSize bytes if possible, or all data remain if not.
			nb = ((iDataSize) >= (unsigned long long)chunkSize) ? chunkSize : int(iDataSize);
			unsigned char* chunkEnd = oPointer + nb;

			base = 0;
			length = AC_MaxLength;

			//Encode chunk
			unsigned char prefix = 0;
			for (unsigned i = 0; i < nb; ++i)
			{
				for (unsigned j = 0; j < 8; ++j)
				{

					unsigned x = model.models[prefix].bit0Prob * (length >> BM_LengthShift);
					if ((iPointer[i] & (1 << j)) == 0) {
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
							propagate_carry(oPointer);

						if (--model.models[prefix].bitsUntilUpdate == 0) model.models[prefix].update();
						prefix <<= 1;
						prefix |= 1U;
					}
					if (length < AC_MinLength) //Renormalization
						if (!renorm_enc(oPointer, base, length, chunkEnd))
							break;
				}
			}


			//Write last bits
			if (oPointer < chunkEnd)
			{
				unsigned init_base = base;
				if (length > 2 * AC_MinLength) {
					base += AC_MinLength;
					length = AC_MinLength >> 1;
				}
				else {
					base += AC_MinLength >> 1;
					length = AC_MinLength >> 9;
				}
				if (init_base > base) propagate_carry(oPointer);
				renorm_enc(oPointer, base, length, chunkEnd);
			}


			//Write encoded chunk size (2 bytes) //(15 bits => 32767 max.)
			if (oPointer < chunkEnd)
			{
				unsigned encodedSize = unsigned(oPointer - chunkHeader);
				codeBytes += encodedSize;
				chunkHeader[0] = encodedSize >> 8;
				chunkHeader[1] = encodedSize;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << " " << encodedSize << std::dec << std::endl;*/
			}
			else	//Compression failed - set "0" as chunk size and replace with original data
			{
				chunkHeader[0] = 0;
				chunkHeader[1] = 0;
				memcpy(chunkHeader + 2, iPointer, nb);
				oPointer = chunkHeader + 2 + nb;
				codeBytes += nb + 2;
				//failedChunks++;

				/*unsigned long long chunkNumber = (chunkHeader - superBegin) + 12;
				if (chunkNumber > 0x20000 && chunkNumber < 0x200A0)
				std::cout << "Chunk #" << std::hex << chunkNumber << std::dec << " 00" << std::endl;*/
				//std::cout << "FAILED CHUNK ";
			}

			iPointer += nb;
			chunksCount++;

		} while (iDataSize -= nb);
		//std::cout << std::endl << "Chunks count: " << chunksCount << std::endl;
		//std::cout << "Failed chunks: " << failedChunks << std::endl;
		return codeBytes;
	}
	void CPUCodec::DecodeData(byte* iData, byte* oData, uint64 oDataSize, ContextAdaptiveBinaryModel& model)
	{
		unsigned nb;
		unsigned char* ac_pointer;
		unsigned char* iDataNext = iData;
		do
		{
			model.reset();

			nb = (int(oDataSize) >= chunkSize) ? chunkSize : int(oDataSize);
			unsigned length = AC_MaxLength;
			short cSize = ((iData[0] << 8) | (iData[1]));
			if (cSize == 0)
			{
				iDataNext = iData + chunkSize + 2;
				memcpy(oData, iData + 2, chunkSize);
			}
			else
			{
				iDataNext = iData + cSize;
				unsigned value = (unsigned(iData[2]) << 24)
					| (unsigned(iData[3]) << 16)
					| (unsigned(iData[4]) << 8)
					| unsigned(iData[5]);
				ac_pointer = iData + 5;

				unsigned char prefix = 0;
				for (unsigned i = 0; i < nb; ++i)
				{
					oData[i] = 0;
					for (unsigned j = 0; j < 8; ++j)
					{
						unsigned x = model.models[prefix].bit0Prob * (length >> BM_LengthShift);
						unsigned bit = (value >= x);
						if (bit == 0) {
							length = x;
							++model.models[prefix].bit0Count;
							if (--model.models[prefix].bitsUntilUpdate == 0) model.models[prefix].update();
							prefix <<= 1;
						}
						else
						{
							oData[i] |= 1 << j;
							value -= x;
							length -= x;
							if (--model.models[prefix].bitsUntilUpdate == 0) model.models[prefix].update();
							prefix <<= 1;
							prefix |= 1;
						}
						if (length < AC_MinLength) //Renormalization
							renorm_dec(ac_pointer, value, length, iDataNext);

						
					}
				}
			}
			oData += nb;
			iData = iDataNext;
		} while (oDataSize -= nb);
	}

}