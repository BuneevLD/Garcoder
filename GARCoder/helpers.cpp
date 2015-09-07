#pragma once
#include <vector>
#include <iostream>
#include <exception>
#include <chrono>

#include "CPUCodec.h"
#include "GPUCodec.h"

class InvalidArgsException : public std::exception
{
public:
	InvalidArgsException()
		: exception("Wrong arguments!") { }
};

class Helpers
{
public:
	static const unsigned long long MainChunkSize = 32 * 1024 * 1024;
	static const unsigned FILE_ID = 0xB8AA3B30U;


#pragma region Helper functions
	static unsigned CRC(unsigned bytes, unsigned char * buffer)
	{
		// data for generating CRC table
		static const unsigned CRC_Gen[8] = {
			0xEC1A5A3EU, 0x5975F5D7U, 0xB2EBEBAEU, 0xE49696F7U,
			0x486C6C45U, 0x90D8D88AU, 0xA0F0F0BFU, 0xC0A0A0D5U };
		// table for fast CRC computation
		static unsigned CRC_Table[256];

		// compute table
		if (CRC_Table[1] == 0)
			for (unsigned k = CRC_Table[0] = 0; k < 8; k++)
			{
				unsigned s = 1 << k, g = CRC_Gen[k];
				for (unsigned n = 0; n < s; n++) CRC_Table[n + s] = CRC_Table[n] ^ g;
			}

		// computes buffer's cyclic redundancy check
		unsigned crc = 0;
		if (bytes)
			do {
				crc = (crc >> 8) ^ CRC_Table[(crc & 0xFFU) ^ unsigned(*buffer++)];
			} while (--bytes);
		return crc;
	}

	static FILE * Open_Input_File(const char * file_name)
	{
		FILE * new_file = fopen(file_name, "rb");
		if (new_file == 0) throw std::exception("cannot open input file");
		return new_file;
	}

	static FILE * Open_Output_File(const char * file_name, bool overwriteCheck = false)
	{
		FILE * new_file = fopen(file_name, "rb");
		if (new_file != 0) {
			fclose(new_file);
			if (overwriteCheck)
			{
				std::cout << "Overwrite file " << file_name << "? (y = yes, otherwise quit)" << std::endl;
				char answer;
				std::cin >> answer;
				if (answer != 'y')
					throw std::exception("File already exists and will not be overwritten.");
			}
		}
		new_file = fopen(file_name, "wb");
		if (new_file == 0)
			throw std::exception("cannot open output file");
		return new_file;
	}

	static void SaveNumber(unsigned n, unsigned char * b)
	{                                                   // decompose 4-byte number
		b[0] = (unsigned char)(n & 0xFFU);
		b[1] = (unsigned char)((n >> 8) & 0xFFU);
		b[2] = (unsigned char)((n >> 16) & 0xFFU);
		b[3] = (unsigned char)(n >> 24);
	}

	static void SaveVLNumber(unsigned n, FILE* file)
	{
		unsigned nRemain = n;
		do
		{
			int fileByte = int(nRemain & 0x7FU);
			if ((nRemain >>= 7) > 0)
				fileByte |= 0x80;
			if (putc(fileByte, file) == EOF)
				throw std::exception("Cannot write compressed data to file.");
		} while (nRemain);
	}

	static unsigned RecoverVLNumber(FILE* file)
	{
		unsigned shift = 0, n = 0;
		int fileByte;
		do
		{
			if ((fileByte = getc(file)) == EOF)
				throw std::exception("Cannot read code from file");
			n |= unsigned(fileByte & 0x7F) << shift;
			shift += 7;
		} while (fileByte & 0x80);
		return n;
	}

	static unsigned RecoverNumber(unsigned char * b)
	{                                                    // recover 4-byte integer
		return unsigned(b[0]) + (unsigned(b[1]) << 8) +
			(unsigned(b[2]) << 16) + (unsigned(b[3]) << 24);
	}
#pragma endregion

	template<typename DataModel>
	static void EncodeFile(const char * dataFileName, const char * codeFileName, GARCoder::ArithmeticCodec& encoder, DataModel& model)
	{
		unsigned int L1ChunkSize = MainChunkSize;
		unsigned int L2ChunkSize = encoder.get_chunkSize();

		FILE * dataFile = Open_Input_File(dataFileName);
		FILE * codeFile = Open_Output_File(codeFileName);

		std::vector<unsigned char> iData(L1ChunkSize);


		// compute CRC (cyclic check) of file
		unsigned nb, bytes = 0, crc = 0;
		do
		{
			nb = fread(iData.data(), 1, L1ChunkSize, dataFile);
			bytes += nb;
			crc ^= CRC(nb, iData.data());
		} while (nb == L1ChunkSize);
		rewind(dataFile);

		//12-byte header
		unsigned char header[12];
		SaveNumber(FILE_ID, header);
		SaveNumber(crc, header + 4);
		SaveNumber(bytes, header + 8);
		if (fwrite(header, 1, 12, codeFile) != 12)
			throw std::exception("Cannot write to file.");

		const unsigned oSize = L1ChunkSize + sizeof(L2ChunkSize)*L1ChunkSize / L2ChunkSize;
		std::vector<unsigned char> oData(oSize);

		auto t1 = std::chrono::high_resolution_clock::now();
		do
		{
			//Read nb bytes to iData vector
			nb = (bytes < L1ChunkSize ? bytes : L1ChunkSize);
			iData.resize(nb);
			if (fread(iData.data(), 1, nb, dataFile) != nb)
				throw std::exception("Cannot read from file.");   // read file data	

			oData.resize(oSize);
			
			unsigned encSize = encoder.EncodeData(iData.data(), iData.size(), oData.data(), model);

			//std::cout << "\tEncoded: " << encSize << std::endl;

			//write variable-length size
			SaveVLNumber(encSize, codeFile);

			//write encoded data
			fwrite(oData.data(), 1, encSize, codeFile);

		} while (bytes -= nb);
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "\tElapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;


		//Done - close files;
		fflush(codeFile);
		unsigned dataBytes = ftell(dataFile), codeBytes = ftell(codeFile);
		printf("\tCompressed size (without header) = %d bytes \n\tSource size = %d bytes\n\tCompression rate = %6.4f:1\n",
			codeBytes - 12, dataBytes, double(dataBytes) / double(codeBytes - 12));
		fclose(dataFile);
		fclose(codeFile);
	}

	template<typename DataModel>
	static void DecodeFile(const char * codeFileName, const char * dataFileName, GARCoder::ArithmeticCodec& decoder, DataModel& model)
	{
		unsigned L1ChunkSize = MainChunkSize;
		unsigned L2ChunkSize = decoder.get_chunkSize();
		FILE * codeFile = Open_Input_File(codeFileName);
		FILE * dataFile = Open_Output_File(dataFileName);

		// read file information from 12-byte header
		unsigned char header[12];
		if (fread(header, 1, 12, codeFile) != 12) throw std::exception("Cannot read from file.");
		unsigned fid = RecoverNumber(header);
		unsigned crc = RecoverNumber(header + 4);
		unsigned bytes = RecoverNumber(header + 8);

		if (fid != FILE_ID) throw std::exception("invalid compressed file");

		std::vector<unsigned char> iData(L1ChunkSize + 16);
		std::vector<unsigned char> oData(L1ChunkSize);

		unsigned nb, new_crc = 0, context = 0;
		do
		{
			//Retrieve chunk size (variable-length)
			//first bit of byte is flag "IsNotLastByte", 7bits are part of length
			unsigned codeBytes = RecoverVLNumber(codeFile);

			iData.resize(codeBytes);
			if (fread(iData.data(), 1, codeBytes, codeFile) != codeBytes)
				throw std::exception("Cannot read code from file");

			nb = (bytes < L1ChunkSize ? bytes : L1ChunkSize);
			oData.resize(nb);

			decoder.DecodeData(iData.data(), oData.data(), oData.size(), model);

			new_crc ^= CRC(nb, oData.data());                // compute CRC of new file
			if (fwrite(oData.data(), 1, nb, dataFile) != nb)
				throw std::exception("Cannot write to file");

		} while (bytes -= nb);

		fclose(dataFile);
		fclose(codeFile);
		if (crc != new_crc)
			throw std::exception("Incorrect file CRC");
	}

	static void CompareFiles(const char* file1name, const char* file2name)
	{
		FILE* file1 = Open_Input_File(file1name);
		FILE* file2 = Open_Input_File(file2name);

		unsigned char buffer1[1024];
		unsigned char buffer2[1024];

		unsigned nb1;
		unsigned nb2;

		do
		{
			nb1 = fread(buffer1, 1, 1024, file1);
			nb2 = fread(buffer2, 1, 1024, file2);
			if (nb1 != nb2)
				throw std::exception("File sizes do not match!");
			if (memcmp(buffer1, buffer2, nb1) != 0)
				throw std::exception("File contents do not match!");
		} while (nb1 > 0);

		std::cout << "Files are identical." << std::endl;
		fclose(file1);
		fclose(file2);
	}

	static float GetZerosProbability(const char* file1name)
	{
		FILE* file1 = Open_Input_File(file1name);
		unsigned char buffer1[1024];

		unsigned nb1;
		uint64_t totalSize = 0;
		uint64_t onesCount = 0;

		do
		{
			nb1 = fread(buffer1, 1, 1024, file1);
			for (int i = 0; i < nb1; i++) {
				unsigned char num = buffer1[i];
				totalSize += 8;
				while (num > 0)
				{
					if (num % 2 == 1)
						onesCount++;
					num = num / 2;
				}
			}
		} while (nb1 > 0);
		fclose(file1);

		return (float)(totalSize - onesCount) / (float)totalSize;
	}
};