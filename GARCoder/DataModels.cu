#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "_Constants.h"

class StaticBinaryModel
{							
public:
	unsigned bit0Prob;		// probability of zero symbol
	StaticBinaryModel() {}
	virtual ~StaticBinaryModel() {}
	virtual void SetZeroProbability(float bit0prob) {
		if ((bit0prob < 0.0001) || (bit0prob > 0.9999))
			throw "invalid bit probability";
		bit0Prob = unsigned(bit0prob * (1 << BM_LengthShift));
	}
};

class SimpleAdaptiveBinaryModel
{
public:
	__device__ __host__ SimpleAdaptiveBinaryModel() {
		reset();
	}
	__device__ __host__ virtual ~SimpleAdaptiveBinaryModel() {

	}
	__device__ __host__ void reset() {
		// initialization to equiprobable model
		bit0Count = 1;
		bitCount = 2;
		bit0Prob = 1U << (BM_LengthShift - 1);
	}
	__device__ __host__ void update() {
		// halve counts when a threshold is reached
		if ((bitCount++) > BM_MaxCount) {
			bitCount = (bitCount + 1) >> 1;
			bit0Count = (bit0Count + 1) >> 1;
			if (bit0Count == bitCount) ++bitCount;
		}
		// compute scaled bit 0 probability
		unsigned scale = 0x80000000U / bitCount;
		bit0Prob = (bit0Count * scale) >> (31 - BM_LengthShift);
	}
	unsigned bit0Prob, bit0Count, bitCount;
};

class AdaptiveBinaryModel
{
public:
	__device__ __host__ AdaptiveBinaryModel() {
		reset();
	}
	__device__ __host__ virtual ~AdaptiveBinaryModel() {
		
	}
	__device__ __host__ void reset() {
		// initialization to equiprobable model
		bit0Count = 1;
		bitCount = 2;
		bit0Prob = 1U << (BM_LengthShift - 1);
		updateCycle = bitsUntilUpdate = 4;         // start with frequent updates
	}
	__device__ __host__ void update() {
		// halve counts when a threshold is reached

		if ((bitCount += updateCycle) > BM_MaxCount) {
			bitCount = (bitCount + 1) >> 1;
			bit0Count = (bit0Count + 1) >> 1;
			if (bit0Count == bitCount) ++bitCount;
		}
		// compute scaled bit 0 probability
		unsigned scale = 0x80000000U / bitCount;
		bit0Prob = (bit0Count * scale) >> (31 - BM_LengthShift);

		// set frequency of model updates
		updateCycle = (5 * updateCycle) >> 2;
		if (updateCycle > 64) updateCycle = 64;
		bitsUntilUpdate = updateCycle;
	}
	unsigned updateCycle, bitsUntilUpdate;
	unsigned bit0Prob, bit0Count, bitCount;
};

class AdaptiveByteModel
{
public:
	AdaptiveByteModel() {
		data_symbols = 0;
		distribution = 0;
		set_alphabet();
	}
	virtual ~AdaptiveByteModel() {
		delete[] distribution;
	}
	
	void reset() {
		if (data_symbols == 0) return;

		// restore probability estimates to uniform distribution
		total_count = 0;
		update_cycle = data_symbols;
		for (unsigned k = 0; k < data_symbols; k++) symbol_count[k] = 1;
		update(false);
		symbols_until_update = update_cycle = (data_symbols + 6) >> 1;
	}
	void set_alphabet() {
		unsigned number_of_symbols = 256;

		if (data_symbols != number_of_symbols) {     // assign memory for data model
			data_symbols = number_of_symbols;
			last_symbol = data_symbols - 1;
			delete[] distribution;
			
			// define size of table for fast decoding
			unsigned table_bits = 3;
			while (data_symbols > (1U << (table_bits + 2))) ++table_bits;
			table_size = (1 << table_bits) + 4;
			table_shift = CM_LengthShift - table_bits;
			distribution = new unsigned[2 * data_symbols + table_size + 6];
			decoder_table = distribution + 2 * data_symbols;

			symbol_count = distribution + data_symbols;
			if (distribution == 0) throw "cannot assign model memory";
		}
		reset();                                                 // initialize model
	}
	void update(bool from_encoder) {
		// halve counts when a threshold is reached
		if ((total_count += update_cycle) > CM_MaxCount) {
			total_count = 0;
			for (unsigned n = 0; n < data_symbols; n++)
				total_count += (symbol_count[n] = (symbol_count[n] + 1) >> 1);
		}
		// compute cumulative distribution, decoder table
		unsigned k, sum = 0, s = 0;
		unsigned scale = 0x80000000U / total_count;

		if (from_encoder || (table_size == 0))
			for (k = 0; k < data_symbols; k++) {
				distribution[k] = (scale * sum) >> (31 - CM_LengthShift);
				sum += symbol_count[k];
			}
		else {
			for (k = 0; k < data_symbols; k++) {
				distribution[k] = (scale * sum) >> (31 - CM_LengthShift);
				sum += symbol_count[k];
				unsigned w = distribution[k] >> table_shift;
				while (s < w) decoder_table[++s] = k - 1;
			}
			decoder_table[0] = 0;
			while (s <= table_size) decoder_table[++s] = data_symbols - 1;
		}
		// set frequency of model updates
		update_cycle = (5 * update_cycle) >> 2;
		unsigned max_cycle = (data_symbols + 6) << 3;
		if (update_cycle > max_cycle) update_cycle = max_cycle;
		symbols_until_update = update_cycle;
	}


	unsigned *distribution, *symbol_count, *decoder_table;
	unsigned total_count, update_cycle, symbols_until_update;
	unsigned data_symbols, last_symbol, table_size, table_shift;
};

class ContextAdaptiveBinaryModel
{
public:
	__device__ __host__ ContextAdaptiveBinaryModel() {
		models = new AdaptiveBinaryModel[256];
	}
	__device__ __host__ virtual ~ContextAdaptiveBinaryModel() {
		delete[] models;
	}
	__device__ __host__ ContextAdaptiveBinaryModel(const ContextAdaptiveBinaryModel& copy) {
		models = new AdaptiveBinaryModel[256];
	}
	__device__ __host__ ContextAdaptiveBinaryModel(ContextAdaptiveBinaryModel& copy) {
		models = new AdaptiveBinaryModel[256];
	}
	__device__ __host__ void reset() {
		for (int i = 0; i < 256; ++i)
			models[i].reset();
	}
	AdaptiveBinaryModel* models;
};