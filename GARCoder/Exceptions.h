#pragma once
#include <exception>
#include "cuda_runtime.h"

namespace GARCoder
{
	class NotImplementedException : public std::exception
	{
	public:
		NotImplementedException(char* message);
		virtual ~NotImplementedException(void);
	};

	class CudaException : public std::exception
	{
	protected:
		cudaError_t error;
	public:
		CudaException(cudaError_t errNo);
		virtual ~CudaException(void);
		virtual const char* what() const;
	};

	class GPUCoderWrongInputException : public std::exception
	{
	public:
		GPUCoderWrongInputException();
		virtual ~GPUCoderWrongInputException(void);
		virtual const char* what() const;
	};
}



