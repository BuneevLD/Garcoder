#include "Exceptions.h"


namespace GARCoder
{
	NotImplementedException::NotImplementedException(char* message = "Not implemented function was called")
		: exception(message)
	{
	}

	NotImplementedException::~NotImplementedException(void)
	{
	}

	CudaException::CudaException(cudaError_t errNo) : error(errNo)
	{
	}
	CudaException::~CudaException()
	{
	}
	const char* CudaException::what() const
	{
		return cudaGetErrorString(error);
	}

	GPUCoderWrongInputException::GPUCoderWrongInputException()
	{
	}
	GPUCoderWrongInputException::~GPUCoderWrongInputException(void)
	{
	}
	const char* GPUCoderWrongInputException::what() const
	{
		return "GPUcodec supports only input with size that is multiply of chunkSize. Consider using GARCodec instead.";
	}
}