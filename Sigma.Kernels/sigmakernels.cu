#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>

__global__ void Sub_V_S(float *a, const float b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = a[i] - b;
	}
}

__global__ void Sub_S_V(const float a, float* b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a - b[i];
	}
}

__global__ void Add_V_S(float* a, const float b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = a[i] + b;
	}
}

__global__ void Mul_Had_V_V(const float* a, float* b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a[i] * b[i];
	}
}

int main()
{
	// do nothing

	return 0;
}