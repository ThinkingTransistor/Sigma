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

__global__ void Add_V_V(const float* a, int aOffset, float* b, int bOffset, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i + bOffset] = a[i + aOffset] + b[i + bOffset];
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

__global__ void Exp_V(float* a, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = __expf(a[i]);
	}
}

__global__ void Sqrt_V(float* a, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = sqrtf(a[i]);
	}
}

__global__ void Sign_V(float* a, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = copysignf(1.0f, a[i]);
	}
}

__global__ void Rel_V(float* a, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = (copysignf(a[i], 1.0f) + a[i]) / 2.0f;
	}
}

__global__ void Log_V(float* a, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = __logf(a[i]);
	}
}

int main()
{
	// do nothing

	return 0;
}