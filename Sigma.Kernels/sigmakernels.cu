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

__global__ void Div_S_V(const float a, float* b, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a / b[i];
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
		a[i] = (fabsf(a[i]) + a[i]) / 2.0f;
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

__global__ void Sum_V(const float* a, float* partial_sums, int n) 
{
	 extern __shared__ float sdata[]; 

	 int i = threadIdx.x + blockIdx.x * blockDim.x;
	 int ti = threadIdx.x;

	 // move global input data to shared memory, pad with zeros
	 float x = 0.0f;
	 if (i < n)
	 {
		x = a[i];
	 }
	 sdata[ti] = x;

	 __syncthreads();

	 // use parallel reduction to contiguously reduce to partial sums
	 for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) 
	 {
		if (ti < offset)
		{
			sdata[ti] += sdata[ti + offset];
		}

		__syncthreads();
	 }

	 if (ti == 0) 
	 {
		partial_sums[blockIdx.x] = sdata[0];
	 }
}

int main()
{
	// do nothing

	return 0;
}