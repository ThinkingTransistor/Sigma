#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>

__global__ void Sub_V_S(float *a, const float b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = a[i] - b;
	}
}

__global__ void Sub_S_V(const float a, float* b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a - b[i];
	}
}

__global__ void Add_V_S(float* a, const float b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = a[i] + b;
	}
}

__global__ void Add_V_V(const float* a, int aOffset, float* b, int bOffset, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i + bOffset] = a[i + aOffset] + b[i + bOffset];
	}
}

__global__ void Mul_Had_V_V(const float* a, float* b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a[i] * b[i];
	}
}

__global__ void Div_S_V(const float a, float* b, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i] = a / b[i];
	}
}

__global__ void Exp_V(float* a, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = __expf(a[i]);
	}
}

__global__ void Sqrt_V(float* a, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = sqrtf(a[i]);
	}
}

__global__ void Sign_V(float* a, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = copysignf(1.0f, a[i]);
	}
}

__global__ void Rel_V(float* a, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = (fabsf(a[i]) + a[i]) / 2.0f;
	}
}

__global__ void Log_V(float* a, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		a[i] = __logf(a[i]);
	}
}

__global__ void Sum_V(const float* a, float* partial_sums, const int n) 
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

__global__ void Softmax_Rowwise_M(float* a, float* maxPerRow, float* sumPerRow, const int rows, const int cols, const int cols2, const int n)
{
	extern __shared__ float sdata[];
	float* rowBuffer = &sdata[blockDim.x];

	int rowsPerBlock = blockDim.x / cols;
	int usedPerBlock = rowsPerBlock * cols;
	int unusedPerBlock = blockDim.x - usedPerBlock;

	int ti = threadIdx.x;
	int i = ti + blockIdx.x * blockDim.x - (unusedPerBlock * blockIdx.x);
	int ri = i / cols;
	int riLocal = ri % rowsPerBlock;
	int tiLocal = ti - riLocal * cols;
	bool inData = i < n && ti < usedPerBlock;

	float x = 0.0f;
	if (inData)
	{
		x = a[i];
	}
	sdata[ti] = rowBuffer[ti] = x;

	__syncthreads();

	// find each rows max value
	for (int offset = cols2 / 2; offset > 0; offset >>= 1) 
	{
		if (tiLocal < offset)
		{
			float currentMax = rowBuffer[ti];
			float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

			rowBuffer[ti] = other > currentMax ? other : currentMax;
		}

		__syncthreads();
	}

	// subtract each value from that row's maximum
	if (inData)
	{
		sdata[ti] = __expf(sdata[ti] - rowBuffer[riLocal * cols]);
	
		if (tiLocal == 0)
		{
			maxPerRow[ri] = rowBuffer[riLocal * cols];
		}
	}
	rowBuffer[ti] = sdata[ti];

	__syncthreads();

	// calculate each rows sum
	for (int offset = cols2 / 2; offset > 0; offset >>= 1) 
	{
		if (tiLocal < offset)
		{
			float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

			rowBuffer[ti] = rowBuffer[ti] + other;
		}

		__syncthreads();
	}

	if (inData)
	{
		a[i] = sdata[ti] / rowBuffer[riLocal * cols];

		if (tiLocal == 0)
		{
			sumPerRow[ri] = rowBuffer[riLocal * cols];
		}
	}
}

__global__ void Softmax_Rowwise_M_Backward(const float* origin, const float* adjoint, const float* primal, const float* prevMaxs, const float* prevSums, 
											float* out, const int rows, const int cols, const int cols2, const int n)
{
	extern __shared__ float sdata[];
	float* rowBuffer = &sdata[blockDim.x];
	float* originData = &sdata[blockDim.x * 2];
	float* adjointData = &sdata[blockDim.x * 3];
	float* primalData = &sdata[blockDim.x * 4];
	float* outData = &sdata[blockDim.x * 5];

	int rowsPerBlock = blockDim.x / cols;
	int usedPerBlock = rowsPerBlock * cols;
	int unusedPerBlock = blockDim.x - usedPerBlock;

	int ti = threadIdx.x;
	int i = ti + blockIdx.x * blockDim.x - (unusedPerBlock * blockIdx.x);
	int ri = i / cols;
	int riLocal = ri % rowsPerBlock;
	int tiLocal = ti - riLocal * cols;
	bool inData = i < n && ti < usedPerBlock;

	float prevMax = prevMaxs[ri];
	float prevSum = prevSums[ri];

	if (inData)
	{
		originData[ti] = origin[i];
		adjointData[ti] = adjoint[i];
		primalData[ti] = primal[i];
	}

	rowBuffer[ti] = adjointData[ti] * (originData[ti] / (prevSum * prevSum)) + adjointData[ti] / prevSum;

	__syncthreads();

	// TODO complete backprop
}


int main()
{
	// do nothing

	return 0;
}