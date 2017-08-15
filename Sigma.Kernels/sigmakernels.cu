#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
#include "curand_kernel.h"
#include "math.h"

#include <stdio.h>

__global__ void Sub_V_S(const float *a, const float b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a[i] - b;
	}
}

__global__ void Sub_S_V(const float a, float* b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a - b[i];
	}
}

__global__ void Add_V_S(const float* a, const float b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a[i] + b;
	}
}

__global__ void Add_V_V_InPlace(const float* a, int aOffset, float* b, int bOffset, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		b[i + bOffset] = a[i + aOffset] + b[i + bOffset];
	}
}

__global__ void Mul_Had_V_V(const float* a, const float* b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a[i] * b[i];
	}
}

__global__ void Div_S_V(const float a, const float* b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a / b[i];
	}
}

__global__ void Div_V_V(const float* a, const float* b, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = a[i] / b[i];
	}
}

__global__ void Exp_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = __expf(a[i]);
	}
}

__global__ void Sqrt_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = sqrtf(a[i]);
	}
}

__global__ void Sign_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = copysignf(1.0f, a[i]);
	}
}

__global__ void Rel_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = (fabsf(a[i]) + a[i]) / 2.0f;
	}
}

__global__ void Log_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = __logf(a[i]);
	}
}

__global__ void Sigmoid_V(const float* a, float* out, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		out[i] = 1.0f / (1.0f + __expf(-a[i]));
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

__global__ void Softmax_Rowwise_M(const float* a, float* maxPerRow, float* maxPerRowIndices, float* sumPerRow, const int rows, const int cols, const int cols2, float* out, const int n)
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

	// write out max index 
	if (maxPerRow[ri] == a[i])
	{
		maxPerRowIndices[ri] = tiLocal;
	}

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
		out[i] = sdata[ti] / rowBuffer[riLocal * cols];

		if (tiLocal == 0)
		{
			sumPerRow[ri] = rowBuffer[riLocal * cols];
		}
	}
}

__global__ void Softmax_Rowwise_M_Backward(const float* origin, const float* adjoint, const float* primal, const float* prevMaxs, const float* prevMaxIndices, 
											const float* prevSums, float* out, const int rows, const int cols, const int cols2, const int n)
{
	extern __shared__ float sdata[];
	float* rowBuffer = sdata;
	float* originData = &sdata[blockDim.x];
	float* adjointData = &sdata[blockDim.x * 2];
	float* primalData = &sdata[blockDim.x * 3];
	float* outData = &sdata[blockDim.x * 4];

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
	int prevMaxIndex = prevMaxIndices[ri];
	float prevSum = prevSums[ri];

	if (inData)
	{
		originData[ti] = origin[i];
		adjointData[ti] = adjoint[i];
		primalData[ti] = primal[i];
	}

	// Div_DM_D				DM (direct)						 D (indirect via Sum_DM)
	rowBuffer[ti] = adjointData[ti] / prevSum + adjointData[ti] * (originData[ti] / (prevSum * prevSum));
	
	// Exp_DM				DM (direct)
	rowBuffer[ti] = rowBuffer[ti] * __expf(originData[ti] - prevMax);
	outData[ti] = rowBuffer[ti];

	__syncthreads();

	// calculate each rows derivatives (in rowBuffer) sum
	for (int offset = cols2 / 2; offset > 0; offset >>= 1) 
	{
		if (tiLocal < offset)
		{
			float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

			rowBuffer[ti] = rowBuffer[ti] + other;
		}

		__syncthreads();
	}

	// Item_DM		D (indirect via Max op via Sub_DM_D op (left part for DM is just passthrough of gradient, so nothing to do there))
	if (tiLocal == prevMaxIndex)
	{
		outData[ti] = outData[ti] - rowBuffer[riLocal * cols]; 
	}

	if (inData)
	{
		out[i] = outData[ti];
	}
}

__global__ void Sum_M_Rowwise(const float* a, const int rows, const int cols, const int cols2, float* sumPerRowPerBlock, const int n)
{
	extern __shared__ float sdata[];

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
	sdata[ti] = x;

	__syncthreads();

	// calculate each rows derivatives (in sdata) sum
	for (int offset = cols2 / 2; offset > 0; offset >>= 1) 
	{
		if (tiLocal < offset)
		{
			float other = (ti + offset) / cols == riLocal ? sdata[ti + offset] : 0.0f;

			sdata[ti] = sdata[ti] + other;
		}

		__syncthreads();
	}

	if (tiLocal == 0)
	{
		sumPerRowPerBlock[blockIdx.x * rowsPerBlock + riLocal] = sdata[riLocal * cols];
	}
}

__global__ void Add_M_Rowwise_V_InPlace(const float* a, const int rows, const int cols, const int cols2, float* b, const int n)
{
	extern __shared__ float sdata[];

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
	sdata[ti] = x;

	__syncthreads();

	// calculate each rows derivatives (in sdata) sum
	for (int offset = cols2 / 2; offset > 0; offset >>= 1) 
	{
		if (tiLocal < offset)
		{
			float other = (ti + offset) / cols == riLocal ? sdata[ti + offset] : 0.0f;

			sdata[ti] = sdata[ti] + other;
		}

		__syncthreads();
	}

	if (tiLocal == 0)
	{
		b[ri] = b[ri] + sdata[riLocal * cols];
	}
}

__global__ void RepeatReshapeCopy_V_MRows(const float* a, float* b, const int rows, const int cols, const int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 

	float value = a[i % cols];
	while (i < n)
	{
		b[i] = value;

		i += blockDim.x;
	}
}

__device__ curandState randomStates[256];

__global__ void InitialiseRandomStates(int seed) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	
	if (i < 256)
	{
		curand_init(seed + i, i, 0, &randomStates[i]);
	}
}

__global__ void FillWithProbabilityMask_V(float* a, const float probability, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 

	if (i < n)
	{
		float rand = curand_uniform(&randomStates[i % 256]);

		a[i] = rand < probability ? 1 : 0;
	}
}

int main()
{
	// do nothing

	return 0;
}