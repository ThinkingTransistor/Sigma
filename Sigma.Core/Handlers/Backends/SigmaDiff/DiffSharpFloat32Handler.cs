/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp.Config;
using DiffSharp.Interop.Float32;
using log4net;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu;
using Sigma.Core.Utils;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// An abstract DiffSharp computation handle for 32-bit floats with dynamic Blas and Lapack backends.
	/// </summary>
	public abstract class DiffSharpFloat32Handler : IComputationHandler
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public IBlasBackend BlasBackend { get; }
		public ILapackBackend LapackBackend { get; }

		public abstract IDataType DataType { get; }

		internal DiffSharpBackendHandle<float> DiffsharpBackendHandle { get; }

		private readonly long _backendTag;

		protected DiffSharpFloat32Handler(IBlasBackend blasBackend, ILapackBackend lapackBackend)
		{
			if (blasBackend == null) throw new ArgumentNullException(nameof(blasBackend));
			if (lapackBackend == null) throw new ArgumentNullException(nameof(lapackBackend));

			BlasBackend = blasBackend;
			LapackBackend = lapackBackend;

			DiffsharpBackendHandle = new DiffSharpFloat32BackendHandle(blasBackend, lapackBackend, backendTag: -1);

			_backendTag = SigmaDiffSharpBackendProvider.Instance.Register(CreateBackendConfig());
			SigmaDiffSharpBackendProvider.AssignToDiffSharpGlobal();

			DiffsharpBackendHandle.BackendTag = _backendTag;
		}

		protected BackendConfig<float> CreateBackendConfig()
		{
			float epsilon = 0.00001f;
			float fpeps = 0.01f;

			return new BackendConfig<float>(DiffsharpBackendHandle, epsilon, 1.0f / epsilon, 0.5f / epsilon, fpeps, 100, 1.2f);
		}

		protected ADNDFloat32Array InternaliseArray(object array)
		{
			return AssignTag((ADNDFloat32Array) array);
		}

		protected ADFloat32Number InternaliseNumber(object number)
		{
			SigmaDiffSharpBackendProvider.Instance.MapToBackend(number, _backendTag);

			return (ADFloat32Number) number;
		}

		protected ADNDFloat32Array AssignTag(ADNDFloat32Array array)
		{
			((SigmaDiffDataBuffer<float>) array.Data).BackendTag = _backendTag;

			return array;
		}

		// IComputationHandler stuff that is probably different for each diffsharp handler implementation
		public abstract void InitAfterDeserialisation(INDArray array);
		public abstract long GetSizeBytes(params INDArray[] array);
		public abstract bool IsInterchangeable(IComputationHandler otherHandler);
		public abstract INDArray NDArray(params long[] shape);
		public abstract INumber Number(object value);
		public abstract IDataBuffer<T> DataBuffer<T>(T[] values);
		public abstract bool CanConvert(INDArray array, IComputationHandler otherHandler);
		public abstract INDArray Convert(INDArray array, IComputationHandler otherHandler);
		public abstract void Fill(INDArray filler, INDArray arrayToFill);
		public abstract void Fill<TOther>(TOther value, INDArray arrayToFill);

		public INDArray Add<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle + internalValue);
		}

		public INDArray Add(INDArray array, INumber value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(internalArray._adArrayHandle + internalValue._adNumberHandle);
		}

		public INDArray Add(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = InternaliseArray(a);
			ADNDFloat32Array internalB = InternaliseArray(b);

			return new ADNDFloat32Array(internalA._adArrayHandle + internalB._adArrayHandle);
		}

		public INDArray Subtract<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle - internalValue);
		}

		public INDArray Subtract(INDArray array, INumber value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(internalArray._adArrayHandle - internalValue._adNumberHandle);
		}

		public INDArray Subtract(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = InternaliseArray(a);
			ADNDFloat32Array internalB = InternaliseArray(b);

			return new ADNDFloat32Array(internalA._adArrayHandle - internalB._adArrayHandle);
		}

		public INDArray Subtract<TOther>(TOther value, INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalValue - internalArray._adArrayHandle);
		}

		public INDArray Subtract(INumber value, INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(internalValue._adNumberHandle - internalArray._adArrayHandle);
		}

		public INDArray Multiply<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle * internalValue);
		}

		public INDArray Multiply(INDArray array, INumber value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(internalValue._adNumberHandle * internalArray._adArrayHandle);
		}

		public INDArray Multiply(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = InternaliseArray(a);
			ADNDFloat32Array internalB = InternaliseArray(b);

			return new ADNDFloat32Array(DNDArray.op_DotMultiply(internalA._adArrayHandle, internalB._adArrayHandle));
		}

		public INDArray Dot(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = InternaliseArray(a);
			ADNDFloat32Array internalB = InternaliseArray(b);

			return new ADNDFloat32Array(internalA._adArrayHandle * internalB._adArrayHandle);
		}

		public INDArray Divide<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle / internalValue);
		}

		public INDArray Divide(INDArray array, INumber value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(internalArray._adArrayHandle / internalValue._adNumberHandle);
		}

		public INDArray Divide(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = InternaliseArray(a);
			ADNDFloat32Array internalB = InternaliseArray(b);

			return new ADNDFloat32Array(DNDArray.op_DotDivide(internalA._adArrayHandle, internalB._adArrayHandle));
		}

		public INDArray Pow(INDArray array, INumber value)
		{
			throw new NotImplementedException();
		}

		public INDArray Pow<TOther>(INDArray array, TOther value)
		{
			throw new NotImplementedException();
		}

		public INumber Abs(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Abs(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Sum(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Max(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Min(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber L1Norm(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber L2Norm(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Sqrt(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Sqrt(INumber array)
		{
			throw new NotImplementedException();
		}

		public INDArray Log(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Log(INumber array)
		{
			throw new NotImplementedException();
		}

		public INumber Determinate(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Sin(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Sin(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Asin(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Asin(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Cos(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Cos(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Acos(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Acos(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Tan(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Tan(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray Atan(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Atan(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray ReL(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber ReL(INumber array)
		{
			throw new NotImplementedException();
		}

		public INDArray Sigmoid(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Sigmoid(INumber array)
		{
			throw new NotImplementedException();
		}

		public INDArray SoftPlus(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber SoftPlus(INumber array)
		{
			throw new NotImplementedException();
		}

		public INumber StandardDeviation(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Variance(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INDArray Trace(INDArray array)
		{
			throw new NotImplementedException();
		}

		public INumber Trace(INumber number)
		{
			throw new NotImplementedException();
		}

		public INDArray MergeBatch(params INDArray[] arrays)
		{
			ADNDArray<float>[] castArrays = arrays.As<INDArray, ADNDArray<float>>();

			long[] totalShape = new long[castArrays[0].Rank];

			Array.Copy(arrays[0].Shape, 1, totalShape, 1, totalShape.Length - 1);

			foreach (ADNDArray<float> array in castArrays)
			{
				totalShape[0] += array.Shape[0];
			}

			ADNDArray<float> merged = new ADNDArray<float>(totalShape);
			DataBuffer<float> mergedData = (DataBuffer<float>) merged.Data;

			long lastIndex = 0L;
			foreach (ADNDArray<float> array in castArrays)
			{
				DataBuffer<float> arrayData = (DataBuffer<float>) array.Data;

				mergedData.SetValues(arrayData, 0, lastIndex, arrayData.Length);

				lastIndex += arrayData.Length;
			}

			return merged;
		}

		static DiffSharpFloat32Handler()
		{
			PlatformDependentUtils.CheckPlatformDependentLibraries();
		}

	}
}
