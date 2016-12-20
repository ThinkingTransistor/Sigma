﻿/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp.Config;
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

			return new BackendConfig<float>(this.DiffsharpBackendHandle, epsilon, 1.0f / epsilon, 0.5f / epsilon, fpeps, 100, 1.2f);
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
			ADNDFloat32Array internalArray = (ADNDFloat32Array) array;
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle + internalValue);
		}

		public INDArray Add(INDArray array, INumber value)
		{
			throw new NotImplementedException();
		}

		public INDArray Add(INDArray a, INDArray b)
		{
			ADNDFloat32Array internalA = (ADNDFloat32Array) a;
			ADNDFloat32Array internalB = (ADNDFloat32Array) b;

			return new ADNDFloat32Array(internalA._adArrayHandle + internalB._adArrayHandle);
		}

		public INDArray Subtract<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = (ADNDFloat32Array) array;
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalArray._adArrayHandle - internalValue);
		}

		public INDArray Subtract(INDArray array, INumber value)
		{
			throw new NotImplementedException();
		}

		public INDArray Subtract(INDArray a, INDArray b)
		{
			throw new NotImplementedException();
		}

		public INDArray Multiply<TOther>(INDArray array, TOther value)
		{
			throw new NotImplementedException();
		}

		public INDArray Multiply(INDArray array, INumber value)
		{
			throw new NotImplementedException();
		}

		public INDArray Multiply(INDArray a, INDArray b)
		{
			throw new NotImplementedException();
		}

		public INDArray Dot(INDArray a, INDArray b)
		{
			throw new NotImplementedException();
		}

		public INDArray Divide<TOther>(INDArray array, TOther value)
		{
			throw new NotImplementedException();
		}

		public INDArray Divide(INDArray array, INumber value)
		{
			throw new NotImplementedException();
		}

		public INDArray Divide(INDArray a, INDArray b)
		{
			throw new NotImplementedException();
		}

		public INDArray MergeBatch(params INDArray[] arrays)
		{
			ADNDArray<float>[] castArrays = arrays.As<INDArray, ADNDArray<float>>();

			long[] totalShape = new long[castArrays[0].Rank];

			System.Array.Copy(arrays[0].Shape, 1, totalShape, 1, totalShape.Length - 1);

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