/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp;
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
		public abstract INDArray NDArray<TOther>(TOther[] values, params long[] shape);
		public abstract INumber Number(object value);
		public abstract IDataBuffer<T> DataBuffer<T>(T[] values);
		public abstract bool CanConvert(INDArray array, IComputationHandler otherHandler);
		public abstract INDArray Convert(INDArray array, IComputationHandler otherHandler);
		public abstract void Fill(INDArray filler, INDArray arrayToFill);
		public abstract void Fill<TOther>(TOther value, INDArray arrayToFill);

		public INDArray Add<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = (ADFloat32Number) Number((float) System.Convert.ChangeType(value, typeof(float)));

			return new ADNDFloat32Array(internalArray._adArrayHandle + internalValue._adNumberHandle);
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

		public INumber Add(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle + internalB._adNumberHandle);
		}

		public INumber Add<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle + internalB);
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

		public INumber Subtract(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle - internalB._adNumberHandle);
		}

		public INumber Subtract<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle - internalB);
		}

		public INumber Subtract<TOther>(TOther a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalB - internalA._adNumberHandle);
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

		public INumber Multiply(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle * internalB._adNumberHandle);
		}

		public INumber Multiply<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle * internalB);
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

		public INumber Divide(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle / internalB._adNumberHandle);
		}

		public INumber Divide<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle / internalB);
		}

		public INDArray Pow(INDArray array, INumber value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADNDFloat32Array(DNDArray.Pow(internalArray._adArrayHandle, internalValue._adNumberHandle));
		}

		public INDArray Pow<TOther>(INDArray array, TOther value)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float) System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(DNDArray.Pow(internalArray._adArrayHandle, internalValue));
		}

		public INumber Pow(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(DNumber.Pow(internalA._adNumberHandle, internalB._adNumberHandle));
		}

		public INumber Pow<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float) System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(DNumber.Pow(internalA._adNumberHandle, internalB));
		}

		public INumber Abs(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Abs(internalValue._adNumberHandle));
		}

		public INDArray Abs(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Abs(internalArray._adArrayHandle));
		}

		public INumber Sum(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Sum(internalArray._adArrayHandle));
		}

		public INumber Max(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(new DNumber(internalArray.Data.GetValue(DNDArray.MaxIndex(internalArray._adArrayHandle))));
		}

		public INumber Min(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(new DNumber(internalArray.Data.GetValue(DNDArray.MinIndex(internalArray._adArrayHandle))));
		}

		public INDArray Sqrt(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Sqrt(internalArray._adArrayHandle));
		}

		public INumber Sqrt(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sqrt(internalValue._adNumberHandle));
		}

		public INDArray Log(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Log(internalArray._adArrayHandle));
		}

		public INumber Log(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Log(internalValue._adNumberHandle));
		}

		public INumber Determinate(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Det(internalArray._adArrayHandle));
		}

		public INDArray Sin(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Sin(internalArray._adArrayHandle));
		}

		public INumber Sin(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sin(internalValue._adNumberHandle));
		}

		public INDArray Asin(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Asin(internalArray._adArrayHandle));
		}

		public INumber Asin(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Asin(internalValue._adNumberHandle));
		}

		public INDArray Cos(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Cos(internalArray._adArrayHandle));
		}

		public INumber Cos(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Cos(internalValue._adNumberHandle));
		}

		public INDArray Acos(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Acos(internalArray._adArrayHandle));
		}

		public INumber Acos(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Acos(internalValue._adNumberHandle));
		}

		public INDArray Tan(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Tan(internalArray._adArrayHandle));
		}

		public INumber Tan(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Tan(internalValue._adNumberHandle));
		}

		public INDArray Atan(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Atan(internalArray._adArrayHandle));
		}

		public INumber Atan(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Atan(internalValue._adNumberHandle));
		}

		public INDArray ReL(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.ReLU(internalArray._adArrayHandle));
		}

		public INumber ReL(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.ReLU(internalValue._adNumberHandle));
		}

		public INDArray Sigmoid(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Sigmoid(internalArray._adArrayHandle));
		}

		public INumber Sigmoid(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sigmoid(internalValue._adNumberHandle));
		}

		public INDArray SoftPlus(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.SoftPlus(internalArray._adArrayHandle));
		}

		public INumber SoftPlus(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.SoftPlus(internalValue._adNumberHandle));
		}

		public INDArray Tanh(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Tanh(internalArray._adArrayHandle));
		}

		public INumber Tanh(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Tanh(internalValue._adNumberHandle));
		}

		public INumber Activation(string activation, INumber number)
		{
			return ActivationManager.ApplyActivation(activation, number, this);
		}

		public INDArray Activation(string activation, INDArray array)
		{
			return ActivationManager.ApplyActivation(activation, array, this);
		}

		public INumber StandardDeviation(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.StandardDev(internalArray._adArrayHandle));
		}

		public INumber Variance(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Variance(internalArray._adArrayHandle));
		}

		public uint BeginTrace()
		{
			return Util.GlobalTagger.Next;
		}

		public TTraceable Trace<TTraceable>(TTraceable traceable, uint traceTag) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable) ((object) new ADFloat32Number(internalNumber._adNumberHandle.GetReverse(traceTag)));
			}
			else if (traceable is ADNDFloat32Array)
			{
				ADNDFloat32Array internalArray = traceable as ADNDFloat32Array;

				return (TTraceable) ((object) new ADNDFloat32Array(internalArray._adArrayHandle.GetReverse(traceTag)));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		public TTraceable ClearTrace<TTraceable>(TTraceable traceableRoot) where TTraceable : ITraceable
		{
			if (traceableRoot is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceableRoot as ADFloat32Number;

				return (TTraceable) ((object) new ADFloat32Number(internalNumber._adNumberHandle.P));
			}
			else if (traceableRoot is ADNDFloat32Array)
			{
				ADNDFloat32Array internalArray = traceableRoot as ADNDFloat32Array;

				return (TTraceable) ((object) new ADNDFloat32Array(internalArray._adArrayHandle.P));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceableRoot} not compatible with this handler).");
			}
		}

		public void ComputeDerivativesTo(ITraceable traceable)
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number number = (ADFloat32Number) traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, number._adNumberHandle.asADD);
			}
			else if (traceable is ADNDFloat32Array)
			{
				ADNDFloat32Array array = (ADNDFloat32Array) traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, array._adArrayHandle.asADDND);
			}
			else
			{
				throw new InvalidOperationException($"Cannot compute derivatives for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		public TTraceable GetDerivative<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable) ((object) new ADFloat32Number(internalNumber._adNumberHandle.A));
			}
			else if (traceable is ADNDFloat32Array)
			{
				ADNDFloat32Array internalArray = traceable as ADNDFloat32Array;

				return (TTraceable) ((object) new ADNDFloat32Array(internalArray._adArrayHandle.A));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		public INDArray MergeBatch(params INDArray[] arrays)
		{
			ADNDFloat32Array[] castArrays = arrays.As<INDArray, ADNDFloat32Array>();

			long[] totalShape = new long[castArrays[0].Rank];

			Array.Copy(arrays[0].Shape, 1, totalShape, 1, totalShape.Length - 1);

			foreach (ADNDFloat32Array array in castArrays)
			{
				totalShape[0] += array.Shape[0];
			}

			ADNDFloat32Array merged = new ADNDFloat32Array(_backendTag, totalShape);
			DataBuffer<float> mergedData = (DataBuffer<float>) merged.Data;

			long lastIndex = 0L;
			foreach (ADNDFloat32Array array in castArrays)
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
