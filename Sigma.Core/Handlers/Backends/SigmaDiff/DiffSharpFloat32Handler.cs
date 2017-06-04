/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using DiffSharp;
using DiffSharp.Config;
using DiffSharp.Interop.Float32;
using log4net;
using Microsoft.FSharp.Core;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Persistence;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// An abstract DiffSharp computation handle for 32-bit floats with dynamic Blas and Lapack backends.
	/// </summary>
	[Serializable]
	public abstract class DiffSharpFloat32Handler : IComputationHandler, ISerialisationNotifier
	{
		public abstract IDataType DataType { get; }

		public IRegistry Registry { get; }

		public IBlasBackend BlasBackend { get; }
		public ILapackBackend LapackBackend { get; }

		internal DiffSharpBackendHandle<float> DiffsharpBackendHandle
		{
			get { return _diffsharpBackendHandle; }
			private set { _diffsharpBackendHandle = value; }
		}

		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private readonly Random _probabilityMaskRng;

		[NonSerialized]
		private DiffSharpBackendHandle<float> _diffsharpBackendHandle;
		private long _backendTag;

		protected DiffSharpFloat32Handler(IBlasBackend blasBackend, ILapackBackend lapackBackend)
		{
			if (blasBackend == null) throw new ArgumentNullException(nameof(blasBackend));
			if (lapackBackend == null) throw new ArgumentNullException(nameof(lapackBackend));

			BlasBackend = blasBackend;
			LapackBackend = lapackBackend;

			Registry = new Registry(tags: "handler");

			InitialiseBackend(blasBackend, lapackBackend);

			_probabilityMaskRng = new Random();
		}

		private void InitialiseBackend(IBlasBackend blasBackend, ILapackBackend lapackBackend)
		{
			DiffsharpBackendHandle = new DiffSharpFloat32BackendHandle(blasBackend, lapackBackend, backendTag: -1);
			_backendTag = SigmaDiffSharpBackendProvider.Instance.Register(CreateBackendConfig());
			SigmaDiffSharpBackendProvider.AssignToDiffSharpGlobal();
			DiffsharpBackendHandle.BackendTag = _backendTag;
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			InitialiseBackend(BlasBackend, LapackBackend);
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
		/// <inheritdoc />
		public abstract void InitAfterDeserialisation(INDArray array);
		/// <inheritdoc />
		public abstract long GetSizeBytes(params INDArray[] array);
		/// <inheritdoc />
		public abstract bool IsInterchangeable(IComputationHandler otherHandler);
		/// <inheritdoc />
		public abstract INDArray NDArray(params long[] shape);
		/// <inheritdoc />
		public abstract INDArray NDArray<TOther>(TOther[] values, params long[] shape);
		/// <inheritdoc />
		public abstract INumber Number(object value);
		/// <inheritdoc />
		public abstract IDataBuffer<T> DataBuffer<T>(T[] values);
		/// <inheritdoc />
		public abstract INDArray AsNDArray(INumber number);
		/// <inheritdoc />
		public abstract INumber AsNumber(INDArray array, params long[] indices);
		/// <inheritdoc />
		public abstract bool CanConvert(INDArray array, IComputationHandler otherHandler);
		/// <inheritdoc />
		public abstract INDArray Convert(INDArray array, IComputationHandler otherHandler);
		/// <inheritdoc />
		public abstract void Fill(INDArray filler, INDArray arrayToFill);
		/// <inheritdoc />
		public abstract void Fill<TOther>(TOther value, INDArray arrayToFill);
		/// <inheritdoc />
		public abstract void Fill(INDArray filler, INDArray arrayToFill, long[] sourceBeginIndices, long[] sourceEndIndices, long[] destinationBeginIndices, long[] destinationEndIndices);
		/// <inheritdoc />
		public abstract void Fill<T>(T[] filler, INDArray arrayToFill, long[] destinationBeginIndices, long[] destinationEndIndices);

		protected ADNDFloat32Array ConvertInternal(INDArray array)
		{
			return new ADNDFloat32Array(_backendTag, array.GetDataAs<float>(), array.Shape);
		}

		public INDArray FlattenFeatures(INDArray array)
		{
			return array.Reshape(array.Shape[0], array.Shape[1], ArrayUtils.Product(2, array.Shape));
		}

		public INDArray FlattenTime(INDArray array)
		{
			long[] newShape = new long[array.Shape.Length - 1];
			newShape[0] = checked(array.Shape[0] * array.Shape[1]);

			for (var i = 0; i < newShape.Length; i++)
			{
				newShape[i] = array.Shape[i + 1];
			}

			return array.Reshape(newShape);
		}

		public INDArray FlattenTimeAndFeatures(INDArray array)
		{
			return array.Reshape(array.Shape[0] * array.Shape[1], ArrayUtils.Product(2, array.Shape));
		}

		public INDArray FlattenAllButLast(INDArray array)
		{
			return array.Reshape(ArrayUtils.Product(0, array.Rank - 1, array.Shape), array.Shape[array.Rank - 1]);
		}

		public TOther[] RowWiseTransform<TOther>(INDArray array, Func<INDArray, TOther> transformFunction)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			INDArray[] rows = SliceRowWise(array, internalArray);
			TOther[] transformedRows = new TOther[rows.Length];

			for (int i = 0; i < rows.Length; i++)
			{
				transformedRows[i] = transformFunction.Invoke(rows[i]);
			}

			return transformedRows;
		}

		public INDArray RowWise(INDArray array, Func<INDArray, INDArray> function)
		{
			// no need to slice if there's only one row
			if (array.Shape[0] == 1)
			{
				return function.Invoke(array);
			}

			ADNDFloat32Array internalArray = InternaliseArray(array);
			INDArray[] rows = SliceRowWise(array, internalArray);

			for (int i = 0; i < rows.Length; i++)
			{
				rows[i] = function.Invoke(rows[i]);
			}

			DNDArray[] internalRowHandles = new DNDArray[rows.Length];
			for (int i = 0; i < rows.Length; i++)
			{
				internalRowHandles[i] = InternaliseArray(rows[i])._adArrayHandle;
			}

			return new ADNDFloat32Array(new DNDArray(DNDArray.OfRows(internalRowHandles, DiffsharpBackendHandle)));
		}

		private static INDArray[] SliceRowWise(INDArray array, ADNDFloat32Array internalArray)
		{
			// no need to slice if there's only one row
			if (array.Shape[0] == 1)
			{
				return new[] { array };
			}

			INDArray[] rows = new INDArray[array.Shape[0]];

			var colStart = FSharpOption<int>.Some(0);
			var colFinish = FSharpOption<int>.Some(checked((int) array.Shape[1] - 1));

			for (var i = 0; i < rows.Length; i++)
			{
				var row = FSharpOption<int>.Some(i);

				rows[i] = new ADNDFloat32Array(internalArray._adArrayHandle.GetSlice(row, row, colStart, colFinish));
			}

			return rows;
		}

		public INDArray GetSlice(INDArray array, int rowIndex, int columnIndex, int rowLength, int columnLength)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			FSharpOption<int> rowStart = FSharpOption<int>.Some(rowIndex);
			FSharpOption<int> rowEnd = FSharpOption<int>.Some(rowIndex + rowLength - 1);
			FSharpOption<int> columnStart = FSharpOption<int>.Some(columnIndex);
			FSharpOption<int> columnEnd = FSharpOption<int>.Some(columnIndex + columnLength - 1);

			return new ADNDFloat32Array(internalArray._adArrayHandle.GetSlice(rowStart, rowEnd, columnStart, columnEnd));
		}

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

		public INDArray Divide<TOther>(TOther value, INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADNDFloat32Array(internalValue / internalArray._adArrayHandle);
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

		public int MaxIndex(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return DNDArray.MaxIndex(internalArray._adArrayHandle);
		}

		public INumber Min(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADFloat32Number(new DNumber(internalArray.Data.GetValue(DNDArray.MinIndex(internalArray._adArrayHandle))));
		}

		public int MinIndex(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return DNDArray.MinIndex(internalArray._adArrayHandle);
		}

		public INDArray SquareRoot(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.Sqrt(internalArray._adArrayHandle));
		}

		public INumber SquareRoot(INumber number)
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

		public INDArray SoftMax(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);

			return new ADNDFloat32Array(DNDArray.SoftMax(internalArray._adArrayHandle));
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

		public INDArray Clip(INDArray array, INumber minValue, INumber maxValue)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			ADFloat32Number internalMinValue = InternaliseNumber(minValue);
			ADFloat32Number internalMaxValue = InternaliseNumber(maxValue);

			DNDArray lowerClipped = DNDArray.Max(internalMinValue._adNumberHandle, internalArray._adArrayHandle);
			DNDArray clipped = DNDArray.Min(internalMaxValue._adNumberHandle, lowerClipped);

			return new ADNDFloat32Array(clipped);
		}

		public void FillWithProbabilityMask(INDArray array, double probability)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			int begin = (int) internalArray.Data.Offset, end = (int) internalArray.Data.Length;

			for (int i = begin; i < end; i++)
			{
				if (_probabilityMaskRng.NextDouble() < probability)
				{
					data[i] = 1;
				}
				else
				{
					data[i] = 0;
				}
			}
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

		public TTraceable ClearTrace<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable) ((object) new ADFloat32Number(internalNumber._adNumberHandle.P));
			}
			else if (traceable is ADNDFloat32Array)
			{
				ADNDFloat32Array internalArray = traceable as ADNDFloat32Array;

				return (TTraceable) ((object) new ADNDFloat32Array(internalArray._adArrayHandle.P));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
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

		public bool IsNaN(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			int begin = (int) internalArray.Data.Offset, end = (int) internalArray.Data.Length;

			for (int i = begin; i < end; i++)
			{
				if (float.IsNaN(data[i]))
				{
					return true;
				}
			}

			return false;
		}

		public bool IsNotFinite(INDArray array)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			int begin = (int) internalArray.Data.Offset, end = (int) internalArray.Data.Length;

			for (int i = begin; i < end; i++)
			{
				if (float.IsInfinity(data[i]))
				{
					return true;
				}
			}

			return false;
		}

		public bool IsNaN(INumber number)
		{
			ADFloat32Number internalNumber = InternaliseNumber(number);

			return float.IsNaN(internalNumber._adNumberHandle.Value);
		}

		public bool IsNotFinite(INumber number)
		{
			ADFloat32Number internalNumber = InternaliseNumber(number);

			return float.IsInfinity(internalNumber._adNumberHandle.Value);
		}

		static DiffSharpFloat32Handler()
		{
			PlatformDependentUtils.CheckPlatformDependentLibraries();
		}
	}
}
