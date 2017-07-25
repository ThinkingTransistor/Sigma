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
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Persistence;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// An abstract DiffSharp computation handle for 32-bit floats with dynamic backends (e.g. BLAS / LAPACK).
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

		protected DiffSharpFloat32Handler(DiffSharpBackendHandle<float> backendHandle)
		{
			if (backendHandle == null) throw new ArgumentNullException(nameof(backendHandle));

			InitialiseBackend(backendHandle);
		}

		protected DiffSharpFloat32Handler(IBlasBackend blasBackend, ILapackBackend lapackBackend)
		{
			if (blasBackend == null) throw new ArgumentNullException(nameof(blasBackend));
			if (lapackBackend == null) throw new ArgumentNullException(nameof(lapackBackend));

			BlasBackend = blasBackend;
			LapackBackend = lapackBackend;

			Registry = new Registry(tags: "handler");

			InitialiseBackend(new DiffSharpFloat32BackendHandle(blasBackend, lapackBackend, backendTag: -1));

			_probabilityMaskRng = new Random();
		}

		protected void InitialiseBackend(DiffSharpBackendHandle<float> backendHandle)
		{
			DiffsharpBackendHandle = backendHandle;
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
		public abstract void OnDeserialised();

		protected BackendConfig<float> CreateBackendConfig()
		{
			float epsilon = 0.00001f;
			float fpeps = 0.01f;

			return new BackendConfig<float>(DiffsharpBackendHandle, epsilon, 1.0f / epsilon, 0.5f / epsilon, fpeps, 100, 1.2f);
		}

		protected ADFloat32NDArray InternaliseArray(object array)
		{
			return AssignTag((ADFloat32NDArray)array);
		}

		protected ADFloat32Number InternaliseNumber(object number)
		{
			return (ADFloat32Number)number;
		}

		protected T AssignTag<T>(T array) where T : ADNDArray<float>
		{
			((SigmaDiffDataBuffer<float>)array.Data).BackendTag = _backendTag;

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

		protected ADFloat32NDArray ConvertInternal(INDArray array)
		{
			return new ADFloat32NDArray(_backendTag, array.GetDataAs<float>(), array.Shape);
		}

		/// <inheritdoc />
		public INDArray FlattenFeatures(INDArray array)
		{
			return array.Reshape(array.Shape[0], array.Shape[1], ArrayUtils.Product(2, array.Shape));
		}

		/// <inheritdoc />
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

		/// <inheritdoc />
		public INDArray FlattenTimeAndFeatures(INDArray array)
		{
			return array.Reshape(array.Shape[0] * array.Shape[1], ArrayUtils.Product(2, array.Shape));
		}

		/// <inheritdoc />
		public INDArray FlattenAllButLast(INDArray array)
		{
			return array.Reshape(ArrayUtils.Product(0, array.Rank - 1, array.Shape), array.Shape[array.Rank - 1]);
		}

		/// <inheritdoc />
		public INDArray PermuteBatchAndTime(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			// swap batch and time dimensions
			int[] permutedDimensions = ArrayUtils.Range(0, array.Rank - 1);
			permutedDimensions[1] = 0;
			permutedDimensions[0] = 1;

			return new ADFloat32NDArray(DNDArray.Permute(internalArray._adArrayHandle, permutedDimensions));
		}

		/// <inheritdoc />
		public TOther[] RowWiseTransform<TOther>(INDArray array, Func<INDArray, TOther> transformFunction)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			INDArray[] rows = SliceRowWise(array, internalArray);
			TOther[] transformedRows = new TOther[rows.Length];

			for (int i = 0; i < rows.Length; i++)
			{
				transformedRows[i] = transformFunction.Invoke(rows[i]);
			}

			return transformedRows;
		}

		/// <inheritdoc />
		public INDArray RowWise(INDArray array, Func<INDArray, INDArray> function)
		{
			// no need to slice if there's only one row
			if (array.Shape[0] == 1)
			{
				return function.Invoke(array);
			}

			ADFloat32NDArray internalArray = InternaliseArray(array);
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

			return new ADFloat32NDArray(new DNDArray(DNDArray.OfRows(internalRowHandles, DiffsharpBackendHandle)));
		}

		private static INDArray[] SliceRowWise(INDArray array, ADFloat32NDArray internalArray)
		{
			// no need to slice if there's only one row
			if (array.Shape[0] == 1)
			{
				return new[] { array };
			}

			INDArray[] rows = new INDArray[array.Shape[0]];

			var colStart = FSharpOption<int>.Some(0);
			var colFinish = FSharpOption<int>.Some(checked((int)array.Shape[1] - 1));

			for (var i = 0; i < rows.Length; i++)
			{
				var row = FSharpOption<int>.Some(i);

				rows[i] = new ADFloat32NDArray(internalArray._adArrayHandle.GetSlice(row, row, colStart, colFinish));
			}

			return rows;
		}

		/// <inheritdoc />
		public INDArray GetSlice(INDArray array, int rowIndex, int columnIndex, int rowLength, int columnLength)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			FSharpOption<int> rowStart = FSharpOption<int>.Some(rowIndex);
			FSharpOption<int> rowEnd = FSharpOption<int>.Some(rowIndex + rowLength - 1);
			FSharpOption<int> columnStart = FSharpOption<int>.Some(columnIndex);
			FSharpOption<int> columnEnd = FSharpOption<int>.Some(columnIndex + columnLength - 1);

			return new ADFloat32NDArray(internalArray._adArrayHandle.GetSlice(rowStart, rowEnd, columnStart, columnEnd));
		}

		/// <inheritdoc />
		public INDArray StackRows(int numberRows, INDArray row)
		{
			ADFloat32NDArray internalArray = InternaliseArray(row);

			return new ADFloat32NDArray(DNDArray.OfRows(numberRows, internalArray._adArrayHandle, _diffsharpBackendHandle));
		}

		/// <inheritdoc />
		public INDArray Add<TOther>(INDArray array, TOther value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = (ADFloat32Number)Number((float)System.Convert.ChangeType(value, typeof(float)));

			return new ADFloat32NDArray(internalArray._adArrayHandle + internalValue._adNumberHandle);
		}

		/// <inheritdoc />
		public INDArray Add(INDArray array, INumber value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(internalArray._adArrayHandle + internalValue._adNumberHandle);
		}

		/// <inheritdoc />
		public INDArray Add(INDArray a, INDArray b)
		{
			ADFloat32NDArray internalA = InternaliseArray(a);
			ADFloat32NDArray internalB = InternaliseArray(b);

			return new ADFloat32NDArray(internalA._adArrayHandle + internalB._adArrayHandle);
		}

		/// <inheritdoc />
		public INumber Add(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle + internalB._adNumberHandle);
		}

		/// <inheritdoc />
		public INumber Add<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle + internalB);
		}

		/// <inheritdoc />
		public INDArray Subtract<TOther>(INDArray array, TOther value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(internalArray._adArrayHandle - internalValue);
		}

		/// <inheritdoc />
		public INDArray Subtract(INDArray array, INumber value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(internalArray._adArrayHandle - internalValue._adNumberHandle);
		}

		/// <inheritdoc />
		public INDArray Subtract(INDArray a, INDArray b)
		{
			ADFloat32NDArray internalA = InternaliseArray(a);
			ADFloat32NDArray internalB = InternaliseArray(b);

			return new ADFloat32NDArray(internalA._adArrayHandle - internalB._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray Subtract<TOther>(TOther value, INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(internalValue - internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray Subtract(INumber value, INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(internalValue._adNumberHandle - internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INumber Subtract(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle - internalB._adNumberHandle);
		}

		/// <inheritdoc />
		public INumber Subtract<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle - internalB);
		}

		/// <inheritdoc />
		public INumber Subtract<TOther>(TOther a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalB - internalA._adNumberHandle);
		}

		/// <inheritdoc />
		public INDArray Multiply<TOther>(INDArray array, TOther value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(internalArray._adArrayHandle * internalValue);
		}

		/// <inheritdoc />
		public INDArray Multiply(INDArray array, INumber value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(internalValue._adNumberHandle * internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray Multiply(INDArray a, INDArray b)
		{
			ADFloat32NDArray internalA = InternaliseArray(a);
			ADFloat32NDArray internalB = InternaliseArray(b);

			return new ADFloat32NDArray(DNDArray.op_DotMultiply(internalA._adArrayHandle, internalB._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Multiply(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle * internalB._adNumberHandle);
		}

		/// <inheritdoc />
		public INumber Multiply<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle * internalB);
		}

		/// <inheritdoc />
		public INDArray Dot(INDArray a, INDArray b)
		{
			ADFloat32NDArray internalA = InternaliseArray(a);
			ADFloat32NDArray internalB = InternaliseArray(b);

			return new ADFloat32NDArray(internalA._adArrayHandle * internalB._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray Divide<TOther>(TOther value, INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(internalValue / internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray Divide<TOther>(INDArray array, TOther value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(internalArray._adArrayHandle / internalValue);
		}

		/// <inheritdoc />
		public INDArray Divide(INDArray array, INumber value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(internalArray._adArrayHandle / internalValue._adNumberHandle);
		}

		/// <inheritdoc />
		public INDArray Divide(INDArray a, INDArray b)
		{
			ADFloat32NDArray internalA = InternaliseArray(a);
			ADFloat32NDArray internalB = InternaliseArray(b);

			return new ADFloat32NDArray(DNDArray.op_DotDivide(internalA._adArrayHandle, internalB._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Divide(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(internalA._adNumberHandle / internalB._adNumberHandle);
		}

		/// <inheritdoc />
		public INumber Divide<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(internalA._adNumberHandle / internalB);
		}

		/// <inheritdoc />
		public INDArray Pow(INDArray array, INumber value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalValue = InternaliseNumber(value);

			return new ADFloat32NDArray(DNDArray.Pow(internalArray._adArrayHandle, internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Pow<TOther>(INDArray array, TOther value)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return new ADFloat32NDArray(DNDArray.Pow(internalArray._adArrayHandle, internalValue));
		}

		/// <inheritdoc />
		public INumber Pow(INumber a, INumber b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			ADFloat32Number internalB = InternaliseNumber(b);

			return new ADFloat32Number(DNumber.Pow(internalA._adNumberHandle, internalB._adNumberHandle));
		}

		/// <inheritdoc />
		public INumber Pow<TOther>(INumber a, TOther b)
		{
			ADFloat32Number internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return new ADFloat32Number(DNumber.Pow(internalA._adNumberHandle, internalB));
		}

		/// <inheritdoc />
		public INumber Abs(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Abs(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Abs(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Abs(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Sum(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Sum(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Max(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(new DNumber(internalArray.Data.GetValue(DNDArray.MaxIndex(internalArray._adArrayHandle))));
		}

		/// <inheritdoc />
		public int MaxIndex(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return DNDArray.MaxIndex(internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INumber Min(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(new DNumber(internalArray.Data.GetValue(DNDArray.MinIndex(internalArray._adArrayHandle))));
		}

		/// <inheritdoc />
		public int MinIndex(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return DNDArray.MinIndex(internalArray._adArrayHandle);
		}

		/// <inheritdoc />
		public INDArray SquareRoot(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Sqrt(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber SquareRoot(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sqrt(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Log(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Log(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Log(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Log(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INumber Determinate(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Det(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INDArray Sin(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Sin(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Sin(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sin(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Asin(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Asin(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Asin(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Asin(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Cos(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Cos(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Cos(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Cos(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Acos(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Acos(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Acos(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Acos(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Tan(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Tan(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Tan(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Tan(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Atan(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Atan(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Atan(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Atan(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray ReL(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.ReLU(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber ReL(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.ReLU(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray Sigmoid(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Sigmoid(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Sigmoid(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Sigmoid(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray SoftPlus(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.SoftPlus(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber SoftPlus(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.SoftPlus(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INDArray SoftMax(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.SoftMax(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INDArray Tanh(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32NDArray(DNDArray.Tanh(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Tanh(INumber number)
		{
			ADFloat32Number internalValue = InternaliseNumber(number);

			return new ADFloat32Number(DNumber.Tanh(internalValue._adNumberHandle));
		}

		/// <inheritdoc />
		public INumber Activation(string activation, INumber number)
		{
			return ActivationManager.ApplyActivation(activation, number, this);
		}

		/// <inheritdoc />
		public INDArray Activation(string activation, INDArray array)
		{
			return ActivationManager.ApplyActivation(activation, array, this);
		}

		/// <inheritdoc />
		public INumber StandardDeviation(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.StandardDev(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INumber Variance(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			return new ADFloat32Number(DNDArray.Variance(internalArray._adArrayHandle));
		}

		/// <inheritdoc />
		public INDArray Clip(INDArray array, INumber minValue, INumber maxValue)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			ADFloat32Number internalMinValue = InternaliseNumber(minValue);
			ADFloat32Number internalMaxValue = InternaliseNumber(maxValue);

			DNDArray lowerClipped = DNDArray.Max(internalMinValue._adNumberHandle, internalArray._adArrayHandle);
			DNDArray clipped = DNDArray.Min(internalMaxValue._adNumberHandle, lowerClipped);

			return new ADFloat32NDArray(clipped);
		}

		private uint _x = 123456789, _y = 362436069, _z = 521288629, _w = 88675123;

		/// <inheritdoc />
		public unsafe void FillWithProbabilityMask(INDArray array, double probability)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			float probabilityAsFloat = (float)probability;
			ushort approximateProbability = (ushort)Math.Round(probabilityAsFloat * ushort.MaxValue);

			int begin = (int)internalArray.Data.Offset, end = (int)(begin + internalArray.Data.Length);
			uint x = _x, y = _y, z = _z, w = _w;

			// credit to Marsaglia for this Xorshift fast RNG implementation (which this is based on)
			int i = begin;
			while (i < end - 8)
			{
				uint tx = x ^ (x << 11);
				uint ty = y ^ (y << 11);
				uint tz = z ^ (z << 11);
				uint tw = w ^ (w << 11);

				x = w ^ (w >> 19) ^ (tx ^ (tx >> 8));
				data[i++] = *(ushort*)&x < approximateProbability ? 1 : 0;
				data[i++] = ((ushort*)&x)[1] < approximateProbability ? 1 : 0;
				y = x ^ (x >> 19) ^ (ty ^ (ty >> 8));
				data[i++] = *(ushort*)&y < approximateProbability ? 1 : 0;
				data[i++] = ((ushort*)&y)[1] < approximateProbability ? 1 : 0;
				z = y ^ (y >> 19) ^ (tz ^ (tz >> 8));
				data[i++] = *(ushort*)&z < approximateProbability ? 1 : 0;
				data[i++] = ((ushort*)&z)[1] < approximateProbability ? 1 : 0;
				w = z ^ (z >> 19) ^ (tw ^ (tw >> 8));
				data[i++] = *(ushort*)&w < approximateProbability ? 1 : 0;
				data[i++] = ((ushort*)&w)[1] < approximateProbability ? 1 : 0;
			}

			for (; i < end; i++)
			{
				uint t = x ^ (x << 11);
				x = y; y = z; z = w;
				uint randomInt = w = w ^ (w >> 19) ^ (t ^ (t >> 8));

				data[i] = *(float*)&randomInt < probabilityAsFloat ? 1 : 0;
			}

			_x = x;
			_y = y;
			_z = z;
			_w = w;
		}

		/// <inheritdoc />
		public uint BeginTrace()
		{
			return Util.GlobalTagger.Next;
		}

		/// <inheritdoc />
		public TTraceable Trace<TTraceable>(TTraceable traceable, uint traceTag) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable)((object)new ADFloat32Number(internalNumber._adNumberHandle.GetReverse(traceTag)));
			}
			else if (traceable is ADFloat32NDArray)
			{
				ADFloat32NDArray internalArray = traceable as ADFloat32NDArray;

				return (TTraceable)((object)new ADFloat32NDArray(internalArray._adArrayHandle.GetReverse(traceTag)));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public TTraceable ClearTrace<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable)((object)new ADFloat32Number(internalNumber._adNumberHandle.P));
			}
			else if (traceable is ADFloat32NDArray)
			{
				ADFloat32NDArray internalArray = traceable as ADFloat32NDArray;

				return (TTraceable)((object)new ADFloat32NDArray(internalArray._adArrayHandle.P));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public void ComputeDerivativesTo(ITraceable traceable)
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number number = (ADFloat32Number)traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, number._adNumberHandle.asADD);
			}
			else if (traceable is ADFloat32NDArray)
			{
				ADFloat32NDArray array = (ADFloat32NDArray)traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, array._adArrayHandle.asADDND);
			}
			else
			{
				throw new InvalidOperationException($"Cannot compute derivatives for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public TTraceable GetDerivative<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is ADFloat32Number)
			{
				ADFloat32Number internalNumber = traceable as ADFloat32Number;

				return (TTraceable)((object)new ADFloat32Number(internalNumber._adNumberHandle.A));
			}
			else if (traceable is ADFloat32NDArray)
			{
				ADFloat32NDArray internalArray = traceable as ADFloat32NDArray;

				return (TTraceable)((object)new ADFloat32NDArray(internalArray._adArrayHandle.A));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public void BeginSession()
		{
			_diffsharpBackendHandle.BufferSessions = true;
			_diffsharpBackendHandle.TransferSessionBuffers();
		}

		/// <inheritdoc />
		public void EndSession()
		{
			_diffsharpBackendHandle.BufferSessions = false;
		}

		/// <inheritdoc />
		public void ClearSession()
		{
			_diffsharpBackendHandle.ClearSessionBuffers();
		}

		/// <inheritdoc />
		public void MarkLimbo(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			_diffsharpBackendHandle.MarkLimbo(internalArray.Data.Data);
		}

		/// <inheritdoc />
		public void FreeLimbo(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);

			_diffsharpBackendHandle.FreeLimbo(internalArray.Data.Data);
		}

		/// <inheritdoc />
		public INDArray MergeBatch(params INDArray[] arrays)
		{
			ADFloat32NDArray[] castArrays = arrays.As<INDArray, ADFloat32NDArray>();

			long[] totalShape = new long[castArrays[0].Rank];

			Array.Copy(arrays[0].Shape, 1, totalShape, 1, totalShape.Length - 1);

			foreach (ADFloat32NDArray array in castArrays)
			{
				totalShape[0] += array.Shape[0];
			}

			ADFloat32NDArray merged = new ADFloat32NDArray(_backendTag, totalShape);
			DataBuffer<float> mergedData = (DataBuffer<float>)merged.Data;

			long lastIndex = 0L;
			foreach (ADFloat32NDArray array in castArrays)
			{
				DataBuffer<float> arrayData = (DataBuffer<float>)array.Data;

				mergedData.SetValues(arrayData, 0, lastIndex, arrayData.Length);

				lastIndex += arrayData.Length;
			}

			return merged;
		}

		/// <inheritdoc />
		public bool IsNaN(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			int begin = (int)internalArray.Data.Offset, end = (int)internalArray.Data.Length;

			for (int i = begin; i < end; i++)
			{
				if (float.IsNaN(data[i]))
				{
					return true;
				}
			}

			return false;
		}

		/// <inheritdoc />
		public bool IsNotFinite(INDArray array)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			float[] data = internalArray.Data.Data;
			int begin = (int)internalArray.Data.Offset, end = (int)internalArray.Data.Length;

			for (int i = begin; i < end; i++)
			{
				if (float.IsInfinity(data[i]))
				{
					return true;
				}
			}

			return false;
		}

		/// <inheritdoc />
		public bool IsNaN(INumber number)
		{
			ADFloat32Number internalNumber = InternaliseNumber(number);

			return float.IsNaN(internalNumber._adNumberHandle.Value);
		}

		/// <inheritdoc />
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
