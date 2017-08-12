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
	public abstract class DiffSharpFloat32Handler<TNDArray, TNumber> : IComputationHandler, ISerialisationNotifier 
		where TNDArray : ADNDArray<float>, IADFloat32NDArrayHandle where TNumber : ADNumber<float>, IADFloat32NumberHandle
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

		/// <summary>
		/// Get an internal (type-cast) version of an <see cref="INDArray"/>.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The internal ndarray.</returns>
		protected TNDArray InternaliseArray(object array)
		{
			return AssignTag((TNDArray)array);
		}

		/// <summary>
		/// Get an internal (type-cast) version of an <see cref="INumber"/>.
		/// </summary>
		/// <param name="number">The number.</param>
		/// <returns>The internal number.</returns>
		protected TNumber InternaliseNumber(object number)
		{
			return (TNumber)number;
		}

		/// <summary>
		/// Assign the tag of this backend handle to an array.
		/// </summary>
		/// <typeparam name="T">The array type.</typeparam>
		/// <param name="array">The array.</param>
		/// <returns>The array (for convenience).</returns>
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
		public abstract bool IsOwnFormat(INDArray array);
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

		/// <summary>
		/// Create an ndarray from a SigmaDiff float32 ndarray handle.
		/// </summary>
		/// <param name="handle">The handle.</param>
		/// <returns>The ndarray.</returns>
		protected abstract TNDArray CreateArrayFromHandle(DNDArray handle);

		/// <summary>
		/// Create a number from a SigmaDiff float32 number handle.
		/// </summary>
		/// <param name="handle">The handle.</param>
		/// <returns>The number.</returns>
		protected abstract TNumber CreateNumberFromHandle(DNumber handle);

		/// <summary>
		/// Convert an array of another type to an internal array or return the given array if it is already of the right type.
		/// </summary>
		/// <param name="array">The array to convert.</param>
		/// <returns>A converted version of the given array (copy if wrong type).</returns>
		protected abstract TNDArray ConvertInternal(INDArray array);

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
			TNDArray internalArray = InternaliseArray(array);

			// swap batch and time dimensions
			int[] permutedDimensions = ArrayUtils.Range(0, array.Rank - 1);
			permutedDimensions[1] = 0;
			permutedDimensions[0] = 1;

			return CreateArrayFromHandle(DNDArray.Permute(internalArray.Handle, permutedDimensions));
		}

		/// <inheritdoc />
		public TOther[] RowWiseTransform<TOther>(INDArray array, Func<INDArray, TOther> transformFunction)
		{
			TNDArray internalArray = InternaliseArray(array);
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

			TNDArray internalArray = InternaliseArray(array);

			if (_RowWiseOptimised(ref internalArray, function))
			{
				return internalArray;
			}

			INDArray[] rows = SliceRowWise(array, internalArray);

			for (int i = 0; i < rows.Length; i++)
			{
				rows[i] = function.Invoke(rows[i]);
			}

			DNDArray[] internalRowHandles = new DNDArray[rows.Length];
			for (int i = 0; i < rows.Length; i++)
			{
				internalRowHandles[i] = InternaliseArray(rows[i]).Handle;
			}

			return CreateArrayFromHandle(new DNDArray(DNDArray.OfRows(internalRowHandles, DiffsharpBackendHandle)));
		}

		/// <summary>
		/// Invoke an optimised version of a <see cref="RowWise"/> application for a specific function if applicable, return false if not supported (for that function or in general).
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="function">The function to apply row-wise.</param>
		/// <returns>A boolean indicating whether an optimised version of the function was available and used (if not, standard row-wise handling is applied).</returns>
		protected virtual bool _RowWiseOptimised(ref TNDArray array, Func<INDArray, INDArray> function)
		{
			return false;
		}

		private INDArray[] SliceRowWise(INDArray array, TNDArray internalArray)
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

				rows[i] = CreateArrayFromHandle(internalArray.Handle.GetSlice(row, row, colStart, colFinish));
			}

			return rows;
		}

		/// <inheritdoc />
		public INDArray GetSlice(INDArray array, int rowIndex, int columnIndex, int rowLength, int columnLength)
		{
			TNDArray internalArray = InternaliseArray(array);
			FSharpOption<int> rowStart = FSharpOption<int>.Some(rowIndex);
			FSharpOption<int> rowEnd = FSharpOption<int>.Some(rowIndex + rowLength - 1);
			FSharpOption<int> columnStart = FSharpOption<int>.Some(columnIndex);
			FSharpOption<int> columnEnd = FSharpOption<int>.Some(columnIndex + columnLength - 1);

			return CreateArrayFromHandle(internalArray.Handle.GetSlice(rowStart, rowEnd, columnStart, columnEnd));
		}

		/// <inheritdoc />
		public INDArray StackRows(int numberRows, INDArray row)
		{
			TNDArray internalArray = InternaliseArray(row);

			return CreateArrayFromHandle(DNDArray.OfRows(numberRows, internalArray.Handle, _diffsharpBackendHandle));
		}

		/// <inheritdoc />
		public INDArray Add<TOther>(INDArray array, TOther value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = (TNumber)Number((float)System.Convert.ChangeType(value, typeof(float)));

			return CreateArrayFromHandle(internalArray.Handle + internalValue.Handle);
		}

		/// <inheritdoc />
		public INDArray Add(INDArray array, INumber value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(internalArray.Handle + internalValue.Handle);
		}

		/// <inheritdoc />
		public INDArray Add(INDArray a, INDArray b)
		{
			TNDArray internalA = InternaliseArray(a);
			TNDArray internalB = InternaliseArray(b);

			return CreateArrayFromHandle(internalA.Handle + internalB.Handle);
		}

		/// <inheritdoc />
		public INumber Add(INumber a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			TNumber internalB = InternaliseNumber(b);

			return CreateNumberFromHandle(internalA.Handle + internalB.Handle);
		}

		/// <inheritdoc />
		public INumber Add<TOther>(INumber a, TOther b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(internalA.Handle + internalB);
		}

		/// <inheritdoc />
		public INDArray Subtract<TOther>(INDArray array, TOther value)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(internalArray.Handle - internalValue);
		}

		/// <inheritdoc />
		public INDArray Subtract(INDArray array, INumber value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(internalArray.Handle - internalValue.Handle);
		}

		/// <inheritdoc />
		public INDArray Subtract(INDArray a, INDArray b)
		{
			TNDArray internalA = InternaliseArray(a);
			TNDArray internalB = InternaliseArray(b);

			return CreateArrayFromHandle(internalA.Handle - internalB.Handle);
		}

		/// <inheritdoc />
		public INDArray Subtract<TOther>(TOther value, INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(internalValue - internalArray.Handle);
		}

		/// <inheritdoc />
		public INDArray Subtract(INumber value, INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(internalValue.Handle - internalArray.Handle);
		}

		/// <inheritdoc />
		public INumber Subtract(INumber a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			TNumber internalB = InternaliseNumber(b);

			return CreateNumberFromHandle(internalA.Handle - internalB.Handle);
		}

		/// <inheritdoc />
		public INumber Subtract<TOther>(INumber a, TOther b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(internalA.Handle - internalB);
		}

		/// <inheritdoc />
		public INumber Subtract<TOther>(TOther a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(internalB - internalA.Handle);
		}

		/// <inheritdoc />
		public INDArray Multiply<TOther>(INDArray array, TOther value)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(internalArray.Handle * internalValue);
		}

		/// <inheritdoc />
		public INDArray Multiply(INDArray array, INumber value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(internalValue.Handle * internalArray.Handle);
		}

		/// <inheritdoc />
		public INDArray Multiply(INDArray a, INDArray b)
		{
			TNDArray internalA = InternaliseArray(a);
			TNDArray internalB = InternaliseArray(b);

			return CreateArrayFromHandle(DNDArray.op_DotMultiply(internalA.Handle, internalB.Handle));
		}

		/// <inheritdoc />
		public INumber Multiply(INumber a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			TNumber internalB = InternaliseNumber(b);

			return CreateNumberFromHandle(internalA.Handle * internalB.Handle);
		}

		/// <inheritdoc />
		public INumber Multiply<TOther>(INumber a, TOther b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(internalA.Handle * internalB);
		}

		/// <inheritdoc />
		public INDArray Dot(INDArray a, INDArray b)
		{
			TNDArray internalA = InternaliseArray(a);
			TNDArray internalB = InternaliseArray(b);

			return CreateArrayFromHandle(internalA.Handle * internalB.Handle);
		}

		/// <inheritdoc />
		public INDArray Divide<TOther>(TOther value, INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(internalValue / internalArray.Handle);
		}

		/// <inheritdoc />
		public INDArray Divide<TOther>(INDArray array, TOther value)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(internalArray.Handle / internalValue);
		}

		/// <inheritdoc />
		public INDArray Divide(INDArray array, INumber value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(internalArray.Handle / internalValue.Handle);
		}

		/// <inheritdoc />
		public INDArray Divide(INDArray a, INDArray b)
		{
			TNDArray internalA = InternaliseArray(a);
			TNDArray internalB = InternaliseArray(b);

			return CreateArrayFromHandle(DNDArray.op_DotDivide(internalA.Handle, internalB.Handle));
		}

		/// <inheritdoc />
		public INumber Divide(INumber a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			TNumber internalB = InternaliseNumber(b);

			return CreateNumberFromHandle(internalA.Handle / internalB.Handle);
		}

		/// <inheritdoc />
		public INumber Divide<TOther>(INumber a, TOther b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(internalA.Handle / internalB);
		}

		/// <inheritdoc />
		public INDArray Pow(INDArray array, INumber value)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalValue = InternaliseNumber(value);

			return CreateArrayFromHandle(DNDArray.Pow(internalArray.Handle, internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Pow<TOther>(INDArray array, TOther value)
		{
			TNDArray internalArray = InternaliseArray(array);
			float internalValue = (float)System.Convert.ChangeType(value, typeof(float));

			return CreateArrayFromHandle(DNDArray.Pow(internalArray.Handle, internalValue));
		}

		/// <inheritdoc />
		public INumber Pow(INumber a, INumber b)
		{
			TNumber internalA = InternaliseNumber(a);
			TNumber internalB = InternaliseNumber(b);

			return CreateNumberFromHandle(DNumber.Pow(internalA.Handle, internalB.Handle));
		}

		/// <inheritdoc />
		public INumber Pow<TOther>(INumber a, TOther b)
		{
			TNumber internalA = InternaliseNumber(a);
			float internalB = (float)System.Convert.ChangeType(b, typeof(float));

			return CreateNumberFromHandle(DNumber.Pow(internalA.Handle, internalB));
		}

		/// <inheritdoc />
		public INumber Abs(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Abs(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Abs(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Abs(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Sum(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(DNDArray.Sum(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Max(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(new DNumber(internalArray.Data.GetValue(DNDArray.MaxIndex(internalArray.Handle))));
		}

		/// <inheritdoc />
		public int MaxIndex(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return DNDArray.MaxIndex(internalArray.Handle);
		}

		/// <inheritdoc />
		public INumber Min(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(new DNumber(internalArray.Data.GetValue(DNDArray.MinIndex(internalArray.Handle))));
		}

		/// <inheritdoc />
		public int MinIndex(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return DNDArray.MinIndex(internalArray.Handle);
		}

		/// <inheritdoc />
		public INDArray SquareRoot(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Sqrt(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber SquareRoot(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Sqrt(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Log(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Log(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Log(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Log(internalValue.Handle));
		}

		/// <inheritdoc />
		public INumber Determinate(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(DNDArray.Det(internalArray.Handle));
		}

		/// <inheritdoc />
		public INDArray Sin(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Sin(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Sin(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Sin(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Asin(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Asin(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Asin(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Asin(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Cos(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Cos(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Cos(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Cos(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Acos(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Acos(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Acos(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Acos(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Tan(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Tan(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Tan(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Tan(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Atan(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Atan(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Atan(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Atan(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray ReL(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.ReLU(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber ReL(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.ReLU(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray Sigmoid(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Sigmoid(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Sigmoid(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Sigmoid(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray SoftPlus(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.SoftPlus(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber SoftPlus(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.SoftPlus(internalValue.Handle));
		}

		/// <inheritdoc />
		public INDArray SoftMax(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.SoftMax(internalArray.Handle));
		}

		/// <inheritdoc />
		public INDArray Tanh(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateArrayFromHandle(DNDArray.Tanh(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Tanh(INumber number)
		{
			TNumber internalValue = InternaliseNumber(number);

			return CreateNumberFromHandle(DNumber.Tanh(internalValue.Handle));
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
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(DNDArray.StandardDev(internalArray.Handle));
		}

		/// <inheritdoc />
		public INumber Variance(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			return CreateNumberFromHandle(DNDArray.Variance(internalArray.Handle));
		}

		/// <inheritdoc />
		public INDArray Clip(INDArray array, INumber minValue, INumber maxValue)
		{
			TNDArray internalArray = InternaliseArray(array);
			TNumber internalMinValue = InternaliseNumber(minValue);
			TNumber internalMaxValue = InternaliseNumber(maxValue);

			DNDArray lowerClipped = DNDArray.Max(internalMinValue.Handle, internalArray.Handle);
			DNDArray clipped = DNDArray.Min(internalMaxValue.Handle, lowerClipped);

			return CreateArrayFromHandle(clipped);
		}

		private uint _x = 123456789, _y = 362436069, _z = 521288629, _w = 88675123;

		/// <inheritdoc />
		public virtual unsafe void FillWithProbabilityMask(INDArray array, double probability)
		{
			TNDArray internalArray = InternaliseArray(array);
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
			if (traceable is TNumber)
			{
				TNumber internalNumber = traceable as TNumber;

				return (TTraceable)((object)CreateNumberFromHandle(internalNumber.Handle.GetReverse(traceTag)));
			}
			else if (traceable is TNDArray)
			{
				TNDArray internalArray = traceable as TNDArray;

				return (TTraceable)((object)CreateArrayFromHandle(internalArray.Handle.GetReverse(traceTag)));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public TTraceable ClearTrace<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is TNumber)
			{
				TNumber internalNumber = traceable as TNumber;

				return (TTraceable)((object)CreateNumberFromHandle(internalNumber.Handle.P));
			}
			else if (traceable is TNDArray)
			{
				TNDArray internalArray = traceable as TNDArray;

				return (TTraceable)((object)CreateArrayFromHandle(internalArray.Handle.P));
			}
			else
			{
				throw new InvalidOperationException($"Cannot get derivative for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public void ComputeDerivativesTo(ITraceable traceable)
		{
			if (traceable is TNumber)
			{
				TNumber number = (TNumber)traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, number.Handle.asADD);
			}
			else if (traceable is TNDArray)
			{
				TNDArray array = (TNDArray)traceable;

				AD.ReverseProp(new DNumber(1.0f).asADD, array.Handle.asADDND);
			}
			else
			{
				throw new InvalidOperationException($"Cannot compute derivatives for traceable of unknown type (type of object {traceable} not compatible with this handler).");
			}
		}

		/// <inheritdoc />
		public TTraceable GetDerivative<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			if (traceable is TNumber)
			{
				TNumber internalNumber = traceable as TNumber;

				return (TTraceable)((object)CreateNumberFromHandle(internalNumber.Handle.A));
			}
			else if (traceable is TNDArray)
			{
				TNDArray internalArray = traceable as TNDArray;

				return (TTraceable)((object)CreateArrayFromHandle(internalArray.Handle.A));
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
		public virtual void MarkLimbo(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			_diffsharpBackendHandle.MarkLimbo(internalArray.Data.Data);
		}

		/// <inheritdoc />
		public virtual void FreeLimbo(INDArray array)
		{
			TNDArray internalArray = InternaliseArray(array);

			_diffsharpBackendHandle.FreeLimbo(internalArray.Data.Data);
		}

		/// <inheritdoc />
		public INDArray MergeBatch(params INDArray[] arrays)
		{
			TNDArray[] castArrays = arrays.As<INDArray, TNDArray>();

			long[] totalShape = new long[castArrays[0].Rank];

			Array.Copy(arrays[0].Shape, 1, totalShape, 1, totalShape.Length - 1);

			foreach (TNDArray array in castArrays)
			{
				totalShape[0] += array.Shape[0];
			}

			TNDArray merged = InternaliseArray(NDArray(totalShape));
			DataBuffer<float> mergedData = (DataBuffer<float>)merged.Data;

			long lastIndex = 0L;
			foreach (TNDArray array in castArrays)
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
			TNDArray internalArray = InternaliseArray(array);
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
			TNDArray internalArray = InternaliseArray(array);
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
			TNumber internalNumber = InternaliseNumber(number);

			return float.IsNaN(internalNumber.Handle.Value);
		}

		/// <inheritdoc />
		public bool IsNotFinite(INumber number)
		{
			TNumber internalNumber = InternaliseNumber(number);

			return float.IsInfinity(internalNumber.Handle.Value);
		}

		static DiffSharpFloat32Handler()
		{
			PlatformDependentUtils.CheckPlatformDependentLibraries();
		}
	}

	/// <summary>
	/// A float32 ndarray with a SigmaDiff backend handle. 
	/// </summary>
	public interface IADFloat32NDArrayHandle
	{
		/// <summary>
		/// A SigmaDiff backend handle for a float32 ndarray.
		/// </summary>
		DNDArray Handle { get; }
	}

	/// <summary>
	/// A float32 number with a SigmaDiff backend handle. 
	/// </summary>
	public interface IADFloat32NumberHandle
	{
		/// <summary>
		/// A SigmaDiff backend handle for a float32 number.
		/// </summary>
		DNumber Handle { get; }
	}
}
