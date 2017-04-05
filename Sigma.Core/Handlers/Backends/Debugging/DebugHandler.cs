/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Handlers.Backends.Debugging
{
	/// <summary>
	/// A debug handler that checks for various invalid operations at runtime.
	/// </summary>
	[Serializable]
	public class DebugHandler : IComputationHandler
	{
		public IComputationHandler UnderlyingHandler { get; }

		public IDataType DataType => UnderlyingHandler.DataType;

		public IRegistry Registry { get; }

		/// <summary>
		/// A boolean indicating whether this debug handler is enabled (does any checks).
		/// </summary>
		public bool Enabled
		{
			get { return Registry.Get<bool>("enabled"); }
			set { Registry.Set("enabled", value, typeof(bool)); }
		}

		/// <summary>
		/// A boolean indicating whether this debug handler should throw an exception when a bad condition is reported (or just log it).
		/// </summary>
		public bool ThrowExceptionOnReport
		{
			get { return Registry.Get<bool>("throw_exeption_on_report"); }
			set { Registry.Set("throw_exeption_on_report", value, typeof(bool)); }
		}

		/// <summary>
		/// A boolean indicating whether this debug handler should check for NaN values.
		/// </summary>
		public bool CheckNaN
		{
			get { return Registry.Get<bool>("check_nan"); }
			set { Registry.Set("check_nan", value, typeof(bool)); }
		}

		/// <summary>
		/// A boolean indicating whether this debug handler should check for infinite values.
		/// </summary>
		public bool CheckInfinite
		{
			get { return Registry.Get<bool>("check_infinite"); }
			set { Registry.Set("check_infinite", value, typeof(bool)); }
		}

		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(typeof(DebugHandler));

		public DebugHandler(IComputationHandler underlyingHandler, bool throwExceptionOnReport = true, bool enabled = true)
		{
			if (underlyingHandler == null) throw new ArgumentNullException(nameof(underlyingHandler));

			UnderlyingHandler = underlyingHandler;
			Registry = new Registry(tags: "handler");

			// these need to be set once so they are set initially in the registry
			//  kind of ugly but saves me from writing more solid property handling
			ThrowExceptionOnReport = throwExceptionOnReport;
			Enabled = enabled;
			CheckNaN = enabled;
			CheckInfinite = enabled;
		}

		private void Report(string message, params object[] values)
		{
			_logger.Error(message);

			if (ThrowExceptionOnReport)
			{
				throw new DebugReportException(message, values);
			}
		}

		private INDArray CheckNice(INDArray array, string paramName = "unspecified")
		{
			if (!Enabled)
			{
				return array;
			}

			if (array == null)
			{
				Report($"ndarray {paramName} is null.");
			}
			else
			{
				if (array.Rank != array.Shape.Length)
				{
					Report($"ndarray {paramName} has inconsistent rank ({array.Rank}) / shape (length {array.Length}).", array);
				}

				if (CheckNaN && UnderlyingHandler.IsNaN(array))
				{
					Report($"ndarray {paramName} contains NaN values.", array);
				}

				if (CheckInfinite && UnderlyingHandler.IsNotFinite(array))
				{
					Report($"ndarray {paramName} contains infinite values.", array);
				}
			}

			return array;
		}

		private INumber CheckNice(INumber number, string paramName = "unspecified")
		{
			if (!Enabled)
			{
				return number;
			}

			if (number == null)
			{
				Report($"number {paramName} is null.");
			}
			else
			{
				if (CheckNaN && UnderlyingHandler.IsNaN(number))
				{
					Report($"number {paramName} is a NaN value.", number);
				}

				if (CheckInfinite && UnderlyingHandler.IsNotFinite(number))
				{
					Report($"number {paramName} is an infinite value.", number);
				}
			}

			return number;
		}

		private void CheckEqualLength(INDArray a, INDArray b, string operation)
		{
			if (!Enabled)
			{
				return;
			}

			if (a.Length != b.Length)
			{
				Report($"ndarrays a (length {a.Length}) and b (length {b.Length}) for operation {operation} should have equal length", a, b);
			}
		}

		private void CheckEqualShape(INDArray a, INDArray b, string operation)
		{
			if (!Enabled)
			{
				return;
			}

			if (a.Shape.Length != b.Shape.Length)
			{
				Report($"ndarray shapes of a (shape length {a.Shape.Length}) and b (shape length {b.Shape.Length}) for operation {operation} should be equal", a, b);
			}

			for (var i = 0; i < a.Shape.Length; i++)
			{
				if (a.Shape[i] != b.Shape[i])
				{
					Report($"ndarray shapes of a and b for operation {operation} should be equal but differ at index {i} (a.Shape[{i}] != b.Shape[{i}])", a, b);
				}
			}
		}

		private void CheckMatrix(INDArray array, string operation)
		{
			if (!Enabled)
			{
				return;
			}

			if (!array.IsMatrix || array.Shape.Length != 2 || array.Rank != 2)
			{
				Report($"ndarray should be a matrix for operation {operation} but wasn't (shape [{string.Join(", ", array.Shape)}])");
			}
		}

		public void InitAfterDeserialisation(INDArray array)
		{
			UnderlyingHandler.InitAfterDeserialisation(array);
		}

		public long GetSizeBytes(params INDArray[] array)
		{
			return UnderlyingHandler.GetSizeBytes(array);
		}

		public bool IsInterchangeable(IComputationHandler otherHandler)
		{
			return UnderlyingHandler.IsInterchangeable(otherHandler);
		}

		public INDArray NDArray(params long[] shape)
		{
			INDArray array = UnderlyingHandler.NDArray(shape);

			CheckNice(array);

			return array;
		}

		public INDArray NDArray<TOther>(TOther[] values, params long[] shape)
		{
			INDArray array = UnderlyingHandler.NDArray(values, shape);

			CheckNice(array);

			return array;
		}

		public INumber Number(object value)
		{
			return CheckNice(UnderlyingHandler.Number(value));
		}

		public IDataBuffer<T> DataBuffer<T>(T[] values)
		{
			return UnderlyingHandler.DataBuffer(values);
		}

		public INDArray AsNDArray(INumber number)
		{
			return CheckNice(UnderlyingHandler.AsNDArray(CheckNice(number)));
		}

		public INumber AsNumber(INDArray array, params long[] indices)
		{
			return CheckNice(UnderlyingHandler.AsNumber(CheckNice(array), indices));
		}

		public INDArray MergeBatch(params INDArray[] arrays)
		{
			for (int i = 0; i < arrays.Length; i++)
			{
				CheckNice(arrays[i], $"arrays[{i}]");
			}

			return CheckNice(UnderlyingHandler.MergeBatch(arrays));
		}

		public bool CanConvert(INDArray array, IComputationHandler otherHandler)
		{
			return UnderlyingHandler.CanConvert(array, otherHandler);
		}

		public INDArray Convert(INDArray array, IComputationHandler otherHandler)
		{
			return CheckNice(UnderlyingHandler.Convert(array, otherHandler));
		}

		public void Fill(INDArray filler, INDArray arrayToFill)
		{
			UnderlyingHandler.Fill(CheckNice(filler), CheckNice(arrayToFill));

			CheckNice(arrayToFill);
		}

		public void Fill<TOther>(TOther value, INDArray arrayToFill)
		{
			UnderlyingHandler.Fill(value, CheckNice(arrayToFill));

			CheckNice(arrayToFill);
		}

		public INDArray FlattenTime(INDArray array)
		{
			if (Enabled && array.Rank < 2) // two or three? technically 2 is enough ([BT]F) but 3 makes more sense
			{
				Report($"ndarray needs to have a rank of 2 or higher (was {array.Rank}) for {nameof(FlattenTime)} operation", array);
			}

			return CheckNice(UnderlyingHandler.FlattenTime(CheckNice(array)));
		}

		public INDArray FlattenFeatures(INDArray array)
		{
			if (Enabled && array.Rank < 3)
			{
				Report($"ndarray needs to have a rank of 3 or higher (was {array.Rank}) for {nameof(FlattenFeatures)} operation", array);
			}

			return CheckNice(UnderlyingHandler.FlattenFeatures(CheckNice(array)));
		}

		public INDArray FlattenTimeAndFeatures(INDArray array)
		{
			if (Enabled && array.Rank < 3)
			{
				Report($"ndarray needs to have a rank of 3 or higher (was {array.Rank}) for {nameof(FlattenTimeAndFeatures)} operation", array);
			}

			return CheckNice(UnderlyingHandler.FlattenTimeAndFeatures(CheckNice(array)));
		}

		public INDArray FlattenAllButLast(INDArray array)
		{
			return CheckNice(UnderlyingHandler.FlattenAllButLast(CheckNice(array)));
		}

		public TOther[] RowWiseTransform<TOther>(INDArray array, Func<INDArray, TOther> transformFunction)
		{
			if (transformFunction == null)
			{
				Report("transform function was null");
			}

			return UnderlyingHandler.RowWiseTransform(CheckNice(array), transformFunction);
		}

		public INDArray RowWise(INDArray array, Func<INDArray, INDArray> function)
		{
			if (function == null)
			{
				Report("function was null");
			}

			return CheckNice(UnderlyingHandler.RowWise(CheckNice(array), function));
		}

		public INDArray GetSlice(INDArray array, int rowIndex, int columnIndex, int rowLength, int columnLength)
		{
			if (Enabled)
			{
				if (rowIndex < 0 || columnIndex < 0)
				{
					Report($"row index and column index must be > 0, but were {rowIndex} and {columnIndex}", array, rowIndex, columnIndex);
				}

				if (rowLength <= 0 || columnLength <= 0)
				{
					Report($"row and column length must be > 0, but were {rowLength} and {columnLength}", array, rowLength, columnLength);
				}

				if (rowIndex + rowLength >= array.Shape[0] || columnIndex + columnLength >= array.Shape[1])
				{
					Report($"row index and column index must be < ndarray.shape[i], but were {rowIndex + rowLength} and {columnIndex + columnLength} (bounds were {array.Shape[0]} and {array.Shape[1]})", array, rowIndex, columnIndex, rowLength, columnLength);
				}
			}

			CheckMatrix(array, "get slice from matrix");

			return CheckNice(UnderlyingHandler.GetSlice(CheckNice(array), rowIndex, columnIndex, rowLength, columnLength));
		}

		public INDArray Add<TOther>(INDArray array, TOther value)
		{
			return CheckNice(UnderlyingHandler.Add(CheckNice(array), value));
		}

		public INDArray Add(INDArray array, INumber value)
		{
			return CheckNice(UnderlyingHandler.Add(CheckNice(array), CheckNice(value)));
		}

		public INDArray Add(INDArray a, INDArray b)
		{
			CheckEqualLength(a, b, "add ndarrays elementwise");

			return CheckNice(UnderlyingHandler.Add(CheckNice(a), CheckNice(b)));
		}

		public INumber Add(INumber a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Add(CheckNice(a), CheckNice(b)));
		}

		public INumber Add<TOther>(INumber a, TOther b)
		{
			return CheckNice(UnderlyingHandler.Add(CheckNice(a), b));
		}

		public INDArray Subtract<TOther>(TOther value, INDArray array)
		{
			return CheckNice(UnderlyingHandler.Subtract(value, CheckNice(array)));
		}

		public INDArray Subtract<TOther>(INDArray array, TOther value)
		{
			return CheckNice(UnderlyingHandler.Subtract(CheckNice(array), value));
		}

		public INDArray Subtract(INDArray array, INumber value)
		{
			return CheckNice(UnderlyingHandler.Subtract(CheckNice(array), CheckNice(value)));
		}

		public INDArray Subtract(INumber value, INDArray array)
		{
			return CheckNice(UnderlyingHandler.Subtract(CheckNice(value), CheckNice(array)));
		}

		public INDArray Subtract(INDArray a, INDArray b)
		{
			CheckEqualLength(a, b, "subtract ndarrays elementwise");

			return CheckNice(UnderlyingHandler.Subtract(CheckNice(a), CheckNice(b)));
		}

		public INumber Subtract(INumber a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Subtract(CheckNice(a), CheckNice(b)));
		}

		public INumber Subtract<TOther>(INumber a, TOther b)
		{
			return CheckNice(UnderlyingHandler.Subtract(CheckNice(a), b));
		}

		public INumber Subtract<TOther>(TOther a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Subtract(a, CheckNice(b)));
		}

		public INDArray Multiply<TOther>(INDArray array, TOther value)
		{
			return CheckNice(UnderlyingHandler.Multiply(CheckNice(array), value));
		}

		public INDArray Multiply(INDArray array, INumber value)
		{
			return CheckNice(UnderlyingHandler.Multiply(CheckNice(array), CheckNice(value)));
		}

		public INDArray Multiply(INDArray a, INDArray b)
		{
			CheckEqualLength(a, b, "multiply ndarrays elementwise");

			return CheckNice(UnderlyingHandler.Multiply(CheckNice(a), CheckNice(b)));
		}

		public INumber Multiply(INumber a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Multiply(CheckNice(a), CheckNice(b)));
		}

		public INumber Multiply<TOther>(INumber a, TOther b)
		{
			return CheckNice(UnderlyingHandler.Multiply(CheckNice(a), b));
		}

		public INDArray Dot(INDArray a, INDArray b)
		{
			CheckMatrix(a, "dot product");
			CheckMatrix(b, "dot product");

			if (Enabled && a.Shape[1] != b.Shape[0])
			{
				Report($"ndarray a.Shape[1] (={a.Shape[1]}) should be b.Shape[0] (={b.Shape[0]}) for operation matrix dot product", a, b);
			}

			return CheckNice(UnderlyingHandler.Dot(CheckNice(a), CheckNice(b)));
		}

		public INDArray Divide<TOther>(TOther value, INDArray array)
		{
			return CheckNice(UnderlyingHandler.Divide(value, CheckNice(array)));
		}

		public INDArray Divide<TOther>(INDArray array, TOther value)
		{
			return CheckNice(UnderlyingHandler.Divide(CheckNice(array), value));
		}

		public INDArray Divide(INDArray array, INumber value)
		{
			return CheckNice(UnderlyingHandler.Divide(CheckNice(array), CheckNice(value)));
		}

		public INDArray Divide(INDArray a, INDArray b)
		{
			CheckEqualLength(a, b, "divide ndarrays elementwise");

			return CheckNice(UnderlyingHandler.Divide(CheckNice(a), CheckNice(b)));
		}

		public INumber Divide(INumber a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Divide(CheckNice(a), CheckNice(b)));
		}

		public INumber Divide<TOther>(INumber a, TOther b)
		{
			return CheckNice(UnderlyingHandler.Divide(CheckNice(a), b));
		}

		public INDArray Pow(INDArray array, INumber value)
		{
			return CheckNice(UnderlyingHandler.Pow(CheckNice(array), CheckNice(value)));
		}

		public INDArray Pow<TOther>(INDArray array, TOther value)
		{
			return CheckNice(UnderlyingHandler.Pow(CheckNice(array), value));
		}

		public INumber Pow(INumber a, INumber b)
		{
			return CheckNice(UnderlyingHandler.Pow(CheckNice(a), CheckNice(b)));
		}

		public INumber Pow<TOther>(INumber a, TOther b)
		{
			return CheckNice(UnderlyingHandler.Pow(CheckNice(a), b));
		}

		public INumber Activation(string activation, INumber number)
		{
			return UnderlyingHandler.Activation(activation, number);
		}

		public INDArray Activation(string activation, INDArray array)
		{
			return UnderlyingHandler.Activation(activation, array);
		}

		public INumber Abs(INumber number)
		{
			return CheckNice(UnderlyingHandler.Abs(CheckNice(number)));
		}

		public INDArray Abs(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Abs(CheckNice(array)));
		}

		public INumber Sum(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Sum(CheckNice(array)));
		}

		public INumber Max(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Max(CheckNice(array)));
		}

		public int MaxIndex(INDArray array)
		{
			return UnderlyingHandler.MaxIndex(CheckNice(array));
		}

		public INumber Min(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Min(CheckNice(array)));
		}

		public int MinIndex(INDArray array)
		{
			return UnderlyingHandler.MinIndex(CheckNice(array));
		}

		public INDArray SquareRoot(INDArray array)
		{
			return CheckNice(UnderlyingHandler.SquareRoot(CheckNice(array)));
		}

		public INumber SquareRoot(INumber number)
		{
			return CheckNice(UnderlyingHandler.SquareRoot(CheckNice(number)));
		}

		public INDArray Log(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Log(CheckNice(array)));
		}

		public INumber Log(INumber number)
		{
			return CheckNice(UnderlyingHandler.Log(CheckNice(number)));
		}

		public INumber Determinate(INDArray array)
		{
			CheckMatrix(array, "matrix determinate");

			return CheckNice(UnderlyingHandler.Determinate(CheckNice(array)));
		}

		public INDArray Sin(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Sin(CheckNice(array)));
		}

		public INumber Sin(INumber number)
		{
			return CheckNice(UnderlyingHandler.Sin(CheckNice(number)));
		}

		public INDArray Asin(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Asin(CheckNice(array)));
		}

		public INumber Asin(INumber number)
		{
			return CheckNice(UnderlyingHandler.Asin(CheckNice(number)));
		}

		public INDArray Cos(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Asin(CheckNice(array)));
		}

		public INumber Cos(INumber number)
		{
			return CheckNice(UnderlyingHandler.Cos(CheckNice(number)));
		}

		public INDArray Acos(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Acos(CheckNice(array)));
		}

		public INumber Acos(INumber number)
		{
			return CheckNice(UnderlyingHandler.Acos(CheckNice(number)));
		}

		public INDArray Tan(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Tan(CheckNice(array)));
		}

		public INumber Tan(INumber number)
		{
			return CheckNice(UnderlyingHandler.Tan(CheckNice(number)));
		}

		public INDArray Atan(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Atan(CheckNice(array)));
		}

		public INumber Atan(INumber number)
		{
			return CheckNice(UnderlyingHandler.Atan(CheckNice(number)));
		}

		public INDArray Tanh(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Tanh(CheckNice(array)));
		}

		public INumber Tanh(INumber number)
		{
			return CheckNice(UnderlyingHandler.Tanh(CheckNice(number)));
		}

		public INDArray ReL(INDArray array)
		{
			return CheckNice(UnderlyingHandler.ReL(CheckNice(array)));
		}

		public INumber ReL(INumber number)
		{
			return CheckNice(UnderlyingHandler.ReL(CheckNice(number)));
		}

		public INDArray Sigmoid(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Sigmoid(CheckNice(array)));
		}

		public INumber Sigmoid(INumber number)
		{
			return CheckNice(UnderlyingHandler.Sigmoid(CheckNice(number)));
		}

		public INDArray SoftPlus(INDArray array)
		{
			return CheckNice(UnderlyingHandler.SoftPlus(CheckNice(array)));
		}

		public INumber SoftPlus(INumber number)
		{
			return CheckNice(UnderlyingHandler.SoftPlus(CheckNice(number)));
		}

		public INDArray SoftMax(INDArray array)
		{
			return CheckNice(UnderlyingHandler.SoftMax(CheckNice(array)));
		}

		public INumber StandardDeviation(INDArray array)
		{
			return CheckNice(UnderlyingHandler.StandardDeviation(CheckNice(array)));
		}

		public INumber Variance(INDArray array)
		{
			return CheckNice(UnderlyingHandler.Variance(CheckNice(array)));
		}

		public INDArray Clip(INDArray array, INumber minValue, INumber maxValue)
		{
			return CheckNice(UnderlyingHandler.Clip(CheckNice(array), CheckNice(minValue), CheckNice(maxValue)));
		}

		public void FillWithProbabilityMask(INDArray array, double probability)
		{
			if (Enabled && (probability < 0.0 || probability > 1.0))
			{
				Report($"probability for fill with probability mask should be 0.0 <= p <= 1.0 (was {probability}).");
			}

			UnderlyingHandler.FillWithProbabilityMask(CheckNice(array), probability);
			CheckNice(array);
		}

		public uint BeginTrace()
		{
			return UnderlyingHandler.BeginTrace();
		}

		public TTraceable Trace<TTraceable>(TTraceable traceable, uint traceTag) where TTraceable : ITraceable
		{
			return UnderlyingHandler.Trace(traceable, traceTag);
		}

		public TTraceable ClearTrace<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			return UnderlyingHandler.ClearTrace(traceable);
		}

		public void ComputeDerivativesTo(ITraceable traceable)
		{
			UnderlyingHandler.ComputeDerivativesTo(traceable);
		}

		public TTraceable GetDerivative<TTraceable>(TTraceable traceable) where TTraceable : ITraceable
		{
			return UnderlyingHandler.GetDerivative(traceable);
		}

		public bool IsNaN(INDArray array)
		{
			return UnderlyingHandler.IsNaN(array);
		}

		public bool IsNotFinite(INDArray number)
		{
			return UnderlyingHandler.IsNotFinite(number);
		}

		public bool IsNaN(INumber number)
		{
			return UnderlyingHandler.IsNaN(number);
		}

		public bool IsNotFinite(INumber number)
		{
			return UnderlyingHandler.IsNotFinite(number);
		}
	}
}
