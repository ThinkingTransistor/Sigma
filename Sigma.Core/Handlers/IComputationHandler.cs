/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Handlers
{
	/// <summary>
	/// A computation backend handler. Creates and manages ndarrays, processes mathematical operations at scale. 
	/// Runtime checks argument checks are not performed by default for maximum performance. For debugging information, attach a DebugHandler. 
	/// </summary>
	public interface IComputationHandler
	{
		/// <summary>
		/// The registry containing relevant parameters of this computation handler. 
		/// </summary>
		IRegistry Registry { get; }

		#region  Data (number, buffer, ndarray) creation and management

		/// <summary>
		/// The underlying data type processed and used in this computation handler. 
		/// </summary>
		IDataType DataType { get; }

		/// <summary>
		/// Initialise a just de-serialised ndarray of this handler's format with this handler and register and initialise components relevant to this handler.
		/// </summary>
		/// <param name="array">The ndarray to serialise.</param>
		void InitAfterDeserialisation(INDArray array);

		/// <summary>
		/// Get the (estimated) size of a number of ndarrays in system memory in bytes. 
		/// If any of the given ndarrays are not of a format this handler can handle, throw an exception.
		/// </summary>
		/// <param name="array">The ndarrays.</param>
		/// <returns>The (estimated) size of all given ndarray in bytes.</returns>
		long GetSizeBytes(params INDArray[] array);

		/// <summary>
		/// Check whether this handler and another handler and its contents are interchangeable (i.e. same format). 
		/// </summary>
		/// <param name="otherHandler">The other handler to check for.</param>
		/// <returns>A boolean indicating whether this handler and its contents are interchangeable with the given other handler.</returns>
		bool IsInterchangeable(IComputationHandler otherHandler);

		/// <summary>
		/// Create an ndarray of a certain shape.
		/// This is not a traceable operation. 
		/// </summary>
		/// <param name="shape">The ndarray shape.</param>
		/// <returns>An ndarray with the given shape.</returns>
		INDArray NDArray(params long[] shape);

		/// <summary>
		/// Create an ndarray of a certain shape with an initial set of values.
		/// This is not a traceable operation. 
		/// </summary>
		/// <param name="shape">The ndarray shape.</param>
		/// <param name="values">The values to fill the ndarray with.</param>
		/// <returns>An ndarray with the given shape.</returns>
		INDArray NDArray<TOther>(TOther[] values, params long[] shape);

		/// <summary>
		/// Create a single value (i.e. number) with a certain initial value.
		/// This is not a traceable operation. 
		/// </summary>
		/// <param name="value">The value to wrap in a single value wrapper.</param>
		/// <returns>A single value wrapper with the given value for computation.</returns>
		INumber Number(object value);

		/// <summary>
		/// Create a data buffer compatible with this handler from the given array.
		/// </summary>
		/// <typeparam name="T">The type of the data buffer / values array.</typeparam>
		/// <param name="values">The values array.</param>
		/// <returns>A data buffer compatible with this handler containing the given values.</returns>
		IDataBuffer<T> DataBuffer<T>(T[] values);

		/// <summary>
		/// Get a certain given number as an ndarray (with continuous tracing).
		/// </summary>
		/// <param name="number">The number-</param>
		/// <returns>The number as an 1x1 ndarray.</returns>
		INDArray AsNDArray(INumber number);

		/// <summary>
		/// Get a certain index of an ndarray as a number (with continuous tracing).
		/// </summary>
		/// <param name="array">The ndarray to get the number from.</param>
		/// <param name="indices">The indices.</param>
		/// <returns>The item in the given ndarray at the specified index as a number.</returns>
		INumber AsNumber(INDArray array, params long[] indices);

		/// <summary>
		/// Merge a number of ndarrays of the same TF shape along the Batch dimension (BTF format).
		///	This is not a traceable operation. 
		/// </summary>
		/// <param name="arrays">The ndarrays to merge (must be of same shape).</param>
		/// <returns>The merged ndarray consisting of the given ndarrays contents.</returns>
		INDArray MergeBatch(params INDArray[] arrays);

		/// <summary>
		/// Check whether this handler can (and should) convert an ndarray coming from another handler to this handler's ndarray format.
		/// </summary>
		/// <param name="array">The array which should be checked for convertibility.</param>
		/// <param name="otherHandler">The handler that created this ndarray comes from.</param>
		/// <returns>A boolean indicating whether the given array by the given other handler can be converted to this handler's format.</returns>
		bool CanConvert(INDArray array, IComputationHandler otherHandler);

		/// <summary>
		/// Converts a certain ndarray from another handler to this handler's format and returns a COPY of its contents in this handler's format.
		/// This is not a traceable operation. 
		/// </summary>
		/// <param name="array">The array for which a copy in this handler's format should be created.</param>
		/// <param name="otherHandler">The other handler which created the array.</param>
		/// <returns>A COPY of the contents of the given ndarray in this handler's format.</returns>
		INDArray Convert(INDArray array, IComputationHandler otherHandler);

		/// <summary>
		/// Fill an ndarray with the contents of another ndarray.
		/// This is not a traceable operation. 
		/// </summary>
		/// <param name="filler">The filler ndarray (from which the values will be copied).</param>
		/// <param name="arrayToFill">The ndarray to fill.</param>
		void Fill(INDArray filler, INDArray arrayToFill);

	    /// <summary>
	    /// Fill an ndarray with the contents of another ndarray within a specific range.
	    /// Note: The index ranges must be of the same size (in source and destination).
	    /// </summary>
	    /// <param name="filler">The filler ndarray (from which the values will be copied in the specified range).</param>
	    /// <param name="arrayToFill">The array to fill within the specified range.</param>
	    /// <param name="sourceBeginIndices">The begin indices in the filler array.</param>
	    /// <param name="sourceEndIndices">The end indices in the filler array.</param>
	    /// <param name="destinationBeginIndices">The begin indices in the array to fill.</param>
	    /// <param name="destinationEndIndices">The end indices in the array to fill.</param>
	    void Fill(INDArray filler, INDArray arrayToFill, long[] sourceBeginIndices, long[] sourceEndIndices, long[] destinationBeginIndices, long[] destinationEndIndices);

	    /// <summary>
	    /// Fill an ndarray with the contents of another ndarray within a specific range.
	    /// Note: The index ranges must be of the same size (in source and destination).
	    /// </summary>
	    /// <param name="filler">The filler ndarray (from which the values will be copied in the specified range).</param>
	    /// <param name="arrayToFill">The array to fill within the specified range.</param>
	    /// <param name="destinationBeginIndices">The begin indices in the array to fill.</param>
	    /// <param name="destinationEndIndices">The end indices in the array to fill.</param>
	    void Fill<T>(T[] filler, INDArray arrayToFill, long[] destinationBeginIndices, long[] destinationEndIndices);

		/// <summary>
		/// Fill an ndarray with a single value.
		/// This is not a traceable operation. 
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="value">The value to set all elements of the ndarray to</param>
		/// <param name="arrayToFill">The ndarray to fill.</param>
		void Fill<TOther>(TOther value, INDArray arrayToFill);

		#endregion

		#region INDArray dimension management (BatchTimeFeatures)

		/// <summary>
		/// Get an ndarray with flattened time dimension.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A flattened version of the given ndarray.</returns>
		INDArray FlattenTime(INDArray array);

		/// <summary>
		/// Get an ndarray with flattened feature dimensions.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A flattened version of the given ndarray.</returns>
		INDArray FlattenFeatures(INDArray array);

		/// <summary>
		/// Get an ndarray with flattened time and feature dimensions.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A flattened version of the given ndarray.</returns>
		INDArray FlattenTimeAndFeatures(INDArray array);

		/// <summary>
		/// Get an ndarray with all flattened dimensions but the last.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A flattened version of the given ndarray.</returns>
		INDArray FlattenAllButLast(INDArray array);

		/// <summary>
		/// Get an ndarray with the time and batch dimensions permuted.
		/// </summary>
		/// <param name="array"></param>
		/// <returns>A version of the given ndarray with the batch dimension permuted (data is affected).</returns>
		INDArray PermuteBatchAndTime(INDArray array);

		/// <summary>
		/// Transform an ndarray row-wise to another type (may also be ndarray).
		/// This is not required to be a traceable operation (and typically isn't).
		/// Note: Traceability should be consistent independent of type of <see cref="TOther"/>.  
		/// </summary>
		/// <typeparam name="TOther">The other type.</typeparam>
		/// <param name="array">The array to split row-wise and then transform.</param>
		/// <param name="transformFunction">The transform function to transform each row with.</param>
		/// <returns>An array of values as given by the transform function when applied to each row individually.</returns>
		TOther[] RowWiseTransform<TOther>(INDArray array, Func<INDArray, TOther> transformFunction);

		/// <summary>
		/// Apply a function along the second (column) dimension of an ndarray.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="function">The function to apply.</param>
		/// <returns>An ndarray with the given function applied column-wise to the given ndarray.</returns>
		INDArray RowWise(INDArray array, Func<INDArray, INDArray> function);

		/// <summary>
		/// Get a traceable slice of an ndarray as a matrix of a certain range.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="rowIndex">The row index (dimension 0 start).</param>
		/// <param name="columnIndex">The column index (dimension 1 start).</param>
		/// <param name="rowLength">The row length (dimension 0 length).</param>
		/// <param name="columnLength">The column length (dimension 1 length).</param>
		/// <returns>A slice of the given ndarray along the given range.</returns>
		INDArray GetSlice(INDArray array, int rowIndex, int columnIndex, int rowLength, int columnLength);

		#endregion

		#region Primitive binary mathematical operations

		/// <summary>
		/// Add a constant value to all elements in an ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of adding value to each array element.</returns>
		INDArray Add<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Add a traceable number to all elements in an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of adding value to each array element.</returns>
		INDArray Add(INDArray array, INumber value);

		/// <summary>
		/// Add an ndarray a to an ndarray b element-wise.
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The result of adding the ndarray a to the ndarray b element-wise.</returns>
		INDArray Add(INDArray a, INDArray b);

		/// <summary>
		/// Add a traceable number a to another traceable number b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of adding the number a to the number b.</returns>
		INumber Add(INumber a, INumber b);

		/// <summary>
		/// Add a traceable number to a constant value b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of adding the constant b to the number a.</returns>
		INumber Add<TOther>(INumber a, TOther b);

		/// <summary>
		/// Subtract  all elements in an ndarray from a constant value.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to subtract.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract<TOther>(TOther value, INDArray array);

		/// <summary>
		/// Subtract a constant value from all elements in an ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to subtract.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Subtract a traceable number from all elements in an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract(INDArray array, INumber value);

		/// <summary>
		/// Subtract all elements in an ndarray from a traceable number.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract(INumber value, INDArray array);

		/// <summary>
		/// Subtract an ndarray b from an ndarray a element-wise.
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The result of subtracting the ndarray b from the ndarray b element-wise.</returns>
		INDArray Subtract(INDArray a, INDArray b);

		/// <summary>
		/// Subtract a traceable number b from another traceable number a.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of subtracting the number b from the number a.</returns>
		INumber Subtract(INumber a, INumber b);

		/// <summary>
		/// Subtract a traceable number from a constant value b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of subtracting the constant b from the number a.</returns>
		INumber Subtract<TOther>(INumber a, TOther b);

		/// <summary>
		/// Subtract a constant value from a traceable number b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of subtracting the number a from the constant b.</returns>
		INumber Subtract<TOther>(TOther a, INumber b);

		/// <summary>
		/// Multiply a constant value with all elements in an ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of multiplying value with each array element.</returns>
		INDArray Multiply<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Multiply a traceable number with all elements in an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of multiplying value with each array element.</returns>
		INDArray Multiply(INDArray array, INumber value);

		/// <summary>
		/// Multiply an ndarray a with an ndarray b element-wise (similar to the Hadamard product).
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The result of multiplying the ndarray a with the ndarray b element-wise.</returns>
		INDArray Multiply(INDArray a, INDArray b);

		/// <summary>
		/// Multiply a traceable number a by another traceable number b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of multiplying the number a by the number b.</returns>
		INumber Multiply(INumber a, INumber b);

		/// <summary>
		/// Multiply a traceable number a by a constant value b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of multiplying the number a by the constant b.</returns>
		INumber Multiply<TOther>(INumber a, TOther b);

		/// <summary>
		/// Get the dot product of two ndarrays a x b (a and b are assumed to be compatible matrices for dot products). 
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The dot product of the ndarray a x ndarray b.</returns>
		INDArray Dot(INDArray a, INDArray b);

		/// <summary>
		/// Divide a constant value by all elements in an ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of dividing array element by the value.</returns>
		INDArray Divide<TOther>(TOther value, INDArray array);

		/// <summary>
		/// Divide all elements in an ndarray by a constant value.
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of dividing array element by the value.</returns>
		INDArray Divide<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Divide all elements in an ndarray by a traceable number.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of dividing array element by the value</returns>
		INDArray Divide(INDArray array, INumber value);

		/// <summary>
		/// Divide an ndarray b by an ndarray a element-wise (similar to the Hadamard product).
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The result of dividing the ndarray b by the ndarray a element-wise.</returns>
		INDArray Divide(INDArray a, INDArray b);

		/// <summary>
		/// Divide a traceable number a by another traceable number b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of dividing the number a by the number b.</returns>
		INumber Divide(INumber a, INumber b);

		/// <summary>
		/// Divide a traceable number a by a constant value b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of dividing the number a by the constant b.</returns>
		INumber Divide<TOther>(INumber a, TOther b);

		/// <summary>
		/// The power of an ndarray by a traceable number. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of the power of the ndarray to the given value.</returns>
		INDArray Pow(INDArray array, INumber value);

		/// <summary>
		/// The power of an ndarray by a constant. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of the power of the ndarray to the given value.</returns>
		INDArray Pow<TOther>(INDArray array, TOther value);

		/// <summary>
		/// The power a traceable number a to another traceable number b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of the power of the number a to the number b.</returns>
		INumber Pow(INumber a, INumber b);

		/// <summary>
		/// The power a traceable number a to a constant value b.
		/// </summary>
		/// <param name="a">The first number.</param>
		/// <param name="b">The second number.</param>
		/// <returns>The result of the power of the number a to the constant b.</returns>
		INumber Pow<TOther>(INumber a, TOther b);

		#endregion

		#region Primitive unary mathematical operations

		/// <summary>
		/// The absolute of a traceable number.
		/// </summary>
		/// <param name="number">The number.</param>
		/// <returns>The absolute number corresponding to the given number.</returns>
		INumber Abs(INumber number);

		/// <summary>
		/// The absolute of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The result of applying the absolute function element-wise to the given ndarray.</returns>
		INDArray Abs(INDArray array);

		/// <summary>
		/// The sum of an ndarray. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The sum of the given ndarray.</returns>
		INumber Sum(INDArray array);

		/// <summary>
		/// The maximum value of an ndarray. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The maximum value of the given ndarray.</returns>
		INumber Max(INDArray array);

		/// <summary>
		/// The index of the maximum value of an ndarray. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The index of the maximum value of the given ndarray.</returns>
		int MaxIndex(INDArray array);

		/// <summary>
		/// The minimum value of an ndarray. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The minimum value of the given ndarray.</returns>
		INumber Min(INDArray array);

		/// <summary>
		/// The index of the minimum value of an ndarray. 
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The index of the minimum value of the given ndarray.</returns>
		int MinIndex(INDArray array);

		/// <summary>
		/// The square root of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The square root of the given array element-wise.</returns>
		INDArray SquareRoot(INDArray array);

		/// <summary>
		/// The square root of a traceable number.
		/// </summary>
		/// <param name="number">The traceable number.</param>
		/// <returns>The square root of the given traceable number.</returns>
		INumber SquareRoot(INumber number);

		/// <summary>
		/// The logarithm base e of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The logarithm base e of the given array element-wise.</returns>
		INDArray Log(INDArray array);

		/// <summary>
		/// The logarithm base e of a traceable number.
		/// </summary>
		/// <param name="number">The traceable number.</param>
		/// <returns>The logarithm base e of the given traceable number.</returns>
		INumber Log(INumber number);

		/// <summary>
		/// The determinate of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The determinate of the given ndarray.</returns>
		INumber Determinate(INDArray array);

		#region Primitive trigonometric unary mathematical operations

		/// <summary>
		/// Apply the sine function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Sin(INDArray array);

		/// <summary>
		/// Apply the sine function to a traceable number.
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Sin(INumber number);

		/// <summary>
		/// Apply the inverse sine function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Asin(INDArray array);

		/// <summary>
		/// Apply the inverse sine function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Asin(INumber number);

		/// <summary>
		/// Apply the cosine function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Cos(INDArray array);

		/// <summary>
		/// Apply the cosine function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Cos(INumber number);

		/// <summary>
		/// Apply the inverse cosine function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Acos(INDArray array);

		/// <summary>
		/// Apply the inverse cosine function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Acos(INumber number);

		/// <summary>
		/// Apply the tangent function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Tan(INDArray array);

		/// <summary>
		/// Apply the tangent function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Tan(INumber number);

		/// <summary>
		/// Apply the inverse tangent function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Atan(INDArray array);

		/// <summary>
		/// Apply the inverse tangent function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Atan(INumber number);

		/// <summary>
		/// Apply the tangent hyperbolic function to an ndarray element-wise. 
		/// </summary>
		/// <param name="array">The ndarray to apply the function to.</param>
		/// <returns></returns>
		INDArray Tanh(INDArray array);

		/// <summary>
		/// Apply the tangent hyperbolic function to a traceable number. 
		/// </summary>
		/// <param name="number">The traceable number to apply the function to.</param>
		/// <returns></returns>
		INumber Tanh(INumber number);

		#endregion

		#endregion

		#region Complex mathematical operations (e.g. activation functions)

		#region Activation functions

		/// <summary>
		/// Apply a certain activation function to a number (e.g. 'rel', 'sigmoid', 'tanh').
		/// </summary>
		/// <param name="activation">The activation to apply.</param>
		/// <param name="number">The number.</param>
		/// <returns>The number with the activation function applied to it.</returns>
		INumber Activation(string activation, INumber number);

		/// <summary>
		/// Apply a certain activation function to a number (e.g. 'rel', 'sigmoid', 'tanh').
		/// </summary>
		/// <param name="activation">The activation to apply.</param>
		/// <param name="array">The array.</param>
		/// <returns>The array with the activation function applied to it.</returns>
		INDArray Activation(string activation, INDArray array);

		/// <summary>
		/// Apply the rectified linear function to an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The ndarray with the rectified linear function applied element-wise.</returns>
		INDArray ReL(INDArray array);

		/// <summary>
		/// Apply the rectified linear function to a traceable number.
		/// </summary>
		/// <param name="number">The traceable number.</param>
		/// <returns>The traceable number with the rectified linear function applied.</returns>
		INumber ReL(INumber number);

		/// <summary>
		/// Apply the sigmoid function to an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The ndarray with the sigmoid function applied element-wise.</returns>
		INDArray Sigmoid(INDArray array);

		/// <summary>
		/// Apply the sigmoid to a traceable number.
		/// </summary>
		/// <param name="number">The traceable number.</param>
		/// <returns>The traceable number with the sigmoid applied.</returns>
		INumber Sigmoid(INumber number);

		/// <summary>
		/// Apply the soft plus function to an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The ndarray with the soft plus function applied element-wise.</returns>
		INDArray SoftPlus(INDArray array);

		/// <summary>
		/// Apply the soft plus function to a traceable number.
		/// </summary>
		/// <param name="number">The traceable number.</param>
		/// <returns>The traceable number with the soft plus function applied.</returns>
		INumber SoftPlus(INumber number);

		/// <summary>
		/// Apply the soft max function to an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The ndarray with the soft plus function applied element-wise.</returns>
		INDArray SoftMax(INDArray array);

		#endregion

		/// <summary>
		/// The standard deviation of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The standard deviation of the given ndarray.</returns>
		INumber StandardDeviation(INDArray array);

		/// <summary>
		/// The variance of an ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The variance of the given ndarray.</returns>
		INumber Variance(INDArray array);

		/// <summary>
		/// Clip the contents of an ndarray to a certain range.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="minValue">The min value.</param>
		/// <param name="maxValue">The max value.</param>
		/// <returns>A clipped version of the given ndarray using the given range.</returns>
		INDArray Clip(INDArray array, INumber minValue, INumber maxValue);

		/// <summary>
		/// Fill an ndarray with a probability mask (0 or 1, with a <see cref="probability"/> chance of it being 1).
		/// This is not a traceable operation. 
		/// Note: This method does not return anything as not to be confused with the traceable operations that do return something.
		/// </summary>
		/// <param name="array">The array to fill.</param>
		/// <param name="probability">The probability that a number will be 1.</param>
		void FillWithProbabilityMask(INDArray array, double probability);

		#endregion

		#region Automatic differentiation and tracing operations

		/// <summary>
		/// Begin an AD tracing session with a certain tag.
		/// </summary>
		/// <returns>A new tag for tracing certain ndarrays and numbers.</returns>
		uint BeginTrace();

		/// <summary>
		/// Trace a certain traceable's (ndarray, number) mathematical operations for automatic differentiation.
		/// </summary>
		/// <typeparam name="TTraceable">The type of the traceable to trace.</typeparam>
		/// <param name="traceable">The traceable to trace (ndarray, number).</param>
		/// <param name="traceTag">The tracing tag (for automatic differentiation with certain traced members).</param>
		/// <returns>The traceable with the trace put on it.</returns>
		TTraceable Trace<TTraceable>(TTraceable traceable, uint traceTag) where TTraceable : ITraceable;

		/// <summary>
		/// Clear a traceable's (ndarray, number) trace.
		/// </summary>
		/// <typeparam name="TTraceable">The type of the traceable to trace.</typeparam>
		/// <param name="traceable">The traceable to clear.</param>
		/// <returns>The cleared traceable without a trace.</returns>
		TTraceable ClearTrace<TTraceable>(TTraceable traceable) where TTraceable : ITraceable;

		/// <summary>
		/// Compute the derivatives (adjoints) with respect to a certain traceable member (ndarray, number), starting the evaluation tree at the given traceable. 
		/// </summary>
		void ComputeDerivativesTo(ITraceable traceable);

		/// <summary>
		/// Get the derivative of a certain traceable after its derivative adjoints have been computed in a <see cref="ComputeDerivativesTo"/> operation.
		/// </summary>
		/// <typeparam name="TTraceable">The type of the traceable (ndarray or number).</typeparam>
		/// <param name="traceable">The traceable.</param>
		/// <returns>The derivative of the given traceable with as computed in the preceding <see cref="ComputeDerivativesTo"/> operation, or null if no derivatives were computed.</returns>
		TTraceable GetDerivative<TTraceable>(TTraceable traceable) where TTraceable : ITraceable;

		#endregion

		#region Debugging helpers

		/// <summary>
		/// Check if an ndarray contains any NaN values.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A boolean indicating if the given ndarray contains any NaN values.</returns>
		bool IsNaN(INDArray array);

		/// <summary>
		/// Check if an ndarray contains any infinite values.
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>A boolean indicating if the given ndarray contains any infinite values.</returns>
		bool IsNotFinite(INDArray array);

		/// <summary>
		/// Check if a number is NaN.
		/// </summary>
		/// <param name="number">The number.</param>
		/// <returns>A boolean indicating if a number is NaN.</returns>
		bool IsNaN(INumber number);

		/// <summary>
		/// Check if a number is infinite.
		/// </summary>
		/// <param name="number">The number.</param>
		/// <returns>A boolean indicating if a number is infinite.</returns>
		bool IsNotFinite(INumber number);

		#endregion
	}
}
