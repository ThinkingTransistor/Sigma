/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Handlers
{
	/// <summary>
	/// A computation backend handler. Creates and manages ndarrays, processes mathematical operations at scale. 
	/// Runtime checks argument checks are not performed by default for maximum performance. For debugging information, attach a DebugHandler. 
	/// </summary>
	public interface IComputationHandler
	{
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
		/// </summary>
		/// <param name="shape">The ndarray shape.</param>
		/// <returns>An ndarray with the given shape.</returns>
		INDArray NDArray(params long[] shape);

		/// <summary>
		/// Create a single value (i.e. number) with a certain initial value.
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
		/// Merge a number of ndarrays of the same TF shape along the Batch dimension (BTF format).
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
		/// </summary>
		/// <param name="array">The array for which a copy in this handler's format should be created.</param>
		/// <param name="otherHandler">The other handler which created the array.</param>
		/// <returns>A COPY of the contents of the given ndarray in this handler's format.</returns>
		INDArray Convert(INDArray array, IComputationHandler otherHandler);

		/// <summary>
		/// Fill an ndarray with the contents of another ndarray.
		/// </summary>
		/// <param name="filler">The filler ndarray (from which the values will be copied).</param>
		/// <param name="arrayToFill">The ndarray to fill.</param>
		void Fill(INDArray filler, INDArray arrayToFill);

		/// <summary>
		/// Fill an ndarray with a single value.
		/// </summary>
		/// <typeparam name="TOther">The type of the value.</typeparam>
		/// <param name="value">The value to set all elements of the ndarray to</param>
		/// <param name="arrayToFill">The ndarray to fill.</param>
		void Fill<TOther>(TOther value, INDArray arrayToFill);

		/// <summary>
		/// Add a constant value to all elements in an ndarray and put the result in another ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to add.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of adding value to each array element.</returns>
		INDArray Add<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Add a traceable number to all elements in an ndarray and put the result in another ndarray.
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
		/// Subtract a constant value from all elements in an ndarray and put the result in another ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to add.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Subtract a traceable number from all elements in an ndarray and put the result in another ndarray.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of subtracting value from each array element.</returns>
		INDArray Subtract(INDArray array, INumber value);

		/// <summary>
		/// Subtract an ndarray b from an ndarray a element-wise.
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The result of subtracting the ndarray b from the ndarray b element-wise.</returns>
		INDArray Subtract(INDArray a, INDArray b);

		/// <summary>
		/// Multiply a constant value with all elements in an ndarray and put the result in another ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to add.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of multiplying value with each array element.</returns>
		INDArray Multiply<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Multiply a traceable number with all elements in an ndarray and put the result in another ndarray.
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
		/// Get the dot product of two ndarrays a x b (a and b are assumed to be compatible matrices for dot products). 
		/// </summary>
		/// <param name="a">The first ndarray.</param>
		/// <param name="b">The second ndarray.</param>
		/// <returns>The dot product of the ndarray a x ndarray b.</returns>
		INDArray Dot(INDArray a, INDArray b);

		/// <summary>
		/// Divide all elements in an ndarray by a constant value and put the result in another ndarray.
		/// </summary>
		/// <typeparam name="TOther">The type of the value to add.</typeparam>
		/// <param name="array">The ndarray.</param>
		/// <param name="value">The value.</param>
		/// <returns>The result of dividing array element by the value.</returns>
		INDArray Divide<TOther>(INDArray array, TOther value);

		/// <summary>
		/// Divide all elements in an ndarray by a traceable number and put the result in another ndarray.
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
	}
}
