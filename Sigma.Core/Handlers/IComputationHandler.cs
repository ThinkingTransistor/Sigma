/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using Sigma.Core.Math;

namespace Sigma.Core.Handlers
{
	/// <summary>
	/// A computation backend handler. Creates and manages ndarrays, implements their mathematical relations. 
	/// </summary>
	public interface IComputationHandler
	{
		/// <summary>
		/// The underlying data type processed and used in this computation handler. 
		/// </summary>
		IDataType DataType { get; }

		/// <summary>
		/// Initialise a deserialised ndarray of this handler's format with this handler and register and initialise components relevant to this handler.
		/// </summary>
		/// <param name="array">The ndarray to serialise.</param>
		/// <param name="stream">The stream to serialise to.</param>
		void InitAfterDeserialisation(INDArray array);

		/// <summary>
		/// Get the (estimated) size of a number of ndarrays in bytes. 
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
		INDArray Create(params long[] shape);

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
		/// <returns>A COPY of the contents of the given ndarray in this handler's format.</returns>
		INDArray Convert(INDArray array, IComputationHandler otherHandler);

		/// <summary>
		/// Fill an ndarray with the contents of another ndarray.
		/// </summary>
		/// <param name="arrayToFill">The ndarray to fill.</param>
		/// <param name="filler">The filler ndarray (from which the values will be copied).</param>
		void Fill(INDArray arrayToFill, INDArray filler);
	}
}
