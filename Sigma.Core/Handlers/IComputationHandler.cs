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
		/// Create an ndarray of a certain shape.
		/// </summary>
		/// <param name="shape">The ndarray shape.</param>
		/// <returns>An ndarray with the given shape.</returns>
		INDArray Create(params long[] shape);

		/// <summary>
		/// Get the (estimated) size of this ndarray in bytes. 
		/// If the given ndarray is not of a format this handler can handle, throw an exception.
		/// </summary>
		/// <param name="array">The ndarray.</param>
		/// <returns>The (estimated) size of the given ndarray in bytes.</returns>
		long GetSizeBytes(INDArray array);
	}
}
