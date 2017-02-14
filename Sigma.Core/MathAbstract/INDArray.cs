/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using System.Diagnostics.CodeAnalysis;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract
{
	/// <summary>
	/// An n-dimensional array of any data type with C-like ordering. Includes utility functions for scalar, vector and matrix manipulation. 
	/// 
	/// Why does INDArray not expose the underlying data? 
	/// Not exposing the internal data directly makes it easier to define the general workflow of how INDArrays should interact in an algorithm without having to think about underlying system data types. 
	/// That also allows anyone to roll their own implementation, defining where data is stored and how it is processed (System memory, GPU cache, the moon (beware of latency issues, batch calculations), ...).
	/// </summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")] // INdArray is stupid 
	public interface INDArray : ITraceable, IDeepCopyable
	{
		/// <summary>
		/// The rank of this ndarray (number of dimensions). 
		/// </summary>
		int Rank { get; }

		/// <summary>
		/// The length of this ndarray (total number of elements).
		/// </summary>
		long Length { get; }

		/// <summary>
		/// The shape of this ndarray (i.e. the dimensions).
		/// </summary>
		long[] Shape { get; }

		/// <summary>
		/// The strides of this ndarray. A stride at any dimension defines how many elements to skip to move to the next element along that dimension.
		/// </summary>
		long[] Strides { get; }

		/// <summary>
		/// Convenience flag indicating if this ndarray is a scalar (i.e. if its shape is {1})
		/// </summary>
		bool IsScalar { get; }

		/// <summary>
		/// Convenience flag indicating if this ndarray is a vector (i.e. if its shape is 1-dimensional)
		/// </summary>
		bool IsVector { get; }

		/// <summary>
		/// Convenience flag indicating if this ndarray is a matrix (i.e. if its shape is 2-dimensional)
		/// </summary>
		bool IsMatrix { get; }

		/// <summary>
		/// Get a COPY of the underlying data buffer as a certain data type. 
		/// </summary>
		/// <typeparam name="TOther">The type the buffer should have.</typeparam>
		/// <returns>The data buffer with the given data type.</returns>
		IDataBuffer<TOther> GetDataAs<TOther>();

		/// <summary>
		/// Get a value at a certain index as a certain type.
		/// Note: The value might have to be internally explicitly cast to the requested type (and thereby change). 
		/// </summary>
		/// <typeparam name="TOther">The type the value should have.</typeparam>
		/// <param name="indices">The indices.</param>
		/// <returns>The value at the given index as the given type.</returns>
		TOther GetValue<TOther>(params long[] indices);

		/// <summary>
		/// Set a value of a certain type at a certain index. 
		/// Note: The value might have to be internally explicitly cast to the internally used type (and thereby change). 
		/// </summary>
		/// <typeparam name="TOther">The type of the given value.</typeparam>
		/// <param name="value">The value.</param>
		/// <param name="indices">The indices.</param>
		void SetValue<TOther>(TOther value, params long[] indices);

		/// <summary>
		/// Get a slice of this ndarray of a certain region as a new ndarray with the same underlying data. 
		/// </summary>
		/// <param name="beginIndices">The begin indices (inclusively, where the slice should begin).</param>
		/// <param name="endIndices">The end indices (exclusively, where the slice should end).</param>
		/// <returns></returns>
		INDArray Slice(long[] beginIndices, long[] endIndices);

		/// <summary>
		/// Get a NEW ndarray with the same data but another shape (and different strides).
		/// Note: The total length of the ndarray cannot change.
		/// </summary>
		/// <param name="newShape">The new shape.</param>
		/// <returns>A NEW ndarray with the same data and the given shape.</returns>
		INDArray Reshape(params long[] newShape);

		/// <summary>
		/// Reshape THIS ndarray to a new shape (this operation also changes strides). 
		/// Note: The total length of the ndarray cannot change. 
		/// </summary>
		/// <param name="newShape"></param>
		/// <returns>This ndarray (for convenience).</returns>
		INDArray ReshapeSelf(params long[] newShape);

		/// <summary>
		/// Get a NEW ndarray with the same data but another permuted shape (and also different strides).
		/// Permutation occurs according to the content of the rearranged dimensions array, where each element represents the number of a dimension to swap with. The permutation array has to have the same number of dimensions as the actual shape.
		/// </summary>
		/// <param name="rearrangedDimensions">The dimensions to swap.</param>
		/// <returns>A NEW ndarray with the same data and a new permuted shape.</returns>
		INDArray Permute(params int[] rearrangedDimensions);

		/// <summary>
		/// Permute THIS ndarray to another permuted shape (and also different strides).
		/// Permutation occurs according to the content of the rearranged dimensions array, where each element represents the number of a dimension to swap with. The permutation array has to have the same number of dimensions as the actual shape.
		/// </summary>
		/// <param name="rearrangedDimensions">The dimensions to swap.</param>
		/// <returns>This ndarray (for convenience).</returns>
		INDArray PermuteSelf(params int[] rearrangedDimensions);

		/// <summary>
		/// Get a NEW ndarray with the same data but transposed (reversed dimensions).
		/// </summary>
		/// <returns>A NEW ndarray with the same data but transposed (reversed dimensions).</returns>
		INDArray Transpose();

		/// <summary>
		/// Transpose THIS ndarray (reverse dimensions). 
		/// </summary>
		/// <returns>This ndarray (for convenience).</returns>
		INDArray TransposeSelf();

		/// <summary>
		/// Get a NEW flattened ndarray with the same data but a flat shape (row vector). 
		/// </summary>
		/// <returns>A NEW flattened ndarray with the same data.</returns>
		INDArray Flatten();

		/// <summary>
		/// Flatten THIS ndarray to a flat row vector. 
		/// </summary>
		/// <returns></returns>
		INDArray FlattenSelf();
	}
}
