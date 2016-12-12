/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.MathAbstract
{
	/// <summary>
	/// A collection of utility methods for ndarrays.
	/// </summary>
	public static class NDArrayUtils
	{
		/// <summary>
		/// Calculate the stride of a given shape. 
		/// Note: The shape is not checked for correctness. To do that <see cref="CheckShape(long[])"/>.
		/// </summary>
		/// <param name="shape"></param>
		/// <returns></returns>
		public static long[] GetStrides(params long[] shape)
		{
			long[] strides = new long[shape.Length];

			long currentStride = 1;
			for (int i = shape.Length - 1; i >= 0; i--)
			{
				strides[i] = currentStride;
				currentStride *= shape[i];
			}

			return strides;
		}

		/// <summary>
		/// Check a shape for logical correctness (i.e. all values must be > 0, total length must be > 0). If incorrect, this method throws an appropriate argument exception.
		/// </summary>
		/// <param name="shape">The shape array to check.</param>
		/// <returns>The same shape array (for convenience).</returns>
		public static long[] CheckShape(params long[] shape)
		{
			if (shape.Length == 0)
			{
				throw new ArgumentException("Shape array cannot be of length 0.");
			}

			for (int i = 0; i < shape.Length; i++)
			{
				if (shape[i] <= 0)
				{
					throw new ArgumentException($"Invalid shape: all shape dimensions must be > 0, but dimension {i} was {shape[i]}.");
				}
			}

			return shape;
		}

		/// <summary>
		/// Get the flattened index given all dimensions indices and a certain ndarray shape and strides.
		/// </summary>
		/// <param name="shape">The shape of the ndarray.</param>
		/// <param name="strides">The strides of the ndarray.</param>
		/// <param name="indices">The indices, where each index represents one dimension of the shape (C style indexing, major/row first).</param>
		/// <returns>The flattened index given the dimensions indices and the ndarray's shape and strides.</returns>
		public static long GetFlatIndex(long[] shape, long[] strides, params long[] indices)
		{
			if (shape.Length != strides.Length || shape.Length != indices.Length)
			{
				throw new ArgumentException($"Shape, stride, and indices array length must be the same, but shape length was {shape.Length}, stride length {strides.Length}, and indices length {indices.Length}.");
			}

			long flatIndex = 0L;
			int length = shape.Length;

			for (int i = 0; i < length; i++)
			{
				if (indices[i] >= shape[i])
				{
					throw new IndexOutOfRangeException($"Indices must be smaller than their dimensions shape, but indices[{i}] was {indices[i]} and shape[{i}] was {shape[i]}.");
				}

				flatIndex += indices[i] * strides[i];
			}

			return flatIndex;
		}

		/// <summary>
		/// Get the per dimension indices corresponding to a certain flattened index and shape and strides of an ndarray.
		/// </summary>
		/// <param name="flatIndex">The flattened index.</param>
		/// <param name="shape">The shape of the ndarray.</param>
		/// <param name="strides">The strides of the ndarray.</param>
		/// <param name="resultIndices">If this argument is non-null and the same length as the shape and strides arrays the result will be stored here and no new indices array will be created.</param>
		/// <returns>The corresponding per dimension indices given a flat index, a shape and strides.</returns>
		public static long[] GetIndices(long flatIndex, long[] shape, long[] strides, long[] resultIndices = null)
		{
			if (shape.Length != strides.Length)
			{
				throw new ArgumentException($"Shape, stride (and [result]) array length must be the same, but shape length was {shape.Length} and stride length {strides.Length}.");
			}

			int rank = shape.Length;

			if (resultIndices == null)
			{
				resultIndices = new long[rank];
			}
			else if (resultIndices.Length != rank)
			{
				throw new ArgumentException($"Shape, stride (and [result]) array length must be the same, but shape length and stride length were {shape.Length} and result length {resultIndices.Length}.");
			}

			for (int i = 0; i < rank; i++)
			{
				resultIndices[i] = flatIndex / strides[i];
				flatIndex -= resultIndices[i] * strides[i];
			}

			return resultIndices;
		}

		/// <summary>
		/// Get slice indices along a certain dimension for a certain index, with all other indices set to 0 if <see cref="sliceEndIndex"/> is false or their shape limit if true.
		/// </summary>
		/// <param name="dimensionIndex">The dimension index, where the significant index will be set.</param>
		/// <param name="index">The significant index to set.</param>
		/// <param name="shape">The shape of the ndarray to be sliced.</param>
		/// <param name="copyResultShape">Indicate whether the resulting shape should be a copy from the passed shape or the original (modified).</param>
		/// <param name="sliceEndIndex">Indicate whether the slice indices should be calculated as a begin index (all except significant 0) or end index (all except significant shape limit).</param>
		/// <returns>The resulting slice indices, with the given index value at the significant dimension and 0 or shape limit in the others (depending on <see cref="sliceEndIndex"/>).</returns>
		public static long[] GetSliceIndicesAlongDimension(int dimensionIndex, long index, long[] shape, bool copyResultShape = true, bool sliceEndIndex = false)
		{
			if (shape == null) throw new ArgumentNullException(nameof(shape));

			if (index < 0 || index > shape.Length)
			{
				throw new ArgumentException($"Index must be >= 0 and < shape.Length (ndarray rank), but index was {index} and shape.Length {shape.Length}.");
			}

			if (dimensionIndex < 0 || dimensionIndex > shape.Length)
			{
				throw new ArgumentException($"Dimension index must be >= 0 and < shape.Length (ndarray rank), but dimension index was {index} and shape.Length {shape.Length}.");
			}

			long[] result = copyResultShape ? new long[shape.Length] : shape;

			if (sliceEndIndex)
			{
				for (int i = 0; i < result.Length; i++)
				{
					result[i] = shape[i];
				}
			}
			else
			{
				for (int i = 0; i < result.Length; i++)
				{
					result[i] = 0;
				}
			}

			result[dimensionIndex] = index;

			return result;
		}
	}
}
