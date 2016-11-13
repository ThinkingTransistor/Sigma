/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using Sigma.Core.Utils;
using System;
using System.Text;

namespace Sigma.Core.Math
{
	/// <summary>
	/// A default, in system memory implementation of the INDArray interface. 
	/// </summary>
	/// <typeparam name="T"></typeparam>
	[Serializable]
	public class NDArray<T> : INDArray
	{
		internal IDataBuffer<T> data;

		public long Length { get; private set; }

		public int Rank { get; private set; }

		public long[] Shape { get; private set; }

		public long[] Strides { get; private set; }

		public bool IsScalar { get; private set; }

		public bool IsVector { get; private set; }

		public bool IsMatrix { get; private set; }

		/// <summary>
		/// Create a vectorised ndarray of a certain buffer.
		/// </summary>
		/// <param name="buffer">The buffer to back this ndarray.</param>
		public NDArray(IDataBuffer<T> buffer)
		{
			Initialise(new long[] { 1, (int) buffer.Length }, GetStrides(1, (int) buffer.Length));

			this.data = buffer;
		}

		/// <summary>
		/// Create a ndarray of a certain buffer and shape.
		/// </summary>
		/// <param name="buffer">The buffer to back this ndarray.</param>
		/// <param name="shape">The shape.</param>
		public NDArray(IDataBuffer<T> buffer, long[] shape)
		{
			if (buffer.Length < ArrayUtils.Product(shape))
			{
				throw new ArgumentException($"Buffer must contain the entire shape, but buffer length was {buffer.Length} and total shape length {ArrayUtils.Product(shape)} (shape = {ArrayUtils.ToString(shape)}).");
			}

			Initialise(CheckShape(shape), GetStrides(shape));

			this.data = buffer;
		}

		/// <summary>
		/// Create a vectorised ndarray of a certain array (array will be COPIED into a data buffer).
		/// </summary>
		/// <param name="data">The data to use to fill this ndarray.</param>
		public NDArray(T[] data)
		{
			Initialise(new long[] { 1, (int) data.Length }, GetStrides(1, (int) data.Length));

			this.data = new DataBuffer<T>(data, 0L, data.Length);
		}

		/// <summary>
		/// Create an ndarray of a certain array (array will be COPIED into a data buffer) and shape.
		/// Total shape length must be smaller or equal than the data array length.
		/// </summary>
		/// <param name="data">The data to use to fill this ndarray.</param>
		/// <param name="shape">The shape.</param>
		public NDArray(T[] data, params long[] shape)
		{
			if (data.Length < ArrayUtils.Product(shape))
			{
				throw new ArgumentException($"Data must contain the entire shape, but data length was {data.Length} and total shape length {ArrayUtils.Product(shape)} (shape = {ArrayUtils.ToString(shape)}).");
			}

			Initialise(CheckShape(shape), GetStrides(shape));

			this.data = new DataBuffer<T>(data, 0L, this.Length);
		}

		/// <summary>
		/// Create an ndarray of a certain shape (initialised with zeros).
		/// </summary>
		/// <param name="shape">The shape.</param>
		public NDArray(params long[] shape)
		{
			Initialise(CheckShape(shape), GetStrides(shape));

			this.data = new DataBuffer<T>(this.Length);
		}

		public NDArray<T> Copy()
		{
			return new NDArray<T>(this.data, this.Shape);
		}

		private void Initialise(long[] shape, long[] strides)
		{
			//if it's a vector with a single dimension we convert to a matrix (row-vector) for easier and faster use
			if (shape.Length == 1 && shape[0] > 1)
			{
				shape = new long[] { 1, shape[0] };
				strides = GetStrides(shape);
			}

			this.Shape = shape;
			this.Strides = strides;
			this.Length = ArrayUtils.Product(shape);

			this.Rank = shape.Length;

			this.IsScalar = Rank == 1 && shape[0] == 1;
			this.IsVector = Rank == 2 && shape[0] == 1 ^ shape[1] == 1;
			this.IsMatrix = Rank == 2 && shape[0] > 1 && shape[1] > 1;
		}

		public IDataBuffer<TOther> GetDataAs<TOther>()
		{
			return this.data.GetValuesAs<TOther>(0L, this.data.Length);
		}

		public TOther GetValue<TOther>(params long[] indices)
		{
			return this.data.GetValueAs<TOther>(GetFlatIndex(this.Shape, this.Strides, indices));
		}

		public void SetValue<TOther>(TOther value, params long[] indices)
		{
			this.data.SetValue((T) Convert.ChangeType(value, this.data.Type.UnderlyingType), GetFlatIndex(this.Shape, this.Strides, indices));
		}

		public INDArray Flatten()
		{
			return this.Reshape(0, this.Length);
		}

		public INDArray FlattenSelf()
		{
			return this.ReshapeSelf(0, this.Length);
		}

		public INDArray Reshape(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			return new NDArray<T>(this.data, newShape);
		}

		public INDArray ReshapeSelf(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			this.Initialise(CheckShape(newShape), GetStrides(newShape));

			return this;
		}

		public INDArray Permute(params int[] rearrangedDimensions)
		{
			bool sameOrder = true;
			for (int i = 0; i < rearrangedDimensions.Length; i++)
			{
				if (rearrangedDimensions[i] != i)
				{
					sameOrder = false;

					break;
				}
			}

			//nothing to do here
			if (sameOrder)
			{
				return this;
			}

			CheckRearrangedDimensions(rearrangedDimensions);

			long[] newShape = ArrayUtils.PermuteArray(this.Shape, rearrangedDimensions);

			return new NDArray<T>(this.data, newShape);
		}

		public INDArray PermuteSelf(params int[] rearrangedDimensions)
		{
			bool sameOrder = true;
			for (int i = 0; i < rearrangedDimensions.Length; i++)
			{
				if (rearrangedDimensions[i] != i)
				{
					sameOrder = false;

					break;
				}
			}

			//nothing to do here
			if (sameOrder)
			{
				return this;
			}

			CheckRearrangedDimensions(rearrangedDimensions);

			long[] newShape = ArrayUtils.PermuteArray(this.Shape, rearrangedDimensions);

			this.Initialise(newShape, GetStrides(newShape));

			return this;
		}

		public INDArray Transpose()
		{
			return Permute(ArrayUtils.Range(this.Rank - 1, 0));
		}

		public INDArray TransposeSelf()
		{
			return PermuteSelf(ArrayUtils.Range(this.Rank - 1, 0));
		}

		private void CheckRearrangedDimensions(int[] rearrangedDimensions)
		{
			int rank = this.Rank;

			if (rearrangedDimensions.Length != rank)
			{
				throw new ArgumentException($"Rearrange dimensions array must of be same length as shape array (i.e. rank), but rearrange dimensions array was of size {rearrangedDimensions.Length} and this ndarray is of rank {this.Rank}");
			}

			for (int i = 0; i < rank; i++)
			{
				if (rearrangedDimensions[i] < 0 || rearrangedDimensions[i] >= rank)
				{
					throw new ArgumentException($"All rearrange dimensions must be >= 0 and < rank, but rearrangedDimensions[{i}] was {rearrangedDimensions[i]}.");
				}

				for (int y = 0; y < rank; y++)
				{
					if (i != y && rearrangedDimensions[i] == rearrangedDimensions[y])
					{
						throw new ArgumentException($"All rearranged dimensions must be unique, but rearrangedDimensions[{i}] and rearrangedDimensions[{i}] both were {i}.");
					}
				}
			}
		}

		public override string ToString()
		{
			return this.ToString();
		}

		public delegate string ToStringElement(T element);

		/// <summary>
		/// Constructs a string representing the contents of this ndarray, formatted properly and somewhat customisable. 
		/// </summary>
		/// <returns>A fancy string representing the contents of this ndarray.</returns>
		public string ToString(ToStringElement toStringElement = null, int dimensionNewLine = 1)
		{
			if (toStringElement == null)
			{
				toStringElement = element => element.ToString();
			}

			int rank = this.Rank;
			int lastIndex = rank - 1;
			int openBraces = 0;

			long[] indices = new long[rank];
			long[] shape = this.Shape;
			long[] strides = this.Strides;
			long length = this.Length;

			StringBuilder builder = new StringBuilder();

			for (long i = 0; i < length; i++)
			{
				indices = GetIndices(i, shape, strides, indices);

				for (int y = rank - 1; y >= 0; y--)
				{
					if (indices[y] == 0)
					{
						builder.Append('[');
						openBraces++;
					}
					else
					{
						break;
					}
				}

				builder.Append(toStringElement(this.data.GetValue(i)));

				if (indices[lastIndex] < shape[lastIndex] - 1)
				{
					builder.Append(", ");
				}

				bool requestNewLine = false;

				int maxRankNewLine = rank - dimensionNewLine;

				for (int y = rank - 1; y >= 0; y--)
				{
					if (indices[y] == shape[y] - 1)
					{
						builder.Append(']');
						openBraces--;

						if (y > 0 && indices[y - 1] != shape[y - 1] - 1)
						{
							builder.Append(", ");

							if (!requestNewLine && y < maxRankNewLine)
							{
								requestNewLine = true;
							}
						}
					}
					else
					{
						break;
					}
				}

				if (requestNewLine)
				{
					builder.Append('\n');

					for (int y = 0; y < openBraces; y++)
					{
						builder.Append(' ');
					}
				}
			}

			return builder.ToString();
		}

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
	}
}
