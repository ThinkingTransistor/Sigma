/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using Sigma.Core.Data;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff
{
	/// <summary>
	/// A default, in-system-memory implementation of the INDArray interface.
	/// </summary>
	/// <typeparam name="T">The data type of this ndarray.</typeparam>
	[Serializable]
	[SuppressMessage("ReSharper", "InconsistentNaming")] //AdndArray looks stupid
	public class ADNDArray<T> : INDArray
	{
		internal readonly IDataBuffer<T> Data;

		[NonSerialized]
		private IComputationHandler _associatedHandler;

		/// <inheritdoc />
		public IComputationHandler AssociatedHandler
		{
			get { return _associatedHandler;}
			set { _associatedHandler = value; }
		}

		/// <inheritdoc />
		public long Length { get; private set; }

		/// <inheritdoc />
		public int Rank { get; private set; }

		/// <inheritdoc />
		public long[] Shape { get; private set; }

		/// <inheritdoc />
		public long[] Strides { get; private set; }

		/// <inheritdoc />
		public bool IsScalar { get; private set; }

		/// <inheritdoc />
		public bool IsVector { get; private set; }

		/// <inheritdoc />
		public bool IsMatrix { get; private set; }

		/// <summary>
		/// Create a vectorised ndarray of a certain buffer.
		/// </summary>
		/// <param name="buffer">The buffer to back this ndarray.</param>
		public ADNDArray(IDataBuffer<T> buffer)
		{
			Initialise(new long[] { 1, (int) buffer.Length }, NDArrayUtils.GetStrides(1, (int) buffer.Length));

			Data = buffer;
		}

		/// <summary>
		/// Create a ndarray of a certain buffer and shape.
		/// </summary>
		/// <param name="buffer">The buffer to back this ndarray.</param>
		/// <param name="shape">The shape.</param>
		public ADNDArray(IDataBuffer<T> buffer, long[] shape)
		{
			if (buffer.Length < ArrayUtils.Product(shape))
			{
				throw new ArgumentException($"Buffer must contain the entire shape, but buffer length was {buffer.Length} and total shape length {ArrayUtils.Product(shape)} (shape = {ArrayUtils.ToString(shape)}).");
			}

			Initialise(NDArrayUtils.CheckShape(shape), NDArrayUtils.GetStrides(shape));

			Data = buffer;
		}

		/// <summary>
		/// Create a vectorised ndarray of a certain array (array will be COPIED into a data buffer).
		/// </summary>
		/// <param name="data">The data to use to fill this ndarray.</param>
		public ADNDArray(T[] data)
		{
			Initialise(new long[] { 1, data.Length }, NDArrayUtils.GetStrides(1, data.Length));

			Data = new DataBuffer<T>(data, 0L, data.Length);
		}

		/// <summary>
		/// Create an ndarray of a certain array (array will be COPIED into a data buffer) and shape.
		/// Total shape length must be smaller or equal than the data array length.
		/// </summary>
		/// <param name="data">The data to use to fill this ndarray.</param>
		/// <param name="shape">The shape.</param>
		public ADNDArray(T[] data, params long[] shape)
		{
			if (data.Length < ArrayUtils.Product(shape))
			{
				throw new ArgumentException($"Data must contain the entire shape, but data length was {data.Length} and total shape length {ArrayUtils.Product(shape)} (shape = {ArrayUtils.ToString(shape)}).");
			}

			Initialise(NDArrayUtils.CheckShape(shape), NDArrayUtils.GetStrides(shape));

			Data = new DataBuffer<T>(data, 0L, Length);
		}

		/// <summary>
		/// Create an ndarray of a certain shape (initialised with zeros).
		/// </summary>
		/// <param name="shape">The shape.</param>
		public ADNDArray(params long[] shape)
		{
			Initialise(NDArrayUtils.CheckShape(shape), NDArrayUtils.GetStrides(shape));

			Data = new DataBuffer<T>(Length);
		}

	    /// <inheritdoc />
	    public virtual object DeepCopy()
		{
			return new ADNDArray<T>((IDataBuffer<T>) Data.DeepCopy(), (long[]) Shape.Clone()).SetAssociatedHandler(AssociatedHandler);
		}

		/// <summary>
		/// Set the associated handler of this ndarrray.
		/// </summary>
		/// <param name="handler">The associated handler.</param>
		/// <returns>This ndarray (for convenience).</returns>
		internal ADNDArray<T> SetAssociatedHandler(IComputationHandler handler)
		{
			AssociatedHandler = handler;

			return this;
		}

		private void Initialise(long[] shape, long[] strides)
		{
			Shape = shape;
			Strides = strides;
			Length = ArrayUtils.Product(shape);

			Rank = shape.Length;

			IsScalar = Rank == 1 && shape[0] == 1;
			IsVector = Rank == 2 && shape[0] == 1 ^ shape[1] == 1;
			IsMatrix = Rank == 2 && shape[0] >= 1 && shape[1] >= 1;
		}

		/// <inheritdoc />
		protected virtual void Reinitialise(long[] shape, long[] strides)
		{
			Initialise(shape, strides);
		}

		/// <inheritdoc />
		public IDataBuffer<TOther> GetDataAs<TOther>()
		{
			return Data.GetValuesAs<TOther>(0L, Data.Length);
		}

		/// <inheritdoc />
		public TOther GetValue<TOther>(params long[] indices)
		{
			return Data.GetValueAs<TOther>(NDArrayUtils.GetFlatIndex(Shape, Strides, indices));
		}

		/// <inheritdoc />
		public void SetValue<TOther>(TOther value, params long[] indices)
		{
			Data.SetValue((T) Convert.ChangeType(value, Data.Type.UnderlyingType), NDArrayUtils.GetFlatIndex(Shape, Strides, indices));
		}

		/// <inheritdoc />
		protected long[] GetSlicedShape(long[] beginIndices, long[] endIndices)
		{
			if (beginIndices.Length != endIndices.Length)
			{
				throw new ArgumentException($"Begin and end indices arrays must be of same length, but begin indices was of length {beginIndices.Length} and end indices {endIndices.Length}.");
			}

			long[] slicedShape = new long[beginIndices.Length];

			for (int i = 0; i < slicedShape.Length; i++)
			{
				slicedShape[i] = endIndices[i] - beginIndices[i];

				if (slicedShape[i] < 0)
				{
					throw new ArgumentException($"Begin indices must be smaller than end indices, but begin indices at [{i}] was {beginIndices[i]} and end indices at [{i}] {endIndices[i]}");
				}
			}

			return slicedShape;
		}

		/// <inheritdoc />
		public virtual INDArray Slice(long[] beginIndices, long[] endIndices)
		{
			long[] slicedShape = GetSlicedShape(beginIndices, endIndices);

			//we want the end indices to be inclusive for easier handling
			endIndices = endIndices.Select(i => i - 1).ToArray();

			long absoluteBeginOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, beginIndices);
			long absoluteEndOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, endIndices);
			long length = absoluteEndOffset - absoluteBeginOffset + 1;

			return new ADNDArray<T>(new DataBuffer<T>(Data, absoluteBeginOffset, length), slicedShape);
		}

		/// <inheritdoc />
		public INDArray Flatten()
		{
			return Reshape(0, Length);
		}

		/// <inheritdoc />
		public INDArray FlattenSelf()
		{
			return ReshapeSelf(0, Length);
		}

		/// <inheritdoc />
		public virtual INDArray Reshape(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			return new ADNDArray<T>(Data, newShape);
		}

		/// <inheritdoc />
		public virtual INDArray ReshapeSelf(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			Reinitialise(NDArrayUtils.CheckShape(newShape), NDArrayUtils.GetStrides(newShape));

			return this;
		}

		/// <inheritdoc />
		public virtual INDArray Permute(params int[] rearrangedDimensions)
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

			long[] newShape = ArrayUtils.PermuteArray(Shape, rearrangedDimensions);

			ADNDArray<T> permuted = (ADNDArray<T>) Reshape(newShape);
			
			_InternalPermuteSelf(Data.Data, (int) Data.Offset, (int) Data.Length, rearrangedDimensions, Shape, newShape);

			return permuted;
		}

		/// <inheritdoc />
		public virtual INDArray PermuteSelf(params int[] rearrangedDimensions)
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

			long[] newShape = ArrayUtils.PermuteArray(Shape, rearrangedDimensions);

			_InternalPermuteSelf(Data.Data, (int)Data.Offset, (int)Data.Length, rearrangedDimensions, Shape, newShape);

			ReshapeSelf(newShape);

			return this;
		}

		/// <inheritdoc />
		public INDArray Transpose()
		{
			return Permute(ArrayUtils.Range(Rank - 1, 0));
		}

		/// <inheritdoc />
		public INDArray TransposeSelf()
		{
			return PermuteSelf(ArrayUtils.Range(Rank - 1, 0));
		}

		/// <summary>
		/// Permute this ndarray's data according to rearranged dimensions in place. 
		/// Note: Arguments are not checked for correctness (and are asssumed to be correct).
		/// </summary>
		/// <param name="rearrangedDimensions">The re-arranged dimensions (numbered 0 to rank - 1).</param>
		/// <param name="originalShape">The original shape.</param>
		/// <param name="rearrangedShape">The new, re-arranged shape.</param>
		/// <param name="data">The data array.</param>
		/// <param name="offset">The data array offset.</param>
		/// <param name="length">The data array length.</param>
		internal static void _InternalPermuteSelf(T[] data, int offset, int length, int[] rearrangedDimensions, long[] originalShape, long[] rearrangedShape)
		{
			long[] originalStrides = NDArrayUtils.GetStrides(originalShape);
			long[] rearrangedStrides = NDArrayUtils.GetStrides(rearrangedShape);

			long[] bufferIndices = new long[rearrangedDimensions.Length];
			BitArray traversedIndicies = new BitArray(length);

			for (int i = 0; i < length; i++)
			{
				int currentIndex = i;
				T previousValue = data[offset + currentIndex];

				if (traversedIndicies[i]) continue;

				do
				{
					NDArrayUtils.GetIndices(currentIndex, originalShape, originalStrides, bufferIndices);

					bufferIndices = ArrayUtils.PermuteArray(bufferIndices, rearrangedDimensions);

					int swapIndex = (int) NDArrayUtils.GetFlatIndex(rearrangedShape, rearrangedStrides, bufferIndices);

					T nextPreviousValue = data[offset + swapIndex];
					if (swapIndex != currentIndex)
					{
						data[offset + swapIndex] = previousValue;
					}
					previousValue = nextPreviousValue;

					traversedIndicies[currentIndex] = true;

					currentIndex = swapIndex;
				} while (i != currentIndex && !traversedIndicies[currentIndex]);
			}
		}

		private void CheckRearrangedDimensions(int[] rearrangedDimensions)
		{
			int rank = Rank;

			if (rearrangedDimensions.Length != rank)
			{
				throw new ArgumentException($"Rearrange dimensions array must of be same length as shape array (i.e. rank), but rearrange dimensions array was of size {rearrangedDimensions.Length} and this ndarray is of rank {Rank}");
			}

			for (int i = 0; i < rank; i++)
			{
				if (rearrangedDimensions[i] < 0 || rearrangedDimensions[i] >= rank)
				{
					throw new ArgumentException($"All rearrange dimensions must be >= 0 and < rank {Rank}, but rearrangedDimensions[{i}] was {rearrangedDimensions[i]}.");
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

		/// <inheritdoc />
		public override string ToString()
		{
			return ToString(element => element.ToString());
		}

		/// <summary>
		/// A function that maps an element to a string.
		/// </summary>
		/// <param name="element">The element.</param>
		/// <returns>The string.</returns>
		public delegate string ToStringElement(T element);

		/// <summary>
		/// Constructs a string representing the contents of this ndarray, formatted properly and somewhat customisable. 
		/// </summary>
		/// <returns>A fancy string representing the contents of this ndarray.</returns>
		public string ToString(ToStringElement toStringElement, int dimensionNewLine = 2, bool printSeperator = true)
		{
			if (toStringElement == null)
			{
				toStringElement = element => element.ToString();
			}

			int rank = Rank;
			int lastIndex = rank - 1;
			int openBraces = 0;

			long[] indices = new long[rank];
			long[] shape = Shape;
			long[] strides = Strides;
			long length = Length;

			StringBuilder builder = new StringBuilder();

			builder.Append("ndarray with shape " + ArrayUtils.ToString(shape) + ": ");

			if (dimensionNewLine < rank)
			{
				builder.Append('\n');
			}

			for (long i = 0; i < length; i++)
			{
				indices = NDArrayUtils.GetIndices(i, shape, strides, indices);

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

				builder.Append(toStringElement(Data.GetValue(i)));

				if (printSeperator && indices[lastIndex] < shape[lastIndex] - 1)
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
					builder.Append("\n ");

					for (int y = 0; y < openBraces; y++)
					{
						builder.Append(' ');
					}
				}
			}

			return builder.ToString();
		}
	}
}
