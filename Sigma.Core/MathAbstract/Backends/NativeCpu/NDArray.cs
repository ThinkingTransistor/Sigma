/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sigma.Core.Data;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract.Backends.NativeCpu
{
	/// <summary>
	/// A default, in-system-memory implementation of the INDArray interface.
	/// </summary>
	/// <typeparam name="T">The data type of this ndarray.</typeparam>
	[Serializable]
	[SuppressMessage("ReSharper", "InconsistentNaming")] //INdArray looks stupid
	public class NDArray<T> : INDArray, IDeepCopyable
	{
		internal readonly IDataBuffer<T> Data;

		[NonSerialized]
		private IComputationHandler _associatedHandler;

		public IComputationHandler AssociatedHandler
		{
			get { return _associatedHandler;}
			set { _associatedHandler = value; }
		}

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
			Initialise(new long[] { 1, (int) buffer.Length }, NDArrayUtils.GetStrides(1, (int) buffer.Length));

			Data = buffer;
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

			Initialise(NDArrayUtils.CheckShape(shape), NDArrayUtils.GetStrides(shape));

			Data = buffer;
		}

		/// <summary>
		/// Create a vectorised ndarray of a certain array (array will be COPIED into a data buffer).
		/// </summary>
		/// <param name="data">The data to use to fill this ndarray.</param>
		public NDArray(T[] data)
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
		public NDArray(T[] data, params long[] shape)
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
		public NDArray(params long[] shape)
		{
			Initialise(NDArrayUtils.CheckShape(shape), NDArrayUtils.GetStrides(shape));

			Data = new DataBuffer<T>(Length);
		}

		public NDArray<T> Copy()
		{
			return new NDArray<T>(Data, Shape).SetAssociatedHandler(AssociatedHandler);
		}

		public object DeepCopy()
		{
			return new NDArray<T>((IDataBuffer<T>) Data.DeepCopy(), (long[]) Shape.Clone()).SetAssociatedHandler(AssociatedHandler);
		}

		/// <summary>
		/// Set the associated handler of this ndarrray.
		/// </summary>
		/// <param name="handler">The associated handler.</param>
		/// <returns>This ndarray (for convenience).</returns>
		internal NDArray<T> SetAssociatedHandler(IComputationHandler handler)
		{
			AssociatedHandler = handler;

			return this;
		}

		private void Initialise(long[] shape, long[] strides)
		{
			//if it's a vector with a single dimension we convert to a matrix (row-vector) for easier and faster use
			if (shape.Length == 1 && shape[0] > 1)
			{
				shape = new[] { 1, shape[0] };
				strides = NDArrayUtils.GetStrides(shape);
			}

			Shape = shape;
			Strides = strides;
			Length = ArrayUtils.Product(shape);

			Rank = shape.Length;

			IsScalar = Rank == 1 && shape[0] == 1;
			IsVector = Rank == 2 && shape[0] == 1 ^ shape[1] == 1;
			IsMatrix = Rank == 2 && shape[0] > 1 && shape[1] > 1;
		}

		public IDataBuffer<TOther> GetDataAs<TOther>()
		{
			return Data.GetValuesAs<TOther>(0L, Data.Length);
		}

		public TOther GetValue<TOther>(params long[] indices)
		{
			return Data.GetValueAs<TOther>(NDArrayUtils.GetFlatIndex(Shape, Strides, indices));
		}

		public void SetValue<TOther>(TOther value, params long[] indices)
		{
			Data.SetValue((T) Convert.ChangeType(value, Data.Type.UnderlyingType), NDArrayUtils.GetFlatIndex(Shape, Strides, indices));
		}

		public INDArray Slice(long[] beginIndices, long[] endIndices)
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

				//we want the end indices to be inclusive for easier handling
				endIndices[i]--;
			}

			long absoluteBeginOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, beginIndices);
			long absoluteEndOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, endIndices);
			long length = absoluteEndOffset - absoluteBeginOffset + 1;

			return new NDArray<T>(new DataBuffer<T>(Data, absoluteBeginOffset, length), slicedShape);
		}

		public INDArray Flatten()
		{
			return Reshape(0, Length);
		}

		public INDArray FlattenSelf()
		{
			return ReshapeSelf(0, Length);
		}

		public INDArray Reshape(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			return new NDArray<T>(Data, newShape);
		}

		public INDArray ReshapeSelf(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			Initialise(NDArrayUtils.CheckShape(newShape), NDArrayUtils.GetStrides(newShape));

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

			long[] newShape = ArrayUtils.PermuteArray(Shape, rearrangedDimensions);

			return new NDArray<T>(Data, newShape);
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

			long[] newShape = ArrayUtils.PermuteArray(Shape, rearrangedDimensions);

			Initialise(newShape, NDArrayUtils.GetStrides(newShape));

			return this;
		}

		public INDArray Transpose()
		{
			return Permute(ArrayUtils.Range(Rank - 1, 0));
		}

		public INDArray TransposeSelf()
		{
			return PermuteSelf(ArrayUtils.Range(Rank - 1, 0));
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
			return ToString(element => element.ToString());
		}

		public delegate string ToStringElement(T element);

		/// <summary>
		/// Constructs a string representing the contents of this ndarray, formatted properly and somewhat customisable. 
		/// </summary>
		/// <returns>A fancy string representing the contents of this ndarray.</returns>
		public string ToString(ToStringElement toStringElement, int dimensionNewLine = 1, bool printSeperator = true)
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

			builder.Append("ndarray with shape " + ArrayUtils.ToString(shape) + ":\n");

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
					builder.Append('\n');

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
