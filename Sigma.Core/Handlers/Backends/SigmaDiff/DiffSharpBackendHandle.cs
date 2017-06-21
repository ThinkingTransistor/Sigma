/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using DiffSharp.Backend;
using static DiffSharp.Util;
using Microsoft.FSharp.Core;
using Sigma.Core.Persistence;
using Sigma.Core.Utils;
using Array = System.Array;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// A DiffSharp backend handle, as passed to the backend provider and used by Sigma.DiffSharp internally for direct operations on Blas and Lapack backends.
	/// </summary>
	/// <typeparam name="T">The primitive data type processed by this backend handle.</typeparam>
	[Serializable]
	public abstract class DiffSharpBackendHandle<T> : Backend<T>, ISerialisationNotifier
	{
		private IDictionary<int, IList<T[]>> _bufferedSessionArrays;
		private IDictionary<int, IList<T[]>> _currentSessionArrays;
		private ISet<T[]> _limboSessionArrays; // arrays that aren't automatically freed at the end of a session, have to be especially marked (e.g. for parameters)

		public bool BufferSessions { get; set; }
		public long BackendTag { get; set; }
		public IBlasBackend BlasBackend { get; set; }
		public ILapackBackend LapackBackend { get; set; }

		internal DiffSharpBackendHandle(IBlasBackend blasBackend, ILapackBackend lapackBackend, long backendTag)
		{
			if (blasBackend == null) throw new ArgumentNullException(nameof(blasBackend));
			if (lapackBackend == null) throw new ArgumentNullException(nameof(lapackBackend));

			BlasBackend = blasBackend;
			LapackBackend = lapackBackend;
			BackendTag = backendTag;

			_bufferedSessionArrays = new Dictionary<int, IList<T[]>>();
			_currentSessionArrays = new Dictionary<int, IList<T[]>>();
			_limboSessionArrays = new HashSet<T[]>();
		}

		private void _InternalAddToCurrentSession(T[] array)
		{
			if (!_currentSessionArrays.ContainsKey(array.Length))
			{
				_currentSessionArrays.Add(array.Length, new List<T[]>());
			}

			_currentSessionArrays[array.Length].Add(array);
		}

		private T[] _InternalGetBufferedArray(int length)
		{
			if (!_bufferedSessionArrays.ContainsKey(length) || _bufferedSessionArrays[length].Count == 0)
			{
				return null;
			}

			IList<T[]> buffer = _bufferedSessionArrays[length];

			lock (buffer)
			{
				T[] array = buffer[buffer.Count - 1];

				buffer.RemoveAt(buffer.Count - 1);

				return array;
			}
		}

		internal void ClearSessionBuffers()
		{
			_bufferedSessionArrays.Clear();
			_currentSessionArrays.Clear();
			_limboSessionArrays.Clear();
		}

		internal void TransferSessionBuffers()
		{
			_bufferedSessionArrays.Clear();
			_bufferedSessionArrays.AddAll(_currentSessionArrays);
			_currentSessionArrays.Clear();
		}

		internal void MarkLimbo(T[] array)
		{
			if (BufferSessions && _currentSessionArrays.ContainsKey(array.Length) && _currentSessionArrays[array.Length].Remove(array))
			{
				_limboSessionArrays.Add(array);
			}
		}

		internal void FreeLimbo(T[] array)
		{
			if (BufferSessions && _limboSessionArrays.Contains(array))
			{
				_limboSessionArrays.Remove(array);
				_InternalAddToCurrentSession(array);
			}
		}

		/// <inheritdoc />
		public T[] CreateUninitialisedArray(int length)
		{
			T[] array;

			if (!BufferSessions || (array = _InternalGetBufferedArray(length)) == null)
			{
				array = new T[length];
			}

			if (BufferSessions)
			{
				_InternalAddToCurrentSession(array);
			}

			return array;
		}

		/// <inheritdoc />
		public T[] CreateZeroArray(int length)
		{
			return CreateValueArray(length, default(T));
		}

		/// <inheritdoc />
		public T[] CreateValueArray(int length, T initialValue)
		{
			T[] array;
			bool alreadyInitialised = false;

			if (!BufferSessions || (array = _InternalGetBufferedArray(length)) == null)
			{
				array = new T[length];
				alreadyInitialised = true;
			}

			if (BufferSessions)
			{
				_InternalAddToCurrentSession(array);
			}

			if (!alreadyInitialised)
			{
				if (initialValue.Equals(default(T)))
				{
					Array.Clear(array, 0, array.Length);
				}
				else
				{
					for (var i = 0; i < array.Length; i++)
					{
						array[i] = initialValue;
					}
				}
			}

			return array;
		}

		public abstract ISigmaDiffDataBuffer<T> CreateDataBuffer(T[] values);
		public abstract T Mul_Dot_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> n);
		public abstract T L1Norm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T L2Norm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T SupNorm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T Sum_V(ISigmaDiffDataBuffer<T> value);
		public abstract T Sum_M(ISigmaDiffDataBuffer<T> value);
		public abstract ISigmaDiffDataBuffer<T> Add_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Add_V_V_InPlace(ISigmaDiffDataBuffer<T> obj0, int obj1, ISigmaDiffDataBuffer<T> obj2, int obj3, int obj4);
		public abstract ISigmaDiffDataBuffer<T> Add_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_V_S(ISigmaDiffDataBuffer<T> a, T b);
		public abstract ISigmaDiffDataBuffer<T> Mul_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Mul_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Mul_M_V_Add_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b, ISigmaDiffDataBuffer<T> obj2);
		public abstract ISigmaDiffDataBuffer<T> Mul_V_M(ISigmaDiffDataBuffer<T> a, ShapedDataBufferView<T> b);
		public abstract FSharpOption<ISigmaDiffDataBuffer<T>> Solve_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract FSharpOption<ISigmaDiffDataBuffer<T>> SolveSymmetric_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Diagonal_M(ShapedDataBufferView<T> a);
		public abstract ISigmaDiffDataBuffer<T> ReshapeCopy_MRows_V(ShapedDataBufferView<T> value);
		public abstract ShapedDataBufferView<T> Mul_Out_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ShapedDataBufferView<T> Add_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Add_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Add_V_MCols(ISigmaDiffDataBuffer<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Sub_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Sub_M_S(ShapedDataBufferView<T> a, T b);
		public abstract ShapedDataBufferView<T> Sub_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_M_M_Add_V_MCols(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b, ISigmaDiffDataBuffer<T> obj2);
		public abstract ShapedDataBufferView<T> Mul_Had_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract FSharpOption<ShapedDataBufferView<T>> Inverse_M(ShapedDataBufferView<T> a);
		public abstract FSharpOption<T> Det_M(ShapedDataBufferView<T> a);
		public abstract ShapedDataBufferView<T> Transpose_M(ShapedDataBufferView<T> a);
		public abstract ShapedDataBufferView<T> Permute_M(ShapedDataBufferView<T> array, int[] rearrangedDimensions);
		public abstract ShapedDataBufferView<T> Reshape_M(ShapedDataBufferView<T> array, long[] newShape);
		public abstract ShapedDataBufferView<T> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<T> value);
		public abstract ShapedDataBufferView<T> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<T> value);
		public abstract ShapedDataBufferView<T> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<T> value);
		public abstract ISigmaDiffDataBuffer<T> Map_F_V(MapOp mapOp, FSharpFunc<T, T> function, ISigmaDiffDataBuffer<T> value);
		public abstract ISigmaDiffDataBuffer<T> Map_F_S_V(T other, MapOp mapOp, FSharpFunc<T, T> function, ISigmaDiffDataBuffer<T> value);
		public abstract ISigmaDiffDataBuffer<T> Map2_F_V_V(MapOp mapOp, FSharpFunc<T, FSharpFunc<T, T>> function, ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ShapedDataBufferView<T> Map_F_M(MapOp mapOp, FSharpFunc<T, T> function, ShapedDataBufferView<T> value);
		public abstract ShapedDataBufferView<T> Map_F_S_M(T other, MapOp mapOp, FSharpFunc<T, T> function, ShapedDataBufferView<T> value);
		public abstract ShapedDataBufferView<T> Map2_F_M_M(MapOp mapOp, FSharpFunc<T, FSharpFunc<T, T>> function, ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			_bufferedSessionArrays = new Dictionary<int, IList<T[]>>();
			_currentSessionArrays = new Dictionary<int, IList<T[]>>();
		}
	}
}
