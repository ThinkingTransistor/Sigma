//
// ImpObservableCollection.cs
//
// Copyright (c) 2010, Ashley Davis, @@email@@, @@website@@
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted 
// provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this list of conditions 
//   and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//   and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;

namespace Sigma.Core.Monitors.WPF.NetView.Utils
{
	/// <summary>
	///     An implementation of observable collection that contains a duplicate internal
	///     list that is retained momentarily after the list is cleared.
	///     This is so that observers can undo events, etc on the list after it has been cleared and
	///     raised a CollectionChanged event with a Reset action.
	/// </summary>
	public class ImpObservableCollection<T> : ObservableCollection<T>, ICloneable
	{
		/// <summary>
		///     Set to 'true' when in a collection changed event.
		/// </summary>
		private bool inCollectionChangedEvent;

		/// <summary>
		///     Inner list.
		/// </summary>
		private readonly List<T> inner = new List<T>();

		public ImpObservableCollection()
		{
		}

		public ImpObservableCollection(IEnumerable<T> range) :
			base(range)
		{
		}

		public ImpObservableCollection(IList<T> list) :
			base(list)
		{
			inner.AddRange(list);
		}

		public object Clone()
		{
			ImpObservableCollection<T> clone = new ImpObservableCollection<T>();
			foreach (ICloneable obj in this)
				clone.Add((T) obj.Clone());

			return clone;
		}

		public void AddRange(T[] range)
		{
			foreach (T item in range)
				Add(item);
		}

		public void AddRange(IEnumerable range)
		{
			foreach (T item in range)
				Add(item);
		}

		public void AddRange(ICollection<T> range)
		{
			foreach (T item in range)
				Add(item);
		}

		public void RemoveRange(T[] range)
		{
			foreach (T item in range)
				Remove(item);
		}

		public void RemoveRange(IEnumerable range)
		{
			foreach (T item in range)
				Remove(item);
		}

		public void RemoveRange(ImpObservableCollection<T> range)
		{
			foreach (T item in range)
				Remove(item);
		}

		public void RemoveRangeAt(int index, int count)
		{
			for (int i = 0; i < count; ++i)
				RemoveAt(index);
		}

		public void RemoveRange(ICollection<T> range)
		{
			foreach (T item in range)
				Remove(item);
		}

		public void RemoveRange(ICollection range)
		{
			foreach (T item in range)
				Remove(item);
		}

		protected override void OnCollectionChanged(NotifyCollectionChangedEventArgs e)
		{
			Trace.Assert(!inCollectionChangedEvent);

			base.OnCollectionChanged(e);

			inCollectionChangedEvent = true;

			try
			{
				if (e.Action == NotifyCollectionChangedAction.Reset)
				{
					if (inner.Count > 0)
						OnItemsRemoved(inner);

					inner.Clear();
				}

				if (e.OldItems != null)
				{
					foreach (T item in e.OldItems)
						inner.Remove(item);

					OnItemsRemoved(e.OldItems);
				}

				if (e.NewItems != null)
				{
					foreach (T item in e.NewItems)
						inner.Add(item);

					OnItemsAdded(e.NewItems);
				}
			}
			finally
			{
				inCollectionChangedEvent = false;
			}
		}

		protected virtual void OnItemsAdded(ICollection items)
		{
			if (ItemsAdded != null)
				ItemsAdded(this, new CollectionItemsChangedEventArgs(items));
		}

		protected virtual void OnItemsRemoved(ICollection items)
		{
			if (ItemsRemoved != null)
				ItemsRemoved(this, new CollectionItemsChangedEventArgs(items));
		}

		/// <summary>
		///     Event raised when items have been added.
		/// </summary>
		public event EventHandler<CollectionItemsChangedEventArgs> ItemsAdded;

		/// <summary>
		///     Event raised when items have been removed.
		/// </summary>
		public event EventHandler<CollectionItemsChangedEventArgs> ItemsRemoved;

		public T[] ToArray()
		{
			return inner.ToArray();
		}

		public T2[] ToArray<T2>()
			where T2 : class
		{
			T2[] array = new T2[Count];
			int i = 0;
			foreach (T obj in this)
			{
				array[i] = obj as T2;
				++i;
			}

			return array;
		}
	}
}