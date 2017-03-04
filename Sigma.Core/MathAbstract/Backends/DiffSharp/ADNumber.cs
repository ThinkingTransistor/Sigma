/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;

namespace Sigma.Core.MathAbstract.Backends.DiffSharp
{
	/// <summary>
	/// A default implementation of the <see cref="INumber"/> interface.
	/// Represents single mathematical value (i.e. number), used for interaction between ndarrays and handlers (is more expressive and faster). 
	/// </summary>
	/// <typeparam name="T">The data type of this single value.</typeparam>
	[Serializable]
	public class ADNumber<T> : INumber
	{
		private T _value;

		[NonSerialized]
		private IComputationHandler _associatedHandler;

		public IComputationHandler AssociatedHandler
		{
			get { return _associatedHandler; }
			set { _associatedHandler = value; }
		}

		/// <summary>
		/// Create a single value (i.e. number) with a certain initial value.
		/// </summary>
		/// <param name="value">The initial value to wrap.</param>
		public ADNumber(T value)
		{
			_value = value;
		}

		public object Value
		{
			get { return _value; }
			set { SetValue((T) Convert.ChangeType(value, typeof(T))); }
		}

		/// <summary>
		/// Get the underlying value as another type.
		/// </summary>
		/// <typeparam name="TOther">The other type.</typeparam>
		/// <returns>The value as the other type.</returns>
		public TOther GetValueAs<TOther>()
		{
			return (TOther) System.Convert.ChangeType(_value, typeof(TOther));
		}

		internal virtual void SetValue(T value)
		{
			_value = value;
		}

		internal ADNumber<T> SetAssociatedHandler(IComputationHandler handler)
		{
			AssociatedHandler = handler;

			return this;
		}

		public object DeepCopy()
		{
			return new ADNumber<T>(_value).SetAssociatedHandler(AssociatedHandler);
		}

		public override string ToString()
		{
			return "number " + _value;
		}
	}
}
