namespace Sigma.Core.MathAbstract.Backends.NativeCpu
{
	/// <summary>
	/// A default implementation of the <see cref="INumber"/> interface.
	/// Represents single mathematical value (i.e. number), used for interaction between ndarrays and handlers (is more expressive and faster). 
	/// </summary>
	/// <typeparam name="T">The data type of this single value.</typeparam>
	public class Number<T> : INumber
	{
		private T _value;

		/// <summary>
		/// Create a single value (i.e. number) with a certain initial value.
		/// </summary>
		/// <param name="value">The initial value to wrap.</param>
		public Number(T value)
		{
			_value = value;
		}

		public object Value
		{
			get { return Value; }
			set { _value = (T) value; }
		}
	}
}
