/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View
{
	/// <summary>
	/// This class is a wrapper for <see cref="ContentControl"/>s.
	/// It is often required that the functionality of those is wrapped;
	/// simply extend from this class and implement your custom behaviour.
	/// </summary>
	/// <typeparam name="T">The type that is wrapped.</typeparam>
	public abstract class UIWrapper<T> where T : ContentControl, new()
	{
		/// <summary>
		/// The content that is contained.
		/// </summary>
		protected T content;

		/// <summary>
		/// Create a new <see cref="UIWrapper{T}"/> with a new T (e.g. new TabItem) as content.
		/// </summary>
		public UIWrapper() : this(new T()) { }

		/// <summary>
		/// Create a new <see cref="UIWrapper{T}"/> with a passed T as content.
		/// </summary>
		/// <param name="content">The data that will be set to content.</param>
		public UIWrapper(T content)
		{
			this.content = content;
		}

		/// <summary>
		/// Property for the content. (The actual data).
		/// </summary>
		public T Content
		{
			get
			{
				return content;
			}
			set
			{
				content = value;
			}
		}

		/// <summary>
		/// Convert the <see cref="UIWrapper{T}"/> to the wrapped content.
		/// </summary>
		/// <param name="wrapper">The wrapper containing the content.</param>
		public static explicit operator T(UIWrapper<T> wrapper)
		{
			return wrapper.content;
		}
	}
}
