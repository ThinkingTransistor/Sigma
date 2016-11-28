/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows.Controls;
// ReSharper disable InconsistentNaming

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
		protected T Content;

		/// <summary>
		/// Create a new <see cref="UIWrapper{T}"/> with a new T (e.g. new TabItem) as content.
		/// </summary>
		protected UIWrapper() : this(new T()) { }

		/// <summary>
		/// Create a new <see cref="UIWrapper{T}"/> with a passed T as content.
		/// </summary>
		/// <param name="content">The data that will be set to content.</param>
		protected UIWrapper(T content)
		{
			Content = content;
		}

		/// <summary>
		/// Property for the content. (The actual data which is wrapped). If you want to
		/// set the content of the <see cref="WrappedContent"/> use <code>WrappedContent.Content</code>.
		/// </summary>
		public T WrappedContent
		{
			get
			{
				return Content;
			}
			set
			{
				Content = value;
			}
		}

		/// <summary>
		/// Convert the <see cref="UIWrapper{T}"/> to the wrapped content.
		/// </summary>
		/// <param name="wrapper">The wrapper containing the content.</param>
		public static explicit operator T(UIWrapper<T> wrapper)
		{
			return wrapper.Content;
		}
	}
}
