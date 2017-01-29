/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;

namespace Sigma.Core.Monitors.WPF.Panels
{
	public class GenericPanel : GenericPanel<UIElement>
	{
		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="content">The content that will be placed inside the panel.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public GenericPanel(string title, UIElement content, object headerContent = null) : base(title, content, headerContent) {}
	}
	
	/// <summary>
	/// This panel is a wrapper for normal SigmaPanels - it provides a content with the correct type.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	public abstract class GenericPanel<T> : SigmaPanel where T : UIElement
	{
		private T _content;


		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="content">The content that will be placed inside the panel.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected GenericPanel(string title, T content, object headerContent = null) : this(title, headerContent)
		{
			Content = content;
		}

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected GenericPanel(string title, object headerContent = null) : base(title, headerContent)
		{}

		/// <summary>
		/// Set the content of the <see cref="GenericPanel"/> and the <see cref="SigmaPanel"/>. 
		/// </summary>
		public new T Content
		{
			get { return _content; }
			set
			{
				_content = value;
				base.Content = _content;
			}
		}
	}
}