using System.Windows;

namespace Sigma.Core.Monitors.WPF.Panels
{
	/// <summary>
	/// A generic panel that works with XAML UI elements (elments with a parameterless cosntructor).
	/// It automatically generates passed <see cref="UIElement"/>.
	/// </summary>
	/// <typeparam name="T">The type of the XAML control.</typeparam>
	public class XamlPanel<T> : GenericPanel<T> where T: UIElement, new()
	{
		/// <summary>
		///     Create a XamlPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="content">The content that will be placed inside the panel.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public XamlPanel(string title, T content, object headerContent = null) : base(title, content, headerContent)
		{
		}

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public XamlPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new T();
		}
	}
}