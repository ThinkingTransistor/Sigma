namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	public class RectanglePanel : GenericPanel<RectangleCanvas>
	{
		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public RectanglePanel(string title, int width, int height, int size, object headerContent = null) : base(title, headerContent)
		{
			Content = new RectangleCanvas
			{
				GridHeight = height,
				GridWidth =  width,
				PointSize = size,
			};

			Content.UpdateRects();
		}
	}
}