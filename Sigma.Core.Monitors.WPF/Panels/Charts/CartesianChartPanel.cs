using System.Collections.Generic;
using LiveCharts.Wpf;
using Sigma.Core.Monitors.WPF.Panels.Charts.Definitions;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class CartesianChartPanel : GenericPanel<CartesianChart>, IPointVisualiser
	{
		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public CartesianChartPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new CartesianChart();


		}

		public void Add(object value)
		{
		}

		public void AddRange(IEnumerable<object> values)
		{

		}
	}
}