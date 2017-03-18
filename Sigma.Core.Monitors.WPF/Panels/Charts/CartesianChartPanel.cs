/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using LiveCharts;
using LiveCharts.Wpf;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class CartesianChartPanel : ChartPanel<CartesianChart, LineSeries, ChartValues<double>, double>
	{
		/// <summary>
		///     Create a ChartPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public CartesianChartPanel(string title, object headerContent = null) : base(title, headerContent) { }
	}
}