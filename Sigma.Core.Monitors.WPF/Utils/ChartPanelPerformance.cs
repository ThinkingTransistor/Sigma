/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core.Monitors.WPF.Panels.Charts;

namespace Sigma.Core.Monitors.WPF.Utils
{
	/// <summary>
	/// This extension class, adds abilities to improve the performance of <see cref="ChartPanel{TChart,TSeries,TData}"/>s. 
	/// </summary>
	public static class ChartPanelPerformance
	{
		/// <summary>
		/// Set all required actions to improve the performance of the <see cref="ChartPanel{TChart,TSeries,TData}"/>.
		/// This method allows to disable animations, hovering, and tooltips. It will not reenable them. 
		/// </summary>
		public static void Fast<TChart, TSeries, TData>(this ChartPanel<TChart, TSeries, TData> chart, bool animationsEnabled = false, bool hoverEnabled = false, bool pointGemeotryEnabled = false) where TChart : Chart, new() where TSeries : Series, new()
		{
			if (!animationsEnabled)
			{
				chart.Content.DisableAnimations = false;
			}

			if (!hoverEnabled)
			{
				chart.Content.Hoverable = false;
				chart.Content.DataTooltip = null;
			}

			if (!pointGemeotryEnabled)
			{
				foreach (TSeries series in chart.Series)
				{
					series.PointGeometry = null;
				}
			}
		}
	}
}