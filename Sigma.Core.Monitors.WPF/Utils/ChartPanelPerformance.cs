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
		/// This method disables animations, hovering, and tooltips. 
		/// </summary>
		public static void Fast<TChart, TSeries, TData>(this ChartPanel<TChart, TSeries, TData> chart) where TChart : Chart, new() where TSeries : Series, new()
		{
			chart.Content.DisableAnimations = true;
			chart.Content.Hoverable = false;
			chart.Content.DataTooltip = null;
		}
	}
}