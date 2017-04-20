using System.Collections.Generic;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core.Monitors.WPF.Panels.Charts;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public static class ChartPanelVisualisation
	{
		public static void SetLineSmoothness<TChart, TSeries, TValues, TData>(this ChartPanel<TChart, TSeries, TValues, TData> chart, double lineSmoothness) where TChart : Chart, new() where TSeries : LineSeries, new() where TValues : IList<TData>, IChartValues, new()
		{
			foreach (TSeries series in chart.Series)
			{
				series.LineSmoothness = lineSmoothness;
			}
		}

		public static void Linearify<TChart, TSeries, TValues, TData>(this ChartPanel<TChart, TSeries, TValues, TData> chart) where TChart : Chart, new() where TSeries : LineSeries, new() where TValues : IList<TData>, IChartValues, new()
		{
			chart.SetLineSmoothness(0);
		}
	}
}