using System;
using System.Windows;
using LiveCharts;
using LiveCharts.Wpf;

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public class LineChartPanel : SigmaPanel
	{
		public new CartesianChart Content { get; }

		public LineChartPanel(string title) : base(title)
		{
			Content = new CartesianChart();

			StepLineSeries stepLine = new StepLineSeries();
			stepLine.Values = new ChartValues<double> { 9, 6, 5, 7, 8, 9, 7, 6, 7, 5 };

			Content.AnimationsSpeed = new TimeSpan(0, 0, 0, 0, 100);

			Content.SetDrawMarginWidth(Content.GetDrawMarginElements() * 0.8);

			Content.Series.Add(stepLine);

			base.Content = Content;
			ContentGrid.Margin = new Thickness(20);
		}
	}
}