using System;
using System.Windows;
using LiveCharts;
using LiveCharts.Wpf;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class LineChartPanel : SigmaPanel
	{
		public LineChartPanel(string title) : base(title)
		{
			Content = new CartesianChart();

			StepLineSeries stepLine = new StepLineSeries
			{
				Values = new ChartValues<double> { 9, 6, 5, 7, 8, 9, 7, 6, 7, 5 }
			};

			Content.AnimationsSpeed = TimeSpan.FromMilliseconds(100);

			Content.SetDrawMarginWidth(Content.GetDrawMarginElements() * 0.9);

			Content.Series.Add(stepLine);


			//Content.VisualElements.Chart.AxisX[0].MinValue = 10;

			base.Content = Content;
			ContentGrid.Margin = new Thickness(20);
		}

		public new CartesianChart Content { get; }
	}
}