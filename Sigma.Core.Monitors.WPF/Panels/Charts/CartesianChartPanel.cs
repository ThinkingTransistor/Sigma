using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using Sigma.Core.Monitors.WPF.Panels.Charts.Definitions;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class CartesianChartPanel : GenericPanel<CartesianChart>
	{
		public SeriesCollection SeriesCollection { get; set; }

		public LineSeries LineSeries { get; set; }

		public ChartValues<double> ChartValues { get; set; }


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
			Content.Zoom = ZoomingOptions.Xy;
			Content.ScrollMode = ScrollMode.XY; 

			//using a gradient brush.
			LinearGradientBrush gradientBrush = new LinearGradientBrush
			{
				StartPoint = new Point(0, 0),
				EndPoint = new Point(0, 1)
			};
			gradientBrush.GradientStops.Add(new GradientStop(Color.FromRgb(33, 148, 241), 0));
			gradientBrush.GradientStops.Add(new GradientStop(Colors.Transparent, 1));

			ChartValues = new ChartValues<double>();

			LineSeries = new LineSeries
			{
				Values = ChartValues,
				Fill = gradientBrush,
				StrokeThickness = 1,
				PointGeometrySize = 0
			};

			SeriesCollection = new SeriesCollection
			{
				LineSeries
			};

			Content.Series = SeriesCollection;

			Content.AxisY.Add(new Axis {MinValue = 0, MaxValue = 10});
		}
	}
}