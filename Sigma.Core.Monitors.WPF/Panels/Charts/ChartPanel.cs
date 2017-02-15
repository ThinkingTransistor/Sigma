using System.Windows;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	//TODO: documentation
	public class ChartPanel<TChart, TSeries, TData> : GenericPanel<TChart> where TChart : Chart, new() where TSeries : Series, new()
	{
		public SeriesCollection SeriesCollection { get; }

		public TSeries Series { get; }

		public ChartValues<TData> ChartValues { get; }

		public Axis AxisX { get; }
		public Axis AxisY { get; }

		/// <summary>
		///     Create a ChartPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public ChartPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new TChart
			{
				Zoom = ZoomingOptions.Xy,
				ScrollMode = ScrollMode.XY
			};

			//using a gradient brush.
			LinearGradientBrush gradientBrush = new LinearGradientBrush
			{
				StartPoint = new Point(0, 0),
				EndPoint = new Point(0, 1)
			};
			gradientBrush.GradientStops.Add(new GradientStop(Color.FromRgb(33, 148, 241), 0));
			gradientBrush.GradientStops.Add(new GradientStop(Colors.Transparent, 1));

			ChartValues = new ChartValues<TData>();

			Series = new TSeries
			{
				Values = ChartValues
				//Fill = gradientBrush,
				//StrokeThickness = 1,
				//PointGeometrySize = 0
			};

			SeriesCollection = new SeriesCollection
			{
				Series
			};

			Content.Series = SeriesCollection;

			AxisY = new Axis();
			AxisX = new Axis();

			Content.AxisX.Add(AxisX);
			Content.AxisY.Add(AxisY);
		}
	}
}