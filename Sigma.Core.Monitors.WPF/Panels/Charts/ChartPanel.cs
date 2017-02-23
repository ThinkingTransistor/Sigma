using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	/// <summary>
	/// This <see cref="SigmaPanel"/> allows the illustration of various different chart types supported by LiveCharts.
	/// 
	/// See <a href="https://lvcharts.net">LiveCharts</a> for more information.
	/// </summary>
	/// <typeparam name="TChart">The type of the chart that is being used.</typeparam>
	/// <typeparam name="TSeries">The type of series that is used to illustrate the points.</typeparam>
	/// <typeparam name="TData">The data that will be displayed in the chart. See <a href="https://lvcharts.net/App/examples/v1/wpf/Types%20and%20Configuration">Types and Configuration</a>.</typeparam>
	public class ChartPanel<TChart, TSeries, TData> : GenericPanel<TChart> where TChart : Chart, new() where TSeries : Series, new()
	{
		/// <summary>
		/// The <see cref="SeriesCollection"/> containing the <see cref="Series"/> (of the type <see cref="TSeries"/>). 
		/// </summary>
		public SeriesCollection SeriesCollection { get; }

		/// <summary>
		/// The <see cref="Series"/> that contains the actual data.
		/// </summary>
		public TSeries Series { get; }

		/// <summary>
		/// The values (i.e. collection of values) that are displayed in the chart.
		/// </summary>
		public ChartValues<TData> ChartValues { get; }

		/// <summary>
		/// A reference to the x-axis.
		/// </summary>
		public Axis AxisX { get; }
		/// <summary>
		/// A reference to the y-axis.
		/// </summary>
		public Axis AxisY { get; }

		/// <summary>
		/// The maximum of points visible at once (others get removed). This number is only considered when calling <see cref="Add"/> and <see cref="AddRange"/>.
		/// If negative (or zero), all points are displayed.
		/// </summary>
		public int MaxPoints { get; set; } = -1;

		/// <summary>
		/// Create a ChartPanel with a given title.
		/// If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
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

			//TODO: make style and don't do that in code!!

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

		public void Add(TData data)
		{
			ChartValues.Add(data);

			KeepValuesInRange();
		}

		public void AddRange(IEnumerable<TData> data)
		{
			ChartValues.AddRange(data);

			KeepValuesInRange();
		}

		private void KeepValuesInRange()
		{
			if (MaxPoints > 0)
			{
				IEnumerator<TData> enumerator = ChartValues.AsEnumerable().GetEnumerator();
				while (ChartValues.Count > MaxPoints)
				{
					ChartValues.Remove(enumerator.Current);
					enumerator.MoveNext();
				}

				enumerator.Dispose();
			}
		}

		public void Clear()
		{
			ChartValues.Clear();
		}
	}
}