/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class TickChartValues<T> : ChartValues<T>, ICollection<T>, IDisposable
	{
		public int Ticks { get; set; } = 200;

		protected Thread Thread { get; }

		protected bool Running { get; set; } = true;

		protected int LastTicks { get; private set; }

		protected ConcurrentBag<T> Values { get; }

		private readonly IList<T> _cacheList;

		void ICollection<T>.Add(T item)
		{
			Values.Add(item);
		}

		public TickChartValues()
		{
			Values = new ConcurrentBag<T>();
			_cacheList = new List<T>(20);

			Thread = new Thread(() =>
			{
				while (Running)
				{
					int currentTicks = Environment.TickCount;
					if (currentTicks - LastTicks >= Ticks)
					{
						Tick();
						LastTicks = currentTicks;
						Thread.Sleep(Ticks);
					}
				}
			});

			Thread.Start();
		}
		public TickChartValues(int ticks) : this()
		{
			Ticks = ticks;
		}

		protected virtual void Tick()
		{
			while (!Values.IsEmpty)
			{
				T item;
				Values.TryTake(out item);
				_cacheList.Add(item);
			}

			AddRange(_cacheList);

			_cacheList.Clear();
		}

		public void Stop()
		{
			Running = false;
		}

		/// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
		public void Dispose()
		{
			Stop();
		}
	}
	/// <summary>
	/// This <see cref="SigmaPanel"/> allows the illustration of various different chart types supported by LiveCharts.
	/// 
	/// See <a href="https://lvcharts.net">LiveCharts</a> for more information.
	/// </summary>
	/// <typeparam name="TChart">The type of the chart that is being used.</typeparam>
	/// <typeparam name="TSeries">The type of series that is used to illustrate the points.</typeparam>
	/// <typeparam name="TData">The data that will be displayed in the chart. See <a href="https://lvcharts.net/App/examples/v1/wpf/Types%20and%20Configuration">Types and Configuration</a>.</typeparam>
	public class ChartPanel<TChart, TSeries, TChartValues, TData> : GenericPanel<TChart> where TChart : Chart, new() where TSeries : Series, new() where TChartValues : IChartValues, ICollection<TData>, new()
	{
		/// <summary>
		/// The <see cref="SeriesCollection"/> containing the <see cref="Series"/> (of the type <see ref="TSeries"/>). 
		/// </summary>
		public SeriesCollection SeriesCollection { get; }

		/// <summary>
		/// A list of <see cref="Series"/>; they contain the actual data.
		/// </summary>
		public List<TSeries> Series { get; }

		/// <summary>
		/// The values (i.e. collection of values) that are displayed in the chart.
		/// </summary>
		public List<TChartValues> ChartValues { get; }

		/// <summary>
		/// A reference to the x-axis.
		/// </summary>
		public Axis AxisX { get; }
		/// <summary>
		/// A reference to the y-axis.
		/// </summary>
		public Axis AxisY { get; }

		/// <summary>
		/// The maximum of points visible at once (others get removed). This number is only considered when calling Add and AddRange.
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
				//ScrollMode = ScrollMode.XY
			};

			//TODO: make style and don't do that in code!!

			//using a gradient brush.
			//LinearGradientBrush gradientBrush = new LinearGradientBrush
			//{
			//	StartPoint = new Point(0, 0),
			//	EndPoint = new Point(0, 1)
			//};

			//gradientBrush.GradientStops.Add(new GradientStop(Color.FromRgb(33, 148, 241), 0));
			//gradientBrush.GradientStops.Add(new GradientStop(Colors.Transparent, 1));

			Series = new List<TSeries>();
			ChartValues = new List<TChartValues>();
			SeriesCollection = new SeriesCollection();

			AddSeries(new TSeries());

			Content.Series = SeriesCollection;

			AxisY = new Axis();
			AxisX = new Axis();

			Content.AxisX.Add(AxisX);
			Content.AxisY.Add(AxisY);
		}

		#region PointManagement

		#region SinglePoint

		/// <summary>
		/// Add a data entry to the first series. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		public void Add(TData data)
		{
			Add(data, 0);
		}

		/// <summary>
		/// Add a data entry to a given series. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="series">The series the data will be added to.</param>
		public void Add(TData data, TSeries series)
		{
			Add(data, Series.IndexOf(series));
		}

		/// <summary>
		/// Add a data entry to a given series (series is passed via index). Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="seriesIndex">The series index the data will be added to. (<see cref="Series"/>[index])</param>
		public void Add(TData data, int seriesIndex)
		{
			Add(data, ChartValues[seriesIndex]);
		}

		/// <summary>
		/// Add a data entry to a given <see cref="ChartValues"/>. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="values">The chart values the data will be added to.</param>
		public void Add(TData data, TChartValues values)
		{
			values.Add(data);

			KeepValuesInRange(values);
		}

		#endregion SinglePoint

		#region PointRange

		/// <summary>
		/// Add a range of data entries to the first series. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		public void AddRange(IEnumerable<TData> data)
		{
			AddRange(data, 0);
		}

		/// <summary>
		/// Add a range of data entries to a given series. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="series">The series the data will be added to.</param>
		public void AddRange(IEnumerable<TData> data, TSeries series)
		{
			AddRange(data, Series.IndexOf(series));
		}

		/// <summary>
		/// Add a range of data entries to a given series (series is passed via index). Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="seriesIndex">The series index the data will be added to. (<see cref="Series"/>[index])</param>
		public void AddRange(IEnumerable<TData> data, int seriesIndex)
		{
			AddRange(data, ChartValues[seriesIndex]);
		}

		/// <summary>
		/// Add a range of data entries to a given <see cref="ChartValues"/>. Additionally, this method maintains a maximum of <see cref="MaxPoints"/>.
		/// </summary>
		/// <param name="data">The data that will be added.</param>
		/// <param name="values">The chart values the data will be added to.</param>
		public void AddRange(IEnumerable<TData> data, TChartValues values)
		{
			values.AddRange(data.Cast<object>());

			KeepValuesInRange(values);
		}

		#endregion PointRange

		/// <summary>
		/// Maintain (i.e. eventually remove points) the point list for a given <see cref="MaxPoints"/> and chart values.
		/// </summary>
		/// <param name="chartValues">The chart values that will be maintained.</param>
		private void KeepValuesInRange(ICollection<TData> chartValues)
		{
			if (MaxPoints > 0)
			{
				IEnumerator<TData> enumerator = chartValues.AsEnumerable().GetEnumerator();
				while (chartValues.Count > MaxPoints)
				{
					chartValues.Remove(enumerator.Current);
					enumerator.MoveNext();
				}

				enumerator.Dispose();
			}
		}

		/// <summary>
		/// Clear all <see cref="ChartValues"/> to be able to "restart" the graph.
		/// </summary>
		public void Clear()
		{
			foreach (TChartValues chartValues in ChartValues)
			{
				((ICollection<TData>) chartValues).Clear();
			}
		}

		#endregion PointManagement

		#region SeriesManagement

		/// <summary>
		/// Adds a single series.
		/// </summary>
		/// <returns>The newly created series.</returns>
		public TSeries AddSeries()
		{
			TSeries series = new TSeries();
			AddSeries(series);

			return series;
		}

		/// <summary>
		/// Adds a given series.
		/// </summary>
		/// <param name="series">The series that will be added.</param>
		public void AddSeries(TSeries series)
		{
			TChartValues chartValues = new TChartValues();
			ChartValues.Add(chartValues);

			series.Values = chartValues;
			//series.Fill = Brushes.Transparent;

			Series.Add(series);
			SeriesCollection.Add(series);
		}

		/// <summary>
		/// Adds a range of series specified by the amount.
		/// </summary>
		/// <param name="amount">The amount of series to add.</param>
		/// <returns>The newly created series.</returns>
		public IEnumerable<TSeries> AddSeriesRange(int amount)
		{
			TSeries[] series = new TSeries[amount];

			for (int i = 0; i < series.Length; i++)
			{
				series[0] = new TSeries();
			}

			AddSeriesRange(series);

			return series;
		}

		/// <summary>
		/// Adds a range of series. Fastest if given an <see ref="TSeries"/>[].
		/// </summary>
		/// <param name="series">The series that will be added.</param>
		public void AddSeriesRange(IEnumerable<TSeries> series)
		{
			TSeries[] collection = series as TSeries[] ?? series.ToArray();

			Series.AddRange(collection);
			SeriesCollection.AddRange(collection);

			TChartValues[] chartValues = new TChartValues[collection.Length];
			for (int i = 0; i < chartValues.Length; i++)
			{
				chartValues[i] = new TChartValues();
			}

			ChartValues.AddRange(chartValues);
		}

		#endregion

		~ChartPanel()
		{
			foreach (TChartValues chartValues in ChartValues)
			{
				IDisposable disposable = chartValues as IDisposable;
				disposable?.Dispose();
			}
		}
	}
}