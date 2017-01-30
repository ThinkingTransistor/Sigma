/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using Sigma.Core.Monitors.WPF.Panels.Charts.Definitions;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class LineChartPanel : GenericPanel<CartesianChart>
	{
		protected readonly StepLineSeries StepLine;

		public LineChartPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new CartesianChart { Zoom = ZoomingOptions.Xy };


			StepLine = new StepLineSeries
			{
				Values = new ChartValues<double>()
			};

			// TODO: set the AnimationSpeed in style
			Content.AnimationsSpeed = TimeSpan.FromMilliseconds(100);

			//Content.SetDrawMarginWidth(Content.GetDrawMarginElements() * 0.9);

			Content.Series.Add(StepLine);

			//Content.VisualElements.Chart.AxisX[0].MinValue = 10;

			ContentGrid.Margin = new Thickness(5);
		}

		/// <summary>
		/// Add a give value to the LineChart and update the view.
		/// </summary>
		/// <param name="value">The value that will be added.</param>
		public virtual void Add(double value)
		{
			StepLine.Values.Add(value);

		}

		/// <summary>
		/// Add a range of given values to the LineChart and update the view once.
		/// </summary>
		/// <param name="values">The range of values that will be added.</param>
		//public virtual void AddRange(IEnumerable<double> values)
		//{
		//	StepLine.Values.AddRange(values);
		//	//TODO: update
		//}
	}
}