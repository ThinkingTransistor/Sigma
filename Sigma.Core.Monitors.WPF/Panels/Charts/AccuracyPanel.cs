/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using LiveCharts;
using LiveCharts.Wpf;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	/// <summary>
	/// This panel is specifically designed to display the accuracy.
	/// The advantage when using this panel is that multiple accuracies can be
	/// added and displayed automatically.
	/// </summary>
	public class AccuracyPanel : ChartPanel<CartesianChart, LineSeries, ChartValues<double>, double>
	{

		/// <summary>
		/// Create an AccuracyPanel with a given title. It displays the best accuracy per epoch.
		/// If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer"></param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public AccuracyPanel(string title, ITrainer trainer, object headerContent = null) : this(title, trainer, headerContent, tops: 1) { }

		/// <summary>
		/// Create an AccuracyPanel with a given title. It displays given accuracies per epoch.
		/// If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer"></param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		/// <param name="tops"></param>
		public AccuracyPanel(string title, ITrainer trainer, object headerContent = null, params int[] tops) : this(title, trainer, TimeStep.Every(1, TimeScale.Epoch), headerContent, tops)
		{
		}

		/// <summary>
		/// Create an AccuracyPanel with a given title. It displays given accuracies per epoch.
		/// If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer"></param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		/// <param name="tops"></param>
		public AccuracyPanel(string title, ITrainer trainer, ITimeStep timeStep, object headerContent = null, params int[] tops) : base(title, headerContent)
		{
			if (timeStep == null) throw new ArgumentNullException(nameof(timeStep));

			// skip the first since its automatically generated
			for (int i = 1; i < tops.Length; i++)
			{
				AddSeries(new LineSeries());
			}

			trainer.AddHook(new ChartValidationAccuracyReport(this, "validation", timeStep, tops));
			trainer.AddGlobalHook(new LambdaHook(TimeStep.Every(1, TimeScale.Stop), (registry, resolver) => Clear()));

			AxisY.MinValue = 0;
			AxisY.MaxValue = 100;
		}

		/// <summary>
		/// This hook is specifically designed to report values to the <see cref="AccuracyPanel"/>.
		/// It reports given top accuracies ranging from 0-100%. 
		/// </summary>
		protected class ChartValidationAccuracyReport : ValidationAccuracyReporter
		{
			private const string PanelIdentifier = "Panel";

			/// <summary>
			/// Create a hook with a certain time step and a set of required global registry entries. 
			/// </summary>
			/// <param name="panel">The panel this reporter belongs to.</param>
			/// <param name="validationIteratorName">The name of the validation data iterator to use (as in the trainer).</param>
			/// <param name="timestep">The time step.</param>
			/// <param name="tops">The tops that will get reported.</param>
			public ChartValidationAccuracyReport(ChartPanel<CartesianChart, LineSeries, ChartValues<double>, double> panel, string validationIteratorName, ITimeStep timestep, params int[] tops) : base(validationIteratorName, timestep, tops)
			{
				ParameterRegistry[PanelIdentifier] = panel;
			}

			/// <summary>
			/// Execute the report for every given top. 
			/// </summary>
			/// <param name="data">The mapping between the tops specified in the constructor and the score of the top.</param>
			protected override void Report(IDictionary<int, double> data)
			{
				base.Report(data);
				ChartPanel<CartesianChart, LineSeries, ChartValues<double>, double> panel = (ChartPanel<CartesianChart, LineSeries, ChartValues<double>, double>) ParameterRegistry[PanelIdentifier];

				int i = 0;
				foreach (KeyValuePair<int, double> top in data)
				{
					panel.Add(top.Value * 100, i++);
				}
			}
		}
	}
}