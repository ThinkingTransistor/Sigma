/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	/// <summary>
	/// This generic <see cref="SigmaPanel"/> (should) work with every chart from LiveCharts.
	/// Depending on the generics, it plots a given chart with given data and a given amount of parameters.
	/// These parameters are automatically fetched at a given <see cref="TimeStep"/> via a <see cref="AccumulatedValueReporter"/>.
	/// </summary>
	/// <typeparam name="TChart">The <see cref="Chart"/> that is used.</typeparam>
	/// <typeparam name="TSeries">The <see cref="Series"/> that is used.</typeparam>
	/// <typeparam name="TChartValues">The data structure that contains the points itself, may be generic of type TData.</typeparam>
	/// <typeparam name="TData">The data the <see cref="Series"/> contains.</typeparam>
	public class TrainerChartPanel<TChart, TSeries, TChartValues, TData> : ChartPanel<TChart, TSeries, TChartValues, TData> where TChart : Chart, new() where TSeries : Series, new() where TChartValues : IList<TData>, IChartValues, new()
	{
		/// <summary>
		/// The trainer to attach the hook to. 
		/// </summary>
		protected ITrainer Trainer;

		/// <summary>
		/// The hook that is attached to the Trainer (see <see cref="Trainer"/>). 
		/// </summary>
		protected IHook AttachedHook;

		///  <summary>
		///  Create a TrainerChartPanel with a given title.
		///  This <see ref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValue">The value that will get hooked (i.e. the value identifier of <see cref="AccumulatedValueReporter"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public TrainerChartPanel(string title, ITrainer trainer, string hookedValue, ITimeStep timestep, bool averageMode = false, object headerContent = null) : base(title, headerContent)
		{
			VisualAccumulatedValueReporterHook hook = new VisualAccumulatedValueReporterHook(this, new[] { hookedValue }, timestep, averageMode);
			Init(trainer, hook);
		}

		///  <summary>
		///  Create a TrainerChartPanel with a given title.
		///  This <see ref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValues">The values that will get hooked (i.e. the value identifiers of <see cref="AccumulatedValueReporter"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public TrainerChartPanel(string title, ITrainer trainer, ITimeStep timestep, bool averageMode = false, object headerContent = null, params string[] hookedValues) : base(title, headerContent)
		{
			VisualAccumulatedValueReporterHook hook = new VisualAccumulatedValueReporterHook(this, hookedValues, timestep, averageMode);
			Init(trainer, hook);
		}

		///  <summary>
		///  Create a TrainerChartPanel with a given title.
		///  This <see ref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hook">The hook (that is responsible for getting the desired value) which will be attached to the trainer. </param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected TrainerChartPanel(string title, ITrainer trainer, VisualAccumulatedValueReporterHook hook, object headerContent = null) : base(title, headerContent)
		{
			Init(trainer, hook);
		}

		/// <summary>
		/// Set the trainer and hook. Attach the hook.
		/// </summary>
		/// <param name="trainer">The trainer that will be set.</param>
		/// <param name="hook">The hook that will be applied.</param>
		protected void Init(ITrainer trainer, VisualAccumulatedValueReporterHook hook)
		{
			Trainer = trainer;
			AttachedHook = hook;
			Trainer.AddHook(hook);
			Trainer.AddGlobalHook(new LambdaHook(TimeStep.Every(1, TimeScale.Stop), (registry, resolver) => Clear()));

			// TODO: is a formatter the best solution?
			AxisX.LabelFormatter = number => (number * hook.TimeStep.Interval).ToString(CultureInfo.InvariantCulture);
			AxisX.Unit = hook.TimeStep.Interval;
		}

		/// <summary>
		/// The hook reports values to a given <see ref="ChartPanel"/>.
		/// </summary>
		protected class VisualAccumulatedValueReporterHook : AccumulatedValueReporter
		{
			/// <summary>
			/// The identifier for the parameter registry that keeps a reference to the chartpanel
			/// </summary>
			protected const string ChartPanelIdentifier = "panel";

			/// <summary>
			/// Create a new <see ref="VisualValueReportHook"/> fully prepared to report values.
			/// </summary>
			/// <param name="chartPanel">The chartpanel to which points will get added.</param>
			/// <param name="valueIdentifiers">The identifiers for the <see cref="AccumulatedValueReporter"/>; these values will get plotted.</param>
			/// <param name="timestep">The <see cref="TimeStep"/> for the hook (i.e. execution definition).</param>
			public VisualAccumulatedValueReporterHook(ChartPanel<TChart, TSeries, TChartValues, TData> chartPanel, string[] valueIdentifiers, ITimeStep timestep, bool averageMode = false) : base(valueIdentifiers, timestep, averageMode, false)
			{
				ParameterRegistry[ChartPanelIdentifier] = chartPanel;
			}

			/// <summary>
			/// Report the values for a certain epoch / iteration to a passed ChartPanel. 
			/// </summary>
			/// <param name="valuesByIdentifier">The values by their identifier.</param>
			/// <param name="reportEpochIteration">A boolean indicating whether or not to report the current epoch / iteration.</param>
			/// <param name="epoch">The current epoch.</param>
			/// <param name="iteration">The current iteration.</param>
			protected override void ReportValues(IDictionary<string, object> valuesByIdentifier, bool reportEpochIteration, int epoch, int iteration)
			{
				ChartPanel<TChart, TSeries, TChartValues, TData> chartPanel = (ChartPanel<TChart, TSeries, TChartValues, TData>)ParameterRegistry[ChartPanelIdentifier];
				chartPanel.Add((TData)valuesByIdentifier.Values.First());

				//TODO: multiple values (in same series)
				//ChartPanel.Dispatcher.InvokeAsync(() => ChartPanel.Series.Values.Add(valuesByIdentifier.Values.First()));
			}
		}
	}
}