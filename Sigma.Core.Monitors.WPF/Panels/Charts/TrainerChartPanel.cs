using System.Collections.Generic;
using System.Linq;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	/// <summary>
	/// This generic <see cref="SigmaPanel"/> (should) work with every chart from LiveCharts.
	/// Depending on the generics, it plots a given chart with given data and a given amount of parameters.
	/// These parameters are automatically fetched at a given <see cref="TimeStep"/> via a <see cref="ValueReporterHook"/>.
	/// </summary>
	/// <typeparam name="TChart">The <see cref="Chart"/> that is used.</typeparam>
	/// <typeparam name="TSeries">The <see cref="Series"/> that is used.</typeparam>
	/// <typeparam name="TData">The data the <see cref="Series"/> contains.</typeparam>
	public class TrainerChartPanel<TChart, TSeries, TData> : ChartPanel<TChart, TSeries, TData> where TChart : Chart, new() where TSeries : Series, new()
	{
		/// <summary>
		/// The trainer. 
		/// </summary>
		protected ITrainer Trainer;

		/// <summary>
		/// The hook that is attached.
		/// </summary>
		protected VisualValueReporterHook<TData> AttachedHook;

		///  <summary>
		///      Create a TrainerChartPanel with a given title.
		///      This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///      If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValue">The value that will get hooked (i.e. the value identifier of <see cref="ValueReporterHook"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public TrainerChartPanel(string title, ITrainer trainer, string hookedValue, ITimeStep timestep, object headerContent = null) : base(title, headerContent)
		{
			VisualValueReporterHook<TData> hook = new VisualValueReporterHook<TData>(this, new[] { hookedValue }, timestep);
			Init(trainer, hook);
		}

		///  <summary>
		///      Create a TrainerChartPanel with a given title.
		///      This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///      If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValues">The values that will get hooked (i.e. the value identifiers of <see cref="ValueReporterHook"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public TrainerChartPanel(string title, ITrainer trainer, ITimeStep timestep, object headerContent = null, params string[] hookedValues) : base(title, headerContent)
		{
			VisualValueReporterHook<TData> hook = new VisualValueReporterHook<TData>(this, hookedValues, timestep);
			Init(trainer, hook);
		}

		private void Init(ITrainer trainer, VisualValueReporterHook<TData> hook)
		{
			Trainer = trainer;

			AttachedHook = hook;
			Trainer.AddHook(hook);
		}

		///  <summary>
		///      Create a TrainerChartPanel with a given title.
		///      This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///      If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hook">The hook (that is responsible for getting the desired value) which will be attached to the trainer. </param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected TrainerChartPanel(string title, ITrainer trainer, VisualValueReporterHook<TData> hook, object headerContent = null) : base(title, headerContent)
		{
			Init(trainer, hook);
		}

		//TODO: not generic?
		protected class VisualValueReporterHook<THookData> : ValueReporterHook
		{
			public readonly ChartPanel<TChart, TSeries, TData> ChartPanel;

			public VisualValueReporterHook(ChartPanel<TChart, TSeries, TData> chartPanel, string valueIdentifier, ITimeStep timestep) : this(chartPanel, new[] { valueIdentifier }, timestep)
			{ }

			public VisualValueReporterHook(ChartPanel<TChart, TSeries, TData> chartPanel, string[] valueIdentifiers, ITimeStep timestep) : base(valueIdentifiers, timestep)
			{
				ChartPanel = chartPanel;
			}

			/// <summary>
			/// Report the values for a certain epoch / iteration to a passed ChartPanel. 
			/// </summary>
			/// <param name="valuesByIdentifier">The values by their identifier.</param>
			protected override void ReportValues(IDictionary<string, object> valuesByIdentifier)
			{
				//TODO: multiple values
				ChartPanel.Dispatcher.InvokeAsync(() => ChartPanel.Series.Values.Add(valuesByIdentifier.Values.First()));
			}
		}
	}
}