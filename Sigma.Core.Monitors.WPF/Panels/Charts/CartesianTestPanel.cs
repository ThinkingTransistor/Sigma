using System.Collections.Generic;
using LiveCharts.Wpf;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	//TODO: remove
	public class CartesianTestPanel : ChartPanel<CartesianChart, LineSeries, double>
	{
		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer"></param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public CartesianTestPanel(string title, ITrainer trainer, object headerContent = null) : base(title, headerContent)
		{
			trainer.AddHook(new ChartValidationAccuracyReport(this, "validation", TimeStep.Every(1, TimeScale.Epoch), tops: 1));
		}

		protected class ChartValidationAccuracyReport : ValidationAccuracyReporter
		{
			private readonly ChartPanel<CartesianChart, LineSeries, double> _panel;

			/// <summary>
			/// Create a hook with a certain time step and a set of required global registry entries. 
			/// </summary>
			/// <param name="validationIteratorName">The name of the validation data iterator to use (as in the trainer).</param>
			/// <param name="timestep">The time step.</param>
			/// <param name="tops">The tops that will get reported.</param>
			public ChartValidationAccuracyReport(ChartPanel<CartesianChart, LineSeries, double> panel, string validationIteratorName, ITimeStep timestep, params int[] tops) : base(validationIteratorName, timestep, tops)
			{
				_panel = panel;
			}

			/// <summary>
			/// Execute the report for every given top. 
			/// </summary>
			/// <param name="data">The mapping between the tops specified in the constructor and the score of the top.</param>
			protected override void Report(IDictionary<int, double> data)
			{
				base.Report(data);
				_panel.Dispatcher.InvokeAsync(() => _panel.ChartValues.Add(data[1]));
			}
		}
	}
}