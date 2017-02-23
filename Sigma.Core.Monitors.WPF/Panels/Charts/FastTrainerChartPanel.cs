using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core.Training;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	public class FastTrainerChartPanel<TChart, TSeries, TData> : TrainerChartPanel<TChart, TSeries, TData> where TChart : Chart, new() where TSeries : Series, new()
	{
		///  <summary>
		///  Create a FastTrainerChartPanel with a given title.
		///  This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValue">The value that will get hooked (i.e. the value identifier of <see cref="Training.Hooks.Reporters.ValueReporterHook"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public FastTrainerChartPanel(string title, ITrainer trainer, string hookedValue, ITimeStep timestep, object headerContent = null) : base(title, trainer, hookedValue, timestep, headerContent)
		{
			Fast();
		}

		///  <summary>
		///  Create a FastTrainerChartPanel with a given title.
		///  This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hookedValues">The values that will get hooked (i.e. the value identifiers of <see cref="Training.Hooks.Reporters.ValueReporterHook"/>).</param>
		/// <param name="timestep">The <see cref="TimeStep"/> for the hook.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public FastTrainerChartPanel(string title, ITrainer trainer, ITimeStep timestep, object headerContent = null, params string[] hookedValues) : base(title, trainer, timestep, headerContent, hookedValues)
		{
			Fast();
		}

		///  <summary>
		///  Create a FastTrainerChartPanel with a given title.
		///  This <see cref="ChartPanel{T,TSeries,TData}"/> automatically receives data via a hook and adds it to the chart.
		///  If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer to attach the hook to.</param>
		/// <param name="hook">The hook (that is responsible for getting the desired value) which will be attached to the trainer. </param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected FastTrainerChartPanel(string title, ITrainer trainer, VisualValueReporterHook hook, object headerContent = null) : base(title, trainer, hook, headerContent)
		{
			Fast();
		}

		/// <summary>
		/// Set all required actions to improve the performance of the <see cref="TrainerChartPanel{TChart,TSeries,TData}"/>.
		/// </summary>
		protected void Fast()
		{
			Content.DisableAnimations = true;
			Content.Hoverable = false;
			Content.DataTooltip = null;
		}
	}
}