using System.Windows.Controls;
using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.View.Parameterisation;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.Panels.Parameterisation
{
	public class ParameterPanel : GenericPanel<ParameterView>
	{
		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public ParameterPanel(string title, IParameterVisualiserManager visualiserManager, ISynchronisationHandler synchronisationHandler, object headerContent = null) : base(title, headerContent)
		{
			Content = new ParameterView(visualiserManager, synchronisationHandler);
		}

	}
}