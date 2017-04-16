using Sigma.Core.Monitors.WPF.Panels;

namespace Sigma.Core.Monitors.WPF.View.Graphing
{
	// maybe directly <NetworkView>
	public class ArchitectureGraphPanel : GenericPanel<ArchitectureGraphView>
	{
		public ArchitectureGraphPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new ArchitectureGraphView();
		}
	}
}