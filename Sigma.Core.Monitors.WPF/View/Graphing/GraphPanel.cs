using Sigma.Core.Monitors.WPF.Panels;

namespace Sigma.Core.Monitors.WPF.View.Graphing
{
	// maybe directly <NetworkView>
	public class GraphPanel : GenericPanel<GraphView>
	{
		public GraphPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			Content = new GraphView();
		}
	}
}