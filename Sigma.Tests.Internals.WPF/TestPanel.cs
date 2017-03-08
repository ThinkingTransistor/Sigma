using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.Panels;

namespace Sigma.Tests.Internals.WPF
{
	public class TestPanel : GenericPanel<StackPanel>
	{
		public TestPanel(string name, object content = null) : base(name, content)
		{
			Content = new StackPanel { Orientation = Orientation.Vertical };
		}
	} 
}