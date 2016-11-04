using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Tabs
{
	/// <summary>
	/// This class is basically only a wrapper 
	/// </summary>
	internal class TabUI : UIWrapper<TabItem>
	{
		public TabUI(string header) : base()
		{
			content.Header = header;
		}
	}
}
