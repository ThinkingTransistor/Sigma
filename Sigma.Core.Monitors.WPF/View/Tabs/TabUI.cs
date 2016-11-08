/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.Model.UI;

namespace Sigma.Core.Monitors.WPF.View.Tabs
{
	/// <summary>
	/// This class is basically only a wrapper for the tabs to be handled easily (and future proof)
	/// </summary>
	internal class TabUI : UIWrapper<TabItem>
	{
		public TabUI(string header) : base()
		{
			content.Header = header;
		}
	}
}
