/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Tabs
{
	/// <summary>
	/// This class is basically only a wrapper for the tabs to be handled easily (and future proof)
	/// </summary>
	internal class TabUI : UIWrapper<TabItem>
	{
		public GridSize @GridSize { get; set; }

		public TabUI(string header, GridSize gridsize) : base()
		{
			content.Header = header;
			GridSize = gridsize;
		}
	}
}
