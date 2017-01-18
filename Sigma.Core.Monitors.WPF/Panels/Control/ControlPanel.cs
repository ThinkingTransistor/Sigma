/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;

namespace Sigma.Core.Monitors.WPF.Panels.Control
{
	public class ControlPanel : SigmaPanel
	{
		public new StackPanel Content { get; }

		public ControlPanel(string title, object content = null) : base(title, content)
		{
			Content = new StackPanel
			{
				Orientation = Orientation.Vertical,
				HorizontalAlignment = HorizontalAlignment.Center,
				Margin = new Thickness(0, 20, 0, 0)
			};

			Content.Children.Add(new SigmaPlaybackControl());

			base.Content = Content;
		}
	}
}
