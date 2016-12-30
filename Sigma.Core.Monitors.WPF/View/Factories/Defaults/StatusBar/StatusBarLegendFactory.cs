using System;
using System.Windows;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	public class StatusBarLegendFactory : IUIFactory<UIElement>
	{
		/// <summary>
		/// </summary>
		/// <param name="app"></param>
		/// <param name="window"></param>
		/// <param name="parameters">One <see cref="StatusBarLegendInfo" /></param>
		/// <returns></returns>
		public UIElement CreateElement(Application app, Window window, params object[] parameters)
		{
			if (parameters.Length != 1)
				throw new ArgumentException(@"Value has to be a single-value array.", nameof(parameters));

			StatusBarLegendInfo info = parameters[0] as StatusBarLegendInfo;

			if (info == null)
				throw new ArgumentException($@"Value cannot be casted to {typeof(StatusBarLegendInfo)}", nameof(parameters));

			return info.Apply(new StatusBarLegend());
		}
	}
}