using System;
using System.Collections.Generic;
using System.Windows;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class TitleBarFactory : IUIFactory<TitleBarControl>
	{
		public Thickness Margin { get; }
		public Thickness Padding { get; }

		public readonly List<Func<Application, Window, TitleBarItem>> TitleBarFuncs;

		public TitleBarFactory() : this(new Thickness(0), new Thickness(0)) { }

		public TitleBarFactory(Thickness margin, Thickness padding)
		{
			Margin = margin;
			Padding = padding;

			TitleBarFuncs = new List<Func<Application, Window, TitleBarItem>>();
		}

		public TitleBarControl CreatElement(Application app, Window window, params object[] parameters)
		{
			TitleBarControl titleBarControl = new TitleBarControl
			{
				Margin = Margin,
				Padding = Padding
			};

			if (TitleBarFuncs.Count == 0)
			{
				InitialiseDefaultTabs();
			}

			foreach (Func<Application, Window, TitleBarItem> titleBarFunc in TitleBarFuncs)
			{
				TitleBarItem newItem = titleBarFunc(app, window);
				titleBarControl.AddItem(app, window, newItem);
			}

			return titleBarControl;
		}

		public virtual void InitialiseDefaultTabs()
		{
			TitleBarFuncs.Add((app, window) => new TitleBarItem("Environment", "Load", "Store", new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));

			AddSigmaFunction((app, window) => new TitleBarItem("Settings", "Toggle Dark",
				(Action) (() => window.Monitor.ColourManager.Dark = !window.Monitor.ColourManager.Dark), "Toggle Alternate",
				(Action) (() => window.Monitor.ColourManager.Alternate = !window.Monitor.ColourManager.Alternate)));
			TitleBarFuncs.Add((app, window) => new TitleBarItem("About", "Sigma"));
		}

		/// <summary>
		/// This method ensures that the passed window is a <see cref="SigmaWindow"/>.
		/// </summary>
		/// <param name="function">The function that will be executed, when a <see cref="TitleBarItem"/> is clicked. </param>
		/// <exception cref="ArgumentException">If the passed window is not a <see cref="SigmaWindow"/>, an exception will be thrown. </exception>
		protected void AddSigmaFunction(Func<Application, SigmaWindow, TitleBarItem> function)
		{
			TitleBarFuncs.Add((app, window) =>
			{
				SigmaWindow sigmaWindow = window as SigmaWindow;

				if (sigmaWindow == null) throw new ArgumentException($@"Unfortunately, the default {nameof(TitleBarFactory)} only works with a {nameof(SigmaWindow)}.", nameof(window));

				return function(app, sigmaWindow);
			});
		}
	}
}