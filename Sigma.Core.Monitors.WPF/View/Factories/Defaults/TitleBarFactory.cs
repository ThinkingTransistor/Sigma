/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using log4net;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class TitleBarFactory : IUIFactory<TitleBarControl>
	{
		public const string RegistryIdentifier = "titlebar_registry";

		public const string AboutFactoryIdentifier = "about_factory";

		public readonly List<Func<Application, Window, TitleBarItem>> TitleBarFuncs;

		/// <summary>
		/// The <see cref="IRegistry"/> where all required factories are contained. 
		/// </summary>
		public IRegistry Registry { get; set; }

		public TitleBarFactory(IRegistry parentRegistry) : this(parentRegistry, new Thickness(0), new Thickness(0))
		{
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="parentRegistry">The parent registry. If it not already contains a registry with the key <see cref="RegistryIdentifier"/>,
		/// a new <see cref="IRegistry"/> will be created and added to the given registry. </param>		/// <param name="margin"></param>
		/// <param name="padding"></param>
		public TitleBarFactory(IRegistry parentRegistry, Thickness margin, Thickness padding)
		{
			if (parentRegistry == null || !parentRegistry.ContainsKey(RegistryIdentifier))
			{
				Registry = new Registry(parentRegistry);
				parentRegistry?.Add(RegistryIdentifier, Registry);
			}
			else
			{
				Registry = (IRegistry) parentRegistry[RegistryIdentifier];
			}

			Margin = margin;
			Padding = padding;

			TitleBarFuncs = new List<Func<Application, Window, TitleBarItem>>();
		}

		public Thickness Margin { get; }
		public Thickness Padding { get; }

		public TitleBarControl CreateElement(Application app, Window window, params object[] parameters)
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
			TitleBarFuncs.Add(
				(app, window) =>
					new TitleBarItem("Environment", "Load", "Store",
						new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));

#if DEBUG
			AddSigmaFunction((app, window) => new TitleBarItem("Debug", "Download mnist", (Action) (() =>
				{
					BaseIterator iterator = window.Monitor.Registry["iterator"] as BaseIterator;
					IComputationHandler handler = window.Monitor.Registry["handler"] as IComputationHandler;
					SigmaEnvironment environment = window.Monitor.Registry["environment"] as SigmaEnvironment;

					new Thread(() => iterator?.Yield(handler, environment).First()).Start();
				}), "10 second long task", (Action) (() =>
				{
					new Thread(() =>
					{
						ITaskObserver task = null;
						try
						{
							task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Download, "http://somedataset.com");

							for (float i = 0; i <= 1; i += 0.0010f)
							{
								task.Progress = i;

								Thread.Sleep(10);
							}
						}
						catch (Exception)
						{
						}
						finally
						{
							SigmaEnvironment.TaskManager.EndTask(task);
						}

						ITaskObserver task2 = null;
						try
						{
							task2 = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare, "Preparing");
							Thread.Sleep(1000);
						}
						catch (Exception)
						{
						}
						finally
						{
							SigmaEnvironment.TaskManager.EndTask(task2);
						}
					}).Start();
				}),
				"5 second indeterminate task",
				(Action) (() =>
				{
					new Thread(() =>
					{
						ITaskObserver task = null;
						try
						{
							task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Preprocess, "Indeterminate task");
							Thread.Sleep(5000);
						}
						catch (Exception)
						{
						}
						finally
						{
							SigmaEnvironment.TaskManager.EndTask(task);
						}
					}).Start();
				}), "Flood", (Action) (() =>
				  {
					  ILog log = LogManager.GetLogger(typeof(TitleBarFactory));
					  new Thread(() =>
					  {
						  for (int i = 1; i <= 100; i++)
						  {
							  log.Debug($"Flood {i}");
							  log.Info($"Flood {i}");
							  log.Warn($"Flood {i}");
							  log.Error($"Flood {i}");
							  log.Fatal($"Flood {i}");
						  }
					  }).Start();
				  })
			));
#endif

			AddSigmaFunction((app, window) => new TitleBarItem("Settings", "Toggle Dark",
				(Action) (() => window.Monitor.ColourManager.Dark = !window.Monitor.ColourManager.Dark), "Toggle Alternate",
				(Action) (() => window.Monitor.ColourManager.Alternate = !window.Monitor.ColourManager.Alternate)));

			AddSigmaFunction((app, window) =>
			{
				IUIFactory<UIElement> aboutFactory = (IUIFactory<UIElement>) Registry.TryGetValue(AboutFactoryIdentifier, () => new AboutFactory(window.DialogHost));
				object aboutContent = aboutFactory.CreateElement(app, window);

				TitleBarItem about = new TitleBarItem(new MenuItem { Header = "About" }, (Action) (async () =>
				{
					window.DialogHost.IsOpen = false;
					await DialogHost.Show(aboutContent, window.DialogHostIdentifier);
				}));

				return about;
			});
		}

		/// <summary>
		///     This method ensures that the passed window is a <see cref="SigmaWindow" />.
		/// </summary>
		/// <param name="function">The function that will be executed, when a <see cref="TitleBarItem" /> is clicked. </param>
		/// <exception cref="ArgumentException">
		///     If the passed window is not a <see cref="SigmaWindow" />, an exception will be
		///     thrown.
		/// </exception>
		protected void AddSigmaFunction(Func<Application, SigmaWindow, TitleBarItem> function)
		{
			TitleBarFuncs.Add((app, window) =>
			{
				SigmaWindow sigmaWindow = window as SigmaWindow;

				if (sigmaWindow == null)
				{
					throw new ArgumentException(
						$@"Unfortunately, the default {nameof(TitleBarFactory)} only works with a {nameof(SigmaWindow)}.", nameof(window));
				}

				return function(app, sigmaWindow);
			});
		}
	}
}
