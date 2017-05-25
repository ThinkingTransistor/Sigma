/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using log4net;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using Sigma.Core.Architecture;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;
using Sigma.Core.Persistence;
using Sigma.Core.Training;
using Sigma.Core.Utils;
using Registry = Sigma.Core.Utils.Registry;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class TitleBarFactory : IUIFactory<TitleBarControl>
	{
		public const string RegistryIdentifier = "titlebar_registry";

		public const string AboutFactoryIdentifier = "about_factory";

		public readonly List<Func<Application, Window, TitleBarItem>> TitleBarFuncs;

		private ILog _log = LogManager.GetLogger(typeof(TitleBarFactory));

		/// <summary>
		/// The <see cref="IRegistry"/> where all required factories are contained. 
		/// </summary>
		public IRegistry Registry { get; set; }

		public TitleBarFactory(IRegistry parentRegistry) : this(parentRegistry, new Thickness(0), new Thickness(0)) { }

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

			//TODO: hack
			if (TitleBarFuncs.Count == 0)
			{
				InitialiseDefaultItems();
			}

			foreach (Func<Application, Window, TitleBarItem> titleBarFunc in TitleBarFuncs)
			{
				TitleBarItem newItem = titleBarFunc(app, window);
				titleBarControl.AddItem(app, window, newItem);
			}

			return titleBarControl;
		}

		/// <summary>
		/// The default item generation that will be called if no other title bar item is specified (i.e. <see cref="TitleBarFuncs"/> is empty).
		/// </summary>
		public virtual void InitialiseDefaultItems()
		{
			_log.Info("Creating default title bar items because no others have been specified.");

			AddSigmaFunction((app, window) =>
				new TitleBarItem(Properties.Resources.ButtonEnvironment,
					Properties.Resources.MenuButtonLoad, (Action) (() =>
					{
						SigmaEnvironment sigma = window.Monitor.Sigma;
						ITrainer activeTrainer = sigma.RunningOperatorsByTrainer.Keys.FirstOrDefault();

						if (activeTrainer != null)
						{
							OpenFileDialog fileDialog = new OpenFileDialog();
							fileDialog.Title = "Open Network";
							fileDialog.Multiselect = false;
							fileDialog.Filter = "Sigma Network Files (*.sgnet)|*.sgnet";
							fileDialog.InitialDirectory = new FileInfo(SigmaEnvironment.Globals.Get<string>("storage_path")).FullName;

							if (fileDialog.ShowDialog() == true)
							{
								try
								{
									INetwork network = Serialisation.Read<INetwork>(Target.FileByPath(fileDialog.FileName), Serialisers.BinarySerialiser, false);

									if (!Network.AreNetworkExternalsCompatible(network, activeTrainer.Network))
									{
										throw new InvalidOperationException($"Unable to switch to network \"{network.Name}\" with incompatible internals (from {fileDialog.FileName}).");
									}

									activeTrainer.Reset();

									bool forceInitialisationBefore = activeTrainer.ForceInitialisation;
									activeTrainer.ForceInitialisation = false;
									activeTrainer.Network = network;
									activeTrainer.Initialise(activeTrainer.Operator.Handler);

									activeTrainer.ForceInitialisation = forceInitialisationBefore;

									Task.Factory.StartNew(() => window.SnackbarMessageQueue.Enqueue($"Switched network \"{network.Name}\", reset training (now using \"{fileDialog.FileName}\")", "Got it", null));
								}
								catch (Exception e)
								{
									_log.Error($"Error while switching to network \"{fileDialog.FileName}\": {e.GetType()} ({e.Message})", e);
								}
							}
						}
						else
						{
							_log.Warn($"Unable to load new network because no trainer is active.");
						}
					}),
					Properties.Resources.MenuButtonSave, new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));

#if DEBUG
			AddSigmaFunction((app, window) => new TitleBarItem(Properties.Resources.ButtonDebug,
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
							// ignore
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
						  for (int i = 1; i <= 1; i++)
						  {
							  log.Debug($"Flood {i}: debug");
							  log.Info($"Flood {i}: info");
							  log.Warn($"Flood {i}: warn");
							  log.Error($"Flood {i}: error");
							  log.Fatal($"Flood {i}: fatal");
						  }
					  }).Start();
				  }), "Print Hierarchy", (Action) (() =>
				{

					SigmaWindow root = window;
					while (!root.IsRoot) root = root.ParentWindow;

					PrintWindow(root);
				})
			));
#endif

			AddSigmaFunction((app, window) => new TitleBarItem(Properties.Resources.ButtonSettings, new TitleBarItem(Properties.Resources.MenuButtonStyle, Properties.Resources.MenuButtonToggleDark,
				(Action) (() => window.Monitor.ColourManager.Dark = !window.Monitor.ColourManager.Dark),
				Properties.Resources.MenuButtonToggleAlternate, (Action) (() => window.Monitor.ColourManager.Alternate = !window.Monitor.ColourManager.Alternate)/*,
				Properties.Resources.MenuButtonLanguage, (Action< Application, Window, TitleBarItem>) ((application, genericWindow, item) =>
				{
					WPFMonitor monitor = window.Monitor;
					monitor.UiCultureInfo = CultureInfo.GetCultureInfo("de-DE");
					monitor.Reload();
				})*/)));

			AddSigmaFunction((app, window) =>
			{
				IUIFactory<UIElement> aboutFactory = (IUIFactory<UIElement>) Registry.TryGetValue(AboutFactoryIdentifier, () => new AboutFactory(window.DialogHost));
				object aboutContent = aboutFactory.CreateElement(app, window);

				TitleBarItem about = new TitleBarItem(Properties.Resources.ButtonHelp, new TitleBarItem(Properties.Resources.MenuButtonAbout, (Action) (async () =>
				{
					window.DialogHost.IsOpen = false;
					await DialogHost.Show(aboutContent, window.DialogHostIdentifier);
				})));

				return about;
			});
		}

#if DEBUG
		private static void PrintWindow(SigmaWindow window)
		{
			Debug.WriteLine("window: " + window + " parent: " + window.ParentWindow + $" children:{window.ChildrenReadOnly.Count}\n================");
			foreach (SigmaWindow child in window.ChildrenReadOnly)
			{
				PrintWindow(child);
			}

			Debug.WriteLine("================");
		}

#endif

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
