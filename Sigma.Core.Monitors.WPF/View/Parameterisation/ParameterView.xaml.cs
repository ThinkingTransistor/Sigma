/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using log4net;
using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.Annotations;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// Interaction logic for ParameterView.xaml
	/// </summary>
	public partial class ParameterView
	{
		private readonly ILog _log = LogManager.GetLogger(typeof(ParameterView));

		/// <summary>
		/// The manager that keeps track of all visualisers.
		/// </summary>
		protected readonly IParameterVisualiserManager Manager;

		/// <summary>
		/// The handler that is used for transactions with the monitor.
		/// </summary>
		protected readonly ISynchronisationHandler SynchronisationHandler;

		/// <summary>
		/// The current rowpost to which elements will be added.
		/// </summary>
		protected int RowPos;

		/// <summary>
		/// Generate a new parameter view with a given manager and synchronisation handler.
		/// </summary>
		/// <param name="manager">A manager of all active visualisers.</param>
		/// <param name="synchronisationHandler">A handler for parameter syncing.</param>
		public ParameterView(IParameterVisualiserManager manager, ISynchronisationHandler synchronisationHandler)
		{
			Manager = manager;
			SynchronisationHandler = synchronisationHandler;
			InitializeComponent();
		}

		/// <summary>
		/// Generate a new parameter view with the manager and synchronisation handler assigned in the environment / window.
		/// <param name="environment">The currently active environment.</param>
		/// <param name="window">The currently active window (i.e. root window).</param>
		/// </summary>
		public ParameterView(SigmaEnvironment environment, SigmaWindow window) : this(window.ParameterVisualiser, environment.SynchronisationHandler)
		{
		}

		/// <summary>
		/// Display a given type stored in given registry (with the given key) next to a label with a given text.
		/// </summary>
		/// <param name="name">The text the label will contain.</param>
		/// <param name="type">The type that will be displayed.</param>
		/// <param name="registry">The registry which contains the value that should be displayed.</param>
		/// <param name="key">The key to access the exact value required.</param>
		public void Add(string name, Type type, IRegistry registry, string key)
		{
			Add(new Label {Content = name}, type, registry, key);
		}

		public void Add(UIElement name, Type visualiserType, IRegistry registry, string key)
		{
			UIElement displayer = (UIElement) Activator.CreateInstance(Manager.VisualiserType(visualiserType));
			IParameterVisualiser visualiser = displayer as IParameterVisualiser;

			if (visualiser == null)
			{
				_log.Warn($"{Manager.VisualiserType(visualiserType).Name} is not an {nameof(IParameterVisualiser)} and can therefore not be linked to a value.");
			}

			Add(name, displayer, visualiser, registry, key);
		}

		public void Add(string name, object visualiserAndDisplayer, IRegistry registry, string key)
		{
			Add(new Label {Content = name}, visualiserAndDisplayer, registry, key);
		}

		public void Add(UIElement name, object visualiserAndDisplayer, IRegistry registry, string key)
		{
			UIElement displayer = visualiserAndDisplayer as UIElement;
			IParameterVisualiser visualiser = visualiserAndDisplayer as IParameterVisualiser;
			if (displayer != null && visualiser != null)
			{
				Add(name, displayer, visualiser, registry, key);
			}
			else
			{
				_log.Warn($"The passed visualiser and displayer {visualiserAndDisplayer} is either not a {typeof(UIElement)} or not a {typeof(IParameterVisualiser)}. Use other Add overloads if you want to add something.");
			}
		}

		/// <summary>
		/// Add a <see cref="UIElement"/> that contains information (e.g. the name of the object), and display it with a given object (e.g. the object to interact with).
		/// </summary>
		/// <param name="name">The element taht displays information about the elment being displayed (e.g. descriptive name).</param>
		/// <param name="displayer">The element that displays the object in the cell (normally the same as <see ref="visualiser"/>).</param>
		/// <param name="visualiser">The object that is responsible for the link with a variable (normally the same as <see ref="displayer"/>).</param>
		/// <param name="registry">The registry which contains the value that should be displayed. May or may not be <c>null</c> (depending on the visualiser).</param>
		/// <param name="key">The key to access the exact value required. May or may not be <c>null</c> (depending on the visualiser).</param>
		public void Add([CanBeNull] UIElement name, [CanBeNull] UIElement displayer, [CanBeNull] IParameterVisualiser visualiser, IRegistry registry, string key)
		{
			Content.RowDefinitions.Add(new RowDefinition {Height = GridLength.Auto});

			if (visualiser != null)
			{
				visualiser.SynchronisationHandler = SynchronisationHandler;
				visualiser.Registry = registry;
				visualiser.Key = key;
				visualiser.Read();
			}

			// add the name to the left
			if (name != null)
			{
				Grid.SetColumn(name, 0);
				Grid.SetRow(name, RowPos);
				Content.Children.Add(name);
			}

			if (displayer != null)
			{
				Grid.SetColumn(displayer, 1);
				Grid.SetRow(displayer, RowPos);
				Content.Children.Add(displayer);
			}

			RowPos++;
		}



	}
}
