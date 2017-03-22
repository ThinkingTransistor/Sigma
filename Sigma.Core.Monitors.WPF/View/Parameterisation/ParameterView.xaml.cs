/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Annotations;
using System.Windows.Controls;
using log4net;
using Sigma.Core.Monitors.Synchronisation;
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

		protected readonly IParameterVisualiserManager Manager;
		protected readonly ISynchronisationHandler SynchronisationHandler;

		protected int RowPos;

		public ParameterView(IParameterVisualiserManager manager, ISynchronisationHandler synchronisationHandler)
		{
			Manager = manager;
			SynchronisationHandler = synchronisationHandler;
			InitializeComponent();
		}

		public void Add(string name, Type type, IRegistry registry, string key)
		{
			Add(new Label { Content = name }, type, registry, key);
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
			Add(new Label { Content = name }, visualiserAndDisplayer, registry, key);
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
		/// 
		/// </summary>
		/// <param name="name"></param>
		/// <param name="displayer">The element that displays the object in the cell (normally the same as <see ref="visualiser"/>).</param>
		/// <param name="visualiser">The object that is responsible for the link with a variable (normally the same as <see ref="displayer"/>).</param>
		/// <param name="registry"></param>
		/// <param name="key"></param>
		public void Add(UIElement name, UIElement displayer, IParameterVisualiser visualiser, IRegistry registry, string key)
		{
			Content.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

			if (visualiser != null)
			{
				visualiser.SynchronisationHandler = SynchronisationHandler;
				visualiser.Registry = registry;
				visualiser.Key = key;
				visualiser.Read();
			}

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
