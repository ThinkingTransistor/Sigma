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

		public void Add(UIElement name, Type type, IRegistry registry, string key)
		{
			Content.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

			Grid.SetColumn(name, 0);
			Grid.SetRow(name, RowPos);

			UIElement displayer = (UIElement)Activator.CreateInstance(Manager.VisualiserType(type));
			IParameterVisualiser visualiser = displayer as IParameterVisualiser;

			if (visualiser == null)
			{
				_log.Warn($"{Manager.VisualiserType(type).Name} is not an {nameof(IParameterVisualiser)} and can therefore not be linked to a value.");
			}
			else
			{
				visualiser.SynchronisationHandler = SynchronisationHandler;
				visualiser.Registry = registry;
				visualiser.Key = key;
				visualiser.Read();
			}

			Grid.SetColumn(displayer, 1);
			Grid.SetRow(displayer, RowPos);

			Content.Children.Add(displayer);
			Content.Children.Add(name);

			RowPos++;
		}
	}
}
