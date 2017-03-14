/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// Interaction logic for ParameterView.xaml
	/// </summary>
	public partial class ParameterView
	{
		protected readonly IParameterVisualiserManager _manager;

		protected int RowPos;

		public ParameterView(IParameterVisualiserManager manager)
		{
			_manager = manager;
			InitializeComponent();
		}

		public void Add(string name, Type type)
		{
			Add(new Label { Content = name }, type);
		}

		public void Add(UIElement name, Type type)
		{
			Content.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

			Grid.SetColumn(name, 0);
			Grid.SetRow(name, RowPos);

			UIElement displayer = (UIElement) Activator.CreateInstance(_manager.VisualiserType(type));
			Grid.SetColumn(displayer, 1);
			Grid.SetRow(displayer, RowPos);

			Content.Children.Add(displayer);
			Content.Children.Add(name);

			RowPos++;
		}
	}
}
