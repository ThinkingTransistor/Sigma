using System;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Defaults.StatusBar
{
	public class StatusBarFactory : IUIFactory<UIElement>
	{
		private readonly double _height;
		private readonly GridLength[] _lengths;

		public StatusBarFactory(double height, params GridLength[] lengths)
		{
			if (height <= 0)
				throw new ArgumentOutOfRangeException(nameof(height));
			if (lengths.Length == 0)
				throw new ArgumentException("Value cannot be an empty collection.", nameof(lengths));

			_height = height;
			_lengths = lengths;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="app"></param>
		/// <param name="window"></param>
		/// <param name="parameters">The length of the perameters has to be equal to the length of <see cref="GridLength"/>s specified in the constructor. 
		/// The parameters have to be of the type <see cref="IUIFactory{T}"/>, where T is a <see cref="UIElement"/>.</param>
		/// <returns></returns>
		public UIElement CreatElement(App app, Window window, params object[] parameters)
		{
			//if (parameters.Length != _lengths.Length) throw new ArgumentException($"Value requires a length of {_lengths.Length} but has {parameters.Length}", nameof(parameters));

			Grid grid = new Grid
			{
				Height = _height,
				Background = UIResources.WindowTitleColorBrush
			};

			//TODO: single row definition rendundant?
			grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(_height) });

			// Add a column for every specified column
			for (int i = 0; i < _lengths.Length; i++)
			{
				GridLength gridLength = _lengths[i];

				ColumnDefinition newColumn = new ColumnDefinition
				{
					Width = new GridLength(gridLength.Value, gridLength.GridUnitType)
				};

				grid.ColumnDefinitions.Add(newColumn);

				var element = new StatusBarLegend {Text = $"Net {i}"};
				//UIElement element = ((IUIFactory<UIElement>) parameters[i]).CreatElement(app, window);

				grid.Children.Add(element);
				Grid.SetColumn(element, i);
			}

			return grid;
		}
	}
}
