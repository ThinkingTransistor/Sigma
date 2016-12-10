using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	public class StatusBarFactory : IUIFactory<UIElement>
	{
		public const string StatusBarLegendFactoryIdentifier = "statusbar_legend_factory";
		private readonly GridLength[] _gridLengths;

		private readonly double _height;

		private readonly int _legendColumn;

		public StatusBarFactory(IRegistry registry, double height, int legendColumn, params GridLength[] gridLengths)
		{
			if (registry == null) throw new ArgumentNullException(nameof(registry));

			if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));

			if (gridLengths.Length == 0)
				throw new ArgumentException(@"Value cannot be an empty collection.", nameof(gridLengths));

			if ((legendColumn < 0) || (legendColumn >= gridLengths.Length))
				throw new ArgumentOutOfRangeException(nameof(legendColumn));

			Registry = registry;

			_height = height;

			_legendColumn = legendColumn;
			_gridLengths = gridLengths;
		}

		public IRegistry Registry { get; set; }

		public UIElement CreatElement(App app, Window window, params object[] parameters)
		{
			IUIFactory<UIElement> legendFactory;
			if (!Registry.TryGetValue(StatusBarLegendFactoryIdentifier, out legendFactory))
			{
				legendFactory = new StatusBarLegendFactory();
				Registry.Add(StatusBarLegendFactoryIdentifier, legendFactory);
			}

			Grid grid = new Grid
			{
				Height = _height,
				Background = UIResources.WindowTitleColorBrush
			};

			// Add a column for every specified column
			foreach (GridLength gridLength in _gridLengths)
			{
				ColumnDefinition newColumn = new ColumnDefinition
				{
					Width = new GridLength(gridLength.Value, gridLength.GridUnitType)
				};

				grid.ColumnDefinitions.Add(newColumn);
			}

			AddLegends(app, window, grid, legendFactory, parameters);

			return grid;
		}

		protected void AddLegends(App app, Window window, Grid grid, IUIFactory<UIElement> factory,
			IEnumerable<object> parameters)
		{
			StackPanel stackPanel = new StackPanel {Orientation = Orientation.Horizontal};

			foreach (object legendInfo in parameters)
			{
				UIElement element = factory.CreatElement(app, window, legendInfo);
				stackPanel.Children.Add(element);
			}

			grid.Children.Add(stackPanel);
			Grid.SetColumn(stackPanel, _legendColumn);
		}
	}
}