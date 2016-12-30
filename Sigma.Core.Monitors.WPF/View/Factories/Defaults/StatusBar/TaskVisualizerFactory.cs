using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.ViewModel.StatusBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	/// <summary>
	/// This factory creates multiple <see cref="CustomControls.StatusBar.TaskVisualizer"/>.
	/// Additionally, it defines the style for the "show more indicator" when too many tasks are concurrently
	/// running. 
	/// </summary>
	public class TaskVisualizerFactory : IUIFactory<UIElement>
	{
		/// <summary>
		/// The amount of maximum <see cref="CustomControls.StatusBar.TaskVisualizer"/>.
		/// </summary>
		private readonly int _maxElements;

		/// <summary>
		/// A list of all generated <see cref="IWpfTaskVisualizationManager"/>. This is required
		/// for passing active and pending tasks through tabs. 
		/// </summary>
		private readonly List<IWpfTaskVisualizationManager> _visualizationManagers;

		/// <summary>
		/// The factory that will be used to generate the "show more indicator".
		/// </summary>
		private readonly IUIFactory<UIElement> _showMoreFactory;

		/// <summary>
		/// Create a <see cref="TaskVisualizerFactory"/> with the given amount of <see cref="maxElements"/>.
		/// 
		/// A factory that produces a <see cref="Label"/> with the <see cref="Style"/> "ShowMoreIndicator" and the content "..." will be used 
		/// for the creation of the "show more indicator". 
		/// </summary>
		/// <param name="maxElements">The amount of maximum <see cref="CustomControls.StatusBar.TaskVisualizer"/>.</param>
		public TaskVisualizerFactory(int maxElements) : this(maxElements, "...") { }

		/// <summary>
		/// Create a <see cref="TaskVisualizerFactory"/> with the given amount of <see cref="maxElements"/>.
		/// 
		/// A factory that produces a <see cref="Label"/> with the <see cref="Style"/> "ShowMoreIndicator" and <see cref="content"/> will be used 
		/// for the creation of the "show more indicator". 
		/// </summary>
		/// <param name="maxElements">The amount of maximum <see cref="CustomControls.StatusBar.TaskVisualizer"/>.</param>
		/// <param name="content">The content for the <see cref="Label"/>.</param>
		public TaskVisualizerFactory(int maxElements, object content) : this(maxElements, Application.Current.Resources["ShowMoreIndicator"] as Style, content) { }

		/// <summary>
		/// Create a <see cref="TaskVisualizerFactory"/> with the given amount of <see cref="maxElements"/>.
		/// A factory that produces a <see cref="Label"/> with a given <see cref="labelStyle"/> and <see cref="content"/> will be used 
		/// for the creation of the "show more indicator". 
		/// 
		/// </summary>
		/// <param name="maxElements">The amount of maximum <see cref="CustomControls.StatusBar.TaskVisualizer"/>.</param>
		/// <param name="labelStyle">The <see cref="Style"/> that will be applied to the created <see cref="Label"/>.</param>
		/// <param name="content">The content for the <see cref="Label"/>.</param>
		public TaskVisualizerFactory(int maxElements, Style labelStyle, object content) : this(maxElements, new ShowMoreFactory(labelStyle, content)) { }

		/// <summary>
		/// Create a <see cref="TaskVisualizerFactory"/> with the given amount of <see cref="maxElements"/>
		/// and a <see cref="showMoreFactory"/>.
		/// </summary>
		/// <param name="maxElements">The amount of maximum <see cref="CustomControls.StatusBar.TaskVisualizer"/>.</param>
		/// <param name="showMoreFactory">The factory that will be used to generate the "show more indicator". May not be <c>null</c>.
		/// 
		/// i.e. If too many tasks are concurrently running, it has to be indicated that there are more tasks running
		/// than displayed. </param>
		public TaskVisualizerFactory(int maxElements, IUIFactory<UIElement> showMoreFactory)
		{
			if (showMoreFactory == null)
			{
				throw new ArgumentNullException(nameof(showMoreFactory));
			}

			_maxElements = maxElements;
			_visualizationManagers = new List<IWpfTaskVisualizationManager>();

			_showMoreFactory = showMoreFactory;
		}

		public UIElement CreateElement(Application app, Window window, params object[] parameters)
		{
			IWpfTaskVisualizationManager manager;
			UIElement showMoreIndicator = _showMoreFactory.CreateElement(app, window, parameters);

			lock (_visualizationManagers)
			{
				IEnumerable<ITaskObserver> activeTasks = _visualizationManagers.Count > 0 ? _visualizationManagers[0].ActiveTasks : null;
				IEnumerable<ITaskObserver> pendingTasks = _visualizationManagers.Count > 0 ? _visualizationManagers[0].PendingTasks : null;

				manager = new TaskVisualizationManager(_maxElements, showMoreIndicator, SigmaEnvironment.TaskManager, activeTasks, pendingTasks);

				_visualizationManagers.Add(manager);
			}

			// dispose the manager on close
			window.Closed += (sender, args) =>
			{
				manager.Dispose();

				lock (_visualizationManagers)
				{
					_visualizationManagers.Remove(manager);
				}
			};

			Grid grid = new Grid();

			// + 1 for the show more indicator
			for (int i = 0; i <= _maxElements; i++)
			{
				grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Auto) });
			}

			// add the TaskVisualizers
			for (int i = 0; i < _maxElements; i++)
			{
				grid.Children.Add(manager.TaskVisualizers[i]);
				Grid.SetColumn(manager.TaskVisualizers[i], i);
			}

			// add the show more indicator
			grid.Children.Add(showMoreIndicator);
			Grid.SetColumn(showMoreIndicator, _maxElements);

			return grid;
		}

		/// <summary>
		/// This factory creates a <see cref="Label"/> with a given <see cref="Style"/>
		/// and content. 
		/// </summary>
		private class ShowMoreFactory : IUIFactory<UIElement>
		{
			/// <summary>
			/// The <see cref="Style"/> for the label.
			/// </summary>
			private readonly Style _labelStyle;

			/// <summary>
			/// The content for the label. 
			/// </summary>
			private readonly object _content;

			/// <summary>
			/// Create a "label factory" to produce labels with given style and content.
			/// </summary>
			/// <param name="labelStyle">The style that should be applied.</param>
			/// <param name="content">The content the label will receive.</param>
			public ShowMoreFactory(Style labelStyle, object content)
			{
				if (labelStyle == null)
				{
					throw new ArgumentNullException(nameof(labelStyle));
				}
				if (content == null)
				{
					throw new ArgumentNullException(nameof(content));
				}

				_labelStyle = labelStyle;
				_content = content;
			}

			public UIElement CreateElement(Application app, Window window, params object[] parameters)
			{
				return new Label { Content = _content, Style = _labelStyle };
			}
		}
	}
}