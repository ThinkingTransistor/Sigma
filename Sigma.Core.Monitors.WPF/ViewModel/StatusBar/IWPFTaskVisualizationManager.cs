using System;
using System.Collections.Generic;
using System.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.StatusBar
{
	public interface IWpfTaskVisualizationManager : IDisposable
	{
		/// <summary>
		/// The <see cref="TaskVisualizer"/>s to keep track of. 
		/// </summary>
		TaskVisualizer[] TaskVisualizers { get; }

		/// <summary>
		/// The indicator that will be shown when too many tasks are visible. 
		/// </summary>
		UIElement MoreTasksIndicator { get; }

		/// <summary>
		/// An array of the active <see cref="ITaskObserver"/>s. 
		/// If there is no task, for an index, the task is null. 
		/// </summary>
		ITaskObserver[] ActiveTasks { get; }

		/// <summary>
		/// A list of the pending tasks. 
		/// </summary>
		List<ITaskObserver> PendingTasks { get; }
	}
}