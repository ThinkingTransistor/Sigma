using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar
{
	public interface ITaskVisualizer
	{
		void SetActive(ITaskObserver task);
	}

	public class TaskVisualizer : Control, ITaskVisualizer
	{
		#region DependencyProperties

		public static readonly DependencyProperty ProgressColorBrushProperty = DependencyProperty.Register(nameof(ProgressColorBrush),
			typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.White));

		public static readonly DependencyProperty TextColorBrushProperty = DependencyProperty.Register(nameof(TextColorBrush),
			typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.Black));

		public static readonly DependencyProperty IsIndeterminateProperty = DependencyProperty.Register(nameof(IsIndeterminate),
			typeof(bool), typeof(TaskVisualizer), new PropertyMetadata(true));

		public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress),
			typeof(string), typeof(TaskVisualizer), new PropertyMetadata("?"));

		#endregion DependencyProperties

		#region Properties

		public Brush ProgressColorBrush
		{
			get { return (Brush) GetValue(ProgressColorBrushProperty); }
			set { SetValue(ProgressColorBrushProperty, value); }
		}

		public Brush TextColorBrush
		{
			get { return (Brush) GetValue(TextColorBrushProperty); }
			set { SetValue(TextColorBrushProperty, value); }
		}

		public bool IsIndeterminate
		{
			get { return (bool) GetValue(IsIndeterminateProperty); }
			set { SetValue(IsIndeterminateProperty, value); }
		}

		public string Progress
		{
			get { return (string) GetValue(ProgressProperty); }
			set { SetValue(ProgressProperty, value); }
		}

		#endregion Properties

		private ITaskObserver _task;

		public ITaskObserver Task
		{
			get { return _task; }
			set
			{
				if (value == null || value.Progress >= 0)
				{
					Progress = "?";
				}
				else
				{
					Progress = (value.Progress * 100).ToString(CultureInfo.CurrentCulture);
				}

				_task = value;
			}
		}

		public ObservableCollection<ITaskObserver> Tasks { get; } = new ObservableCollection<ITaskObserver>();

		private readonly BackgroundWorker _visualisationWorker;

		static TaskVisualizer()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(TaskVisualizer), new FrameworkPropertyMetadata(typeof(TaskVisualizer)));

			//TODO: fix margin
			//MarginProperty.OverrideMetadata();
		}

		public TaskVisualizer()
		{
			_visualisationWorker = new BackgroundWorker();
		}

		public void SetActive(ITaskObserver task)
		{
			if (task == null)
			{
				throw new ArgumentNullException(nameof(task));
			}

			_task = task;

			task.ProgressChanged += UpdatedTask;
		}

		private void UpdatedTask(object sender, TaskEventArgs args)
		{
			Progress = (args.NewValue * 100).ToString(CultureInfo.CurrentCulture);

			if (Task.Status != TaskObserveStatus.Running)
			{
				Task.ProgressChanged -= UpdatedTask;
			}
		}
	}
}
