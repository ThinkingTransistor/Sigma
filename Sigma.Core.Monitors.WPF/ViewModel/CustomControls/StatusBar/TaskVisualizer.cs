using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
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

		public static readonly DependencyProperty ProgressColorBrushProperty =
			DependencyProperty.Register(nameof(ProgressColorBrush),
				typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.White));

		public static readonly DependencyProperty TextColorBrushProperty = DependencyProperty.Register(nameof(TextColorBrush),
			typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.Black));

		public static readonly DependencyProperty IsIndeterminateProperty =
			DependencyProperty.Register(nameof(IsIndeterminate),
				typeof(bool), typeof(TaskVisualizer), new PropertyMetadata(true));

		public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress),
			typeof(double), typeof(TaskVisualizer), new PropertyMetadata(0d));

		public static readonly DependencyProperty ActiveTaskDescriptionProperty =
			DependencyProperty.Register(nameof(ActiveTaskDescription),
				typeof(string), typeof(TaskVisualizer), new PropertyMetadata(null));


		public static readonly DependencyProperty ActiveExpressedTypeProperty = DependencyProperty.Register(nameof(ActiveExpressedType),
			typeof(string), typeof(TaskVisualizer), new PropertyMetadata(null));

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

		public double Progress
		{
			get { return (double) GetValue(ProgressProperty); }
			set { SetValue(ProgressProperty, value); }
		}

		public string ActiveTaskDescription
		{
			get { return (string) GetValue(ActiveTaskDescriptionProperty); }
			set { SetValue(ActiveTaskDescriptionProperty, value); }
		}

		public string ActiveExpressedType
		{
			get { return (string) GetValue(ActiveExpressedTypeProperty); }
			set { SetValue(ActiveExpressedTypeProperty, value); }
		}

		#endregion Properties

		public ITaskObserver ActiveTask { get; private set; }

		static TaskVisualizer()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(TaskVisualizer),
				new FrameworkPropertyMetadata(typeof(TaskVisualizer)));

			//TODO: fix margin
			//MarginProperty.OverrideMetadata();
		}

		public TaskVisualizer()
		{
			SetActive(null);
		}

		public void SetActive(ITaskObserver task)
		{
			if (task != null)
			{
				Visibility = Visibility.Visible;

				task.ProgressChanged += UpdatedTask;

				ActiveExpressedType = task.Type.ExpressedType;
				ActiveTaskDescription = task.Description;
			}
			else
			{
				Visibility = Visibility.Hidden;
			}

			if (ActiveTask != null)
			{
				ActiveTask.ProgressChanged -= UpdatedTask;
			}

			ActiveTask = task;
		}

		private void UpdatedTask(object sender, TaskProgressEventArgs args)
		{
			Dispatcher.Invoke(() =>
			{
				Progress = Math.Round(args.NewValue * 100);

				if (Progress < 0)
				{
					Progress = 0;
				}
			});
		}
	}
}
