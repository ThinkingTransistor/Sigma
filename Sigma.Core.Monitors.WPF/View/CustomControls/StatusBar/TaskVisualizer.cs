using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.StatusBar
{
	/// <summary>
	///     This <see cref="Control" /> provides and easy way to visualise tasks.
	///     A task can be set for this <see cref="TaskVisualizer" />, and then the
	///     progress will be automatically updated.
	///     A <see cref="TaskVisualizer" /> is visible until a <see cref="ITaskObserver" />
	///     has been set. If the <see cref="ITaskObserver" /> is set to <c>null</c>, the <see cref="TaskVisualizer" />
	///     will be invisible again.
	/// </summary>
	public class TaskVisualizer : Control
	{
		static TaskVisualizer()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(TaskVisualizer),
				new FrameworkPropertyMetadata(typeof(TaskVisualizer)));

			//TODO: fix margin
			//MarginProperty.OverrideMetadata();
		}

		/// <summary>
		///     Create a new <see cref="TaskVisualizer" /> without an active <see cref="ITaskObserver" />.
		/// </summary>
		public TaskVisualizer()
		{
			SetActive(null);
		}

		/// <summary>
		///     The currently active <see cref="ITaskObserver" />.
		/// </summary>
		public ITaskObserver ActiveTask { get; private set; }

		/// <summary>
		///     Set the active <see cref="ITaskObserver" />.
		///     The <see cref="TaskVisualizer" /> is <see cref="Visibility.Hidden" />
		///     if the <see cref="ITaskObserver" /> is null, <see cref="Visibility.Visible" />
		///     otherwise.
		/// </summary>
		/// <param name="task">The <see cref="ITaskObserver" /> that the visualizer will be locked to.</param>
		public void SetActive(ITaskObserver task)
		{
			if (ActiveTask != null)
			{
				ActiveTask.ProgressChanged -= UpdatedTask;
			}

			if (task != null)
			{
				Visibility = Visibility.Visible;

				Progress = 0;

				task.ProgressChanged += UpdatedTask;

				ActiveExpressedType = task.Type.ExpressedType;
				ActiveTaskDescription = task.Description;
			}
			else
			{
				Visibility = Visibility.Hidden;
			}

			ActiveTask = task;
		}

		/// <summary>
		///     This method is meant to be called from a non-UI thread, it will be called
		///     when a <see cref="ITaskObserver.ProgressChanged" /> is raised.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="args"></param>
		private void UpdatedTask(object sender, TaskProgressEventArgs args)
		{
			Dispatcher.Invoke(() =>
			{
				Progress = args.NewValue*100;

				if (Progress < 0)
				{
					Progress = 0;
				}
			});
		}

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


		public static readonly DependencyProperty ActiveExpressedTypeProperty =
			DependencyProperty.Register(nameof(ActiveExpressedType),
				typeof(string), typeof(TaskVisualizer), new PropertyMetadata(null));

		#endregion DependencyProperties

		#region Properties

		/// <summary>
		///     The brush that will be used for the <see cref="ProgressBar" />.
		/// </summary>
		public Brush ProgressColorBrush
		{
			get { return (Brush) GetValue(ProgressColorBrushProperty); }
			set { SetValue(ProgressColorBrushProperty, value); }
		}

		/// <summary>
		///     The brush that will be used for the text (i.e. <see cref="Label" />)
		///     displaying the <see cref="ITaskObserver.Type" />.
		/// </summary>
		public Brush TextColorBrush
		{
			get { return (Brush) GetValue(TextColorBrushProperty); }
			set { SetValue(TextColorBrushProperty, value); }
		}

		/// <summary>
		///     Determines whether the given task is indeterminate or not.
		/// </summary>
		public bool IsIndeterminate
		{
			get { return (bool) GetValue(IsIndeterminateProperty); }
			set { SetValue(IsIndeterminateProperty, value); }
		}

		/// <summary>
		///     The current progress of the task. This value is between 0 and 100.
		/// </summary>
		public double Progress
		{
			get { return (double) GetValue(ProgressProperty); }
			set { SetValue(ProgressProperty, value); }
		}

		/// <summary>
		///     This is the <see cref="ITaskObserver.Description" /> of the active task.
		/// </summary>
		public string ActiveTaskDescription
		{
			get { return (string) GetValue(ActiveTaskDescriptionProperty); }
			set { SetValue(ActiveTaskDescriptionProperty, value); }
		}

		/// <summary>
		///     This is the <see cref="ITaskObserver.Type" /> of the active task.
		/// </summary>
		public string ActiveExpressedType
		{
			get { return (string) GetValue(ActiveExpressedTypeProperty); }
			set { SetValue(ActiveExpressedTypeProperty, value); }
		}

		#endregion Properties
	}
}