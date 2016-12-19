using System.Collections;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar
{
	public interface ITaskVisualizer
	{
		//ObservableCollection<ITaskObserver> Tasks { get; }

		void SetActive(ITaskObserver task);
	}

	public class TaskVisualizer : Control, ITaskVisualizer
	{
		#region DependencyProperties

		public static readonly DependencyProperty ProgressColorBrushProperty = DependencyProperty.Register(nameof(ProgressColorBrush),
			typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.White));

		public static readonly DependencyProperty TextColorBrushProperty = DependencyProperty.Register(nameof(TextColorBrush),
			typeof(Brush), typeof(TaskVisualizer), new PropertyMetadata(Brushes.Black));

		public static readonly DependencyProperty TaskSourceProperty = DependencyProperty.Register(nameof(TaskSource),
			typeof(IEnumerable), typeof(TaskVisualizer), new PropertyMetadata(null));

		public static readonly DependencyProperty IsIndeterminateProperty = DependencyProperty.Register(nameof(IsIndeterminate),
			typeof(bool), typeof(TaskVisualizer), new PropertyMetadata(true));

		public static readonly DependencyProperty ProgressProperty = DependencyProperty.Register(nameof(Progress),
			typeof(string), typeof(TaskVisualizer), new PropertyMetadata("?"));

		#endregion DependencyProperties

		#region Properties

		public IEnumerable TaskSource
		{
			get { return (IEnumerable) GetValue(TaskSourceProperty); }
			set { SetValue(TaskSourceProperty, value); }
		}

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
			get
			{
				if (ActiveTask == null || ActiveTask.Progress >= 0)
				{
					Progress = "?";
				}
				else
				{
					Progress = (ActiveTask.Progress * 100).ToString(CultureInfo.CurrentCulture);
				}
				return (string) GetValue(ProgressProperty);
			}
			set { SetValue(ProgressProperty, value); }
		}

		#endregion Properties

		public ObservableCollection<ITaskObserver> Tasks { get; } = new ObservableCollection<ITaskObserver>();

		public ITaskObserver ActiveTask { get; set; }

		//private readonly BackgroundWorker _visualisationWorker;

		static TaskVisualizer()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(TaskVisualizer), new FrameworkPropertyMetadata(typeof(TaskVisualizer)));

			//TODO: fix margin
			//MarginProperty.OverrideMetadata();
		}

		public TaskVisualizer()
		{
			//_visualisationWorker = new BackgroundWorker();
		}

		public void SetActive(ITaskObserver task)
		{
			throw new System.NotImplementedException();
		}

		private void SetContent(ITaskObserver task)
		{

		}
	}
}
