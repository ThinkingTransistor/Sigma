using System.Collections;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar
{
	public interface ITaskVisualizer
	{
		//ObservableCollection<ITaskObserver> Tasks { get; }

		void SetActive(ITaskObserver task);
	}

	/// <summary>
	/// Follow steps 1a or 1b and then 2 to use this custom control in a XAML file.
	///
	/// Step 1a) Using this custom control in a XAML file that exists in the current project.
	/// Add this XmlNamespace attribute to the root element of the markup file where it is 
	/// to be used:
	///
	///     xmlns:MyNamespace="clr-namespace:Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar"
	///
	///
	/// Step 1b) Using this custom control in a XAML file that exists in a different project.
	/// Add this XmlNamespace attribute to the root element of the markup file where it is 
	/// to be used:
	///
	///     xmlns:MyNamespace="clr-namespace:Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;assembly=Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar"
	///
	/// You will also need to add a project reference from the project where the XAML file lives
	/// to this project and Rebuild to avoid compilation errors:
	///
	///     Right click on the target project in the Solution Explorer and
	///     "Add Reference"->"Projects"->[Browse to and select this project]
	///
	///
	/// Step 2)
	/// Go ahead and use your control in the XAML file.
	///
	///     <MyNamespace:TaskVisualizer/>
	///
	/// </summary>
	public class TaskVisualizer : Control, ITaskVisualizer
	{
		#region DependencyProperties

		public static readonly DependencyProperty CycleProperty = DependencyProperty.Register(nameof(Cycle),
			typeof(bool), typeof(TaskVisualizer), new UIPropertyMetadata(true));

		public static readonly DependencyProperty TaskSourceProperty = DependencyProperty.Register(nameof(TaskSource),
			typeof(IEnumerable), typeof(TaskVisualizer), new PropertyMetadata(null));

		#endregion DependencyProperties

		#region Properties

		public bool Cycle
		{
			get { return (bool) GetValue(CycleProperty); }
			set { SetValue(CycleProperty, value); }
		}

		public IEnumerable TaskSource
		{
			get { return (IEnumerable) GetValue(TaskSourceProperty); }
			set { SetValue(TaskSourceProperty, value); }
		}

		#endregion Properties

		public ObservableCollection<ITaskObserver> Tasks { get; } = new ObservableCollection<ITaskObserver>();

		public ITaskObserver ActiveTask { get; set; }

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
			throw new System.NotImplementedException();
		}

		private void SetContent(ITaskObserver task)
		{

		}
	}
}
