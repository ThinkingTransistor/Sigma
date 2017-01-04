using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control
{
	public class SigmaPlaybackControl : System.Windows.Controls.Control
	{
		static SigmaPlaybackControl()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(SigmaPlaybackControl), new FrameworkPropertyMetadata(typeof(SigmaPlaybackControl)));
		}

		public bool Running
		{
			get { return (bool) GetValue(RunningProperty); }
			set { SetValue(RunningProperty, value); }
		}

		public static readonly DependencyProperty RunningProperty = DependencyProperty.Register(nameof(Running),
			typeof(bool), typeof(SigmaPlaybackControl), new UIPropertyMetadata(false));

		public Orientation Orientation
		{
			get { return (Orientation) GetValue(OrientationProperty); }
			set { SetValue(OrientationProperty, value); }
		}

		public static readonly DependencyProperty OrientationProperty = DependencyProperty.Register(nameof(Orientation),
			typeof(Orientation), typeof(SigmaPlaybackControl), new UIPropertyMetadata(Orientation.Horizontal));

		public ICommand TogglePlay
		{
			get { return (ICommand) GetValue(TogglePlayProperty); }
			set { SetValue(TogglePlayProperty, value); }
		}

		public static readonly DependencyProperty TogglePlayProperty =
			DependencyProperty.Register("TogglePlay", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(null));

		public ICommand Rewind
		{
			get { return (ICommand) GetValue(RewindProperty); }
			set { SetValue(RewindProperty, value); }
		}

		public static readonly DependencyProperty RewindProperty =
			DependencyProperty.Register("Rewind", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(null));

		public ICommand Step
		{
			get { return (ICommand) GetValue(StepProperty); }
			set { SetValue(StepProperty, value); }
		}

		public static readonly DependencyProperty StepProperty =
			DependencyProperty.Register("Step", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(null));

		public SigmaPlaybackControl()
		{
			TogglePlay = new DefaultTogglePlay(this);
			Rewind = new DefaultRewind(this);
			Step = new DefaultStep(this);
		}

		public SigmaPlaybackControl(ICommand togglePlay = null, ICommand rewind = null, ICommand step = null) : this()
		{
			if (togglePlay != null)
			{
				TogglePlay = togglePlay;
			}

			if (rewind != null)
			{
				Rewind = rewind;
			}

			if (step != null)
			{
				Step = step;
			}
		}

		private abstract class DefaultCommand : ICommand
		{
			protected readonly SigmaPlaybackControl Control;

			protected DefaultCommand(SigmaPlaybackControl control)
			{
				Control = control;
			}

			public bool CanExecute(object parameter)
			{
				return true;
			}

			public abstract void Execute(object parameter);

			public event EventHandler CanExecuteChanged;
		}

#if DEBUG
		internal ITaskObserver Task;
#endif

		private class DefaultTogglePlay : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				Debug.WriteLine("Toggle play clicked");
#if DEBUG
				if (Control.Running)
				{
					Control.Task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Train, "Well, now I'm training");
				}
				else
				{
					SigmaEnvironment.TaskManager.CancelTask(Control.Task);
				}
#endif
			}

			public DefaultTogglePlay(SigmaPlaybackControl control) : base(control)
			{
			}
		}

		private class DefaultRewind : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				Debug.WriteLine("Rewind!");
				Control.Running = false;
#if DEBUG
				SigmaEnvironment.TaskManager.CancelTask(Control.Task);
#endif
			}

			public DefaultRewind(SigmaPlaybackControl control) : base(control)
			{
			}
		}

		private class DefaultStep : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				Debug.WriteLine("Step!");
			}

			public DefaultStep(SigmaPlaybackControl control) : base(control)
			{
			}
		}
	}
}
