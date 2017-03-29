/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using log4net;
using Sigma.Core.Training;
using Sigma.Core.Training.Operators;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control
{
	/// <summary>
	/// This "playback control" works controlling the music. Play it, pause it, rewind it.
	/// And this all works with the trainer.
	/// </summary>
	public class SigmaPlaybackControl : System.Windows.Controls.Control
	{
		static SigmaPlaybackControl()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(SigmaPlaybackControl), new FrameworkPropertyMetadata(typeof(SigmaPlaybackControl)));
		}

		public ITrainer Trainer
		{
			get { return (ITrainer) GetValue(TrainerProperty); }
			set { SetValue(TrainerProperty, value); }
		}

		public static readonly DependencyProperty TrainerProperty =
			DependencyProperty.Register("Trainer", typeof(ITrainer), typeof(SigmaPlaybackControl), new PropertyMetadata(null));

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

		private class DefaultTogglePlay : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				ITrainer trainer = Control.Trainer;
				IOperator @operator = trainer.Operator;

				if (@operator.State == ExecutionState.Running)
				{
					@operator.SignalPause();
				}
				else if (@operator.State == ExecutionState.Paused)
				{
					@operator.SignalResume();
				}
				else if (@operator.State == ExecutionState.None)
				{
					@operator.Start();
				}
			}

			public DefaultTogglePlay(SigmaPlaybackControl control) : base(control) { }
		}

		private class DefaultRewind : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				//Debug.WriteLine("Rewind!");
				Control.Running = false;
				ITrainer trainer = Control.Trainer;
				trainer.Reset();
				trainer.Initialise(trainer.Operator.Handler); // because we're manually resetting we have to initialise manually as well
															  // TODO maybe find a nicer way to reset and reinitialise - maybe separate command?
			}

			public DefaultRewind(SigmaPlaybackControl control) : base(control) { }
		}

		private class DefaultStep : DefaultCommand
		{
			public override void Execute(object parameter)
			{
				//Debug.WriteLine("Step!");

				Control.Running = false;

				LogManager.GetLogger(typeof(DefaultStep)).Fatal("Step not yet implemented!");

				//#if DEBUG
				//				if (Control.Task != null)
				//				{
				//					SigmaEnvironment.TaskManager.CancelTask(Control.Task);
				//				}
				//#endif
			}

			public DefaultStep(SigmaPlaybackControl control) : base(control) { }
		}
	}
}
