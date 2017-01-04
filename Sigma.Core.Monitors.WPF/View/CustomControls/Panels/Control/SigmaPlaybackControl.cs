using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

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
            DependencyProperty.Register("TogglePlay", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(new DefaultTogglePlay()));

        public ICommand Rewind
        {
            get { return (ICommand) GetValue(RewindProperty); }
            set { SetValue(RewindProperty, value); }
        }

        public static readonly DependencyProperty RewindProperty =
            DependencyProperty.Register("Rewind", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(new DefaultRewind()));

        public ICommand Step
        {
            get { return (ICommand) GetValue(StepProperty); }
            set { SetValue(StepProperty, value); }
        }

        public static readonly DependencyProperty StepProperty =
            DependencyProperty.Register("Step", typeof(ICommand), typeof(SigmaPlaybackControl), new PropertyMetadata(new DefaultStep()));

        public SigmaPlaybackControl()
        {

        }

        public SigmaPlaybackControl(ICommand togglePlay = null, ICommand rewind = null, ICommand step = null)
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
                Debug.WriteLine("Toggle Play");
            }
        }

        private class DefaultRewind : DefaultCommand
        {
            public override void Execute(object parameter)
            {
                Debug.WriteLine("Rewind!");
            }
        }

        private class DefaultStep : DefaultCommand
        {
            public override void Execute(object parameter)
            {
                Debug.WriteLine("Step!");
            }
        }
    }
}
