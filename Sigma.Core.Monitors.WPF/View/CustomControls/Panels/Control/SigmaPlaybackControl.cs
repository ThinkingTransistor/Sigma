using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control
{
	public class SigmaPlaybackControl : System.Windows.Controls.Control
	{
		static SigmaPlaybackControl()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(SigmaPlaybackControl), new FrameworkPropertyMetadata(typeof(SigmaPlaybackControl)));
		}

		#region DependencyProperties

		public static readonly DependencyProperty RunningProperty = DependencyProperty.Register(nameof(Running),
			typeof(bool), typeof(SigmaPlaybackControl), new UIPropertyMetadata(false));

		public static readonly DependencyProperty OrientationProperty = DependencyProperty.Register(nameof(Orientation),
			typeof(Orientation), typeof(SigmaPlaybackControl), new UIPropertyMetadata(Orientation.Horizontal));

		#endregion DependencyProperties

		#region Properties

		public bool Running
		{
			get { return (bool) GetValue(RunningProperty); }
			set { SetValue(RunningProperty, value); }
		}

		public Orientation Orientation
		{
			get { return (Orientation) GetValue(OrientationProperty); }
			set { SetValue(OrientationProperty, value); }
		}

		#endregion Properties

		internal void TogglePlay()
		{

		}

		internal void Rewind()
		{

		}

		internal void Step()
		{

		}
	}
}
