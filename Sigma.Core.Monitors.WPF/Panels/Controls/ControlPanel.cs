/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;
using Sigma.Core.Training;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	/// <summary>
	/// This <see cref="SigmaPanel"/> allows to control the training progress for a trainer and
	/// visualises most important data. 
	/// </summary>
	public class ControlPanel : GenericPanel<StackPanel>
	{
		private readonly SigmaPlaybackControl _playbackControl;

		private ITrainer _trainer;

		/// <summary>
		/// The this ControlPanel belongs to. It can control this trainer (play, pause, next stop, rewind).
		/// </summary>
		public ITrainer Trainer
		{
			get { return _trainer; }
			set
			{
				_trainer = value;
				_playbackControl.Trainer = value;
			}
		}

		public ControlPanel(string title, object content = null) : this(title, null, content) { }

		public ControlPanel(string title, ITrainer trainer, object content = null) : base(title, content)
		{
			_trainer = trainer;

			//TODO: style?
			Content = new StackPanel
			{
				Orientation = Orientation.Vertical,
				HorizontalAlignment = HorizontalAlignment.Center,
				Margin = new Thickness(0, 20, 0, 0)
			};

			_playbackControl = new SigmaPlaybackControl { Trainer = Trainer };

			Content.Children.Add(_playbackControl);
		}
	}
}
