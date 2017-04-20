/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;
using Sigma.Core.Monitors.WPF.View.Parameterisation;
using Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	/// <summary>
	/// This <see cref="SigmaPanel"/> allows to control the training progress for a trainer and
	/// visualises most important data. 
	/// </summary>
	public class ControlPanel : GenericPanel<StackPanel>
	{
		private SigmaPlaybackControl _playbackControl;
		private ParameterView _parameterView;

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

		/// <summary>
		/// This list stores all trainers that have been initialised.
		/// Required to only add one hook per trainer.
		/// </summary>
		private static readonly IList<ITrainer> Trainers;

		static ControlPanel()
		{
			Trainers = new List<ITrainer>();
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
		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected override void OnInitialise(WPFWindow window)
		{
			throw new InvalidOperationException($"{nameof(ControlPanel)} is only compatible with {nameof(SigmaWindow)}s.");
		}

		/// <summary>
		/// This method will be called after the panel has been added (window, monitor set...)
		/// </summary>
		protected override void OnInitialise(SigmaWindow window)
		{
			if (!Trainers.Contains(Trainer))
			{
				ValueSourceReporterHook valueHook = new ValueSourceReporterHook(TimeStep.Every(1, TimeScale.Epoch), "runtime_millis");
				_trainer.AddGlobalHook(valueHook);
				Monitor.Sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);
				Trainers.Add(Trainer);

				valueHook = new ValueSourceReporterHook(TimeStep.Every(1, TimeScale.Iteration), "iteration");
				_trainer.AddLocalHook(valueHook);
				Monitor.Sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);
			}

			//TODO: style?
			_playbackControl = new SigmaPlaybackControl { Trainer = Trainer, Margin = new Thickness(0, 0, 0, 20), HorizontalAlignment = HorizontalAlignment.Center };

			Content.Children.Add(_playbackControl);

			_parameterView = new ParameterView(Monitor.Sigma, window);

			SigmaTextBlock timeBox = (SigmaTextBlock) _parameterView.Add(Properties.Resources.RunningTime, typeof(object), _trainer.Operator.Registry, "runtime_millis");
			timeBox.AutoPollValues(_trainer, TimeStep.Every(1, TimeScale.Epoch));
			timeBox.Postfix = " ms";

			UserControlParameterVisualiser epochBox = (UserControlParameterVisualiser) _parameterView.Add(Properties.Resources.CurrentEpoch, typeof(object), _trainer.Operator.Registry, "epoch");
			epochBox.AutoPollValues(_trainer, TimeStep.Every(1, TimeScale.Epoch));

			UserControlParameterVisualiser iterationBox = (UserControlParameterVisualiser) _parameterView.Add(Properties.Resources.CurrentIteration, typeof(object), _trainer.Operator.Registry, "iteration");
			iterationBox.AutoPollValues(_trainer, TimeStep.Every(1, TimeScale.Iteration));

			IRegistry registry = new Registry
			{
				{ "operator", Trainer.Operator.GetType().Name },
				{ "optimiser", Trainer.Optimiser.GetType().Name }
			};
			_parameterView.Add(Properties.Resources.CurrentOperator, typeof(object), registry, "operator");
			_parameterView.Add(Properties.Resources.CurrentOptimiser, typeof(object), registry, "optimiser");
			//TODO: completely hardcoded activation function
			UserControlParameterVisualiser activationBox = (UserControlParameterVisualiser) _parameterView.Add(Properties.Resources.CurrentActivationFunction, typeof(object), _trainer.Operator.Registry, "network.layers.2-fullyconnected.activation");
			activationBox.AutoPollValues(_trainer, TimeStep.Every(1, TimeScale.Start));

			Content.Children.Add(_parameterView);
		}
	}
}
