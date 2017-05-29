using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Markup;

namespace Sigma.Core.Monitors.WPF.Panels.Games.TicTacToe
{
	[ContentProperty("MainContent")]
	public class TicTacToeButton : Control
	{
		static TicTacToeButton()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(TicTacToeButton), new FrameworkPropertyMetadata(typeof(TicTacToeButton)));
		}

		public TicTacToeButton()
		{
			ClickCommand = new ClickCommandImpl(this);
		}

		public object MainContent
		{
			get { return GetValue(MainContentProperty); }
			set { SetValue(MainContentProperty, value); }
		}

		public static readonly DependencyProperty MainContentProperty =
			DependencyProperty.Register("MainContent", typeof(object), typeof(TicTacToeButton), null);

		// Create a custom routed event by first registering a RoutedEventID
		// This event uses the bubbling routing strategy
		public static readonly RoutedEvent ClickEvent = EventManager.RegisterRoutedEvent(
			nameof(Click), RoutingStrategy.Bubble, typeof(RoutedEventHandler), typeof(TicTacToeButton));

		// Provide CLR accessors for the event
		public event RoutedEventHandler Click
		{
			add { AddHandler(ClickEvent, value); }
			remove { RemoveHandler(ClickEvent, value); }
		}

		// This method raises the Tap event
		private void RaiseClickEvent()
		{
			RoutedEventArgs newEventArgs = new RoutedEventArgs(ClickEvent);
			RaiseEvent(newEventArgs);
		}

		public ICommand ClickCommand
		{
			get { return (ICommand)GetValue(ClickCommandProperty); }
			set { SetValue(ClickCommandProperty, value); }
		}

		// Using a DependencyProperty as the backing store for ClickCommand.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty ClickCommandProperty =
			DependencyProperty.Register("ClickCommand", typeof(ICommand), typeof(TicTacToeButton), new PropertyMetadata(null));

		private class ClickCommandImpl : ICommand
		{
			private readonly TicTacToeButton _button;
			public event EventHandler CanExecuteChanged;

			public ClickCommandImpl(TicTacToeButton button)
			{
				_button = button;
			}

			public bool CanExecute(object parameter)
			{
				return true;
			}

			public void Execute(object parameter)
			{
				_button.RaiseClickEvent();
			}
		}
	}
}
