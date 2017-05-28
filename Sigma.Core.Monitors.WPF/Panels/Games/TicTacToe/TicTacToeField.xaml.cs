using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Sigma.Core.Handlers;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Panels.Games.TicTacToe
{
	public enum TicTacToePlayer
	{
		None = 0,
		X,
		O
	}

	public static class TicTacToePlayerExtension
	{
		public static string GetPlayerText(this TicTacToePlayer player)
		{
			return player == TicTacToePlayer.None ? "" : player.ToString();
		}

		public static TicTacToePlayer TogglePlayer(this TicTacToePlayer player)
		{
			if (player == TicTacToePlayer.None) return TicTacToePlayer.None;
			return player == TicTacToePlayer.O ? TicTacToePlayer.X : TicTacToePlayer.O;
		}
	}

	public partial class TicTacToeField
	{
		private readonly WPFMonitor _monitor;

		/// <summary>
		/// The player that will start the game.
		/// </summary>
		public readonly TicTacToePlayer StartTicTacToePlayer = TicTacToePlayer.X;
		/// <summary>
		/// The real player. 
		/// </summary>
		public readonly TicTacToePlayer RealTicTacToePlayer = TicTacToePlayer.O;

		/// <summary>
		/// The next player (the current turn player).
		/// </summary>
		public TicTacToePlayer NextTicTacToePlayer { get; protected set; }


		/// <summary>
		/// The current game field - d onot set the values manually, since the gamefield has to update accordingly.
		/// </summary>
		private TicTacToePlayer[,] _field;

		/// <summary>
		/// All buttons that are added in the UI.
		/// </summary>
		private readonly TicTacToeButton[,] _buttons;

		private bool _gameOver;

		public TicTacToeField(WPFMonitor monitor)
		{
			_monitor = monitor;
			InitializeComponent();

			_buttons = new TicTacToeButton[3, 3];
			for (int i = 0; i < _buttons.GetLength(0); i++)
			{
				for (int j = 0; j < _buttons.GetLength(1); j++)
				{
					int row = i;
					int column = j;

					TicTacToeButton button = new TicTacToeButton();
					button.Click += (sender, args) => ButtonClick(row, column);
					_buttons[i, j] = button;

					ContentGrid.Children.Add(button);
					Grid.SetRow(button, row);
					Grid.SetColumn(button, column);
				}
			}

			InitGame();
		}

		public void AsINDarray(IComputationHandler handler)
		{
			int[] vals = new int[_field.GetLength(0) * _field.GetLength(1)];
			for (int i = 0; i < _field.GetLength(0); i++)
			{
				for (int j = 0; j < _field.GetLength(1); j++)
				{
					int number = 0;

					if (_field[i, j] == RealTicTacToePlayer)
					{
						number = -1;
					}
					else if (_field[i, j] != TicTacToePlayer.None)
					{
						number = 1;
					}

					vals[i * _field.GetLength(0) + j] = number;
				}
			}

			handler.NDArray(vals, vals.Length);
		}

		private void InitField()
		{
			if (_field == null)
			{
				_field = new TicTacToePlayer[_buttons.GetLength(0), _buttons.GetLength(1)];
			}
			for (int i = 0; i < _field.GetLength(0); i++)
			{
				for (int j = 0; j < _field.GetLength(1); j++)
				{
					SetIndexNoCheck(i, j, TicTacToePlayer.None);
				}
			}
		}

		private void SetIndexNoCheck(int row, int column, TicTacToePlayer move)
		{
			_field[row, column] = move;
			_buttons[row, column].MainContent = move.GetPlayerText();
		}

		public void SetIndex(int row, int column, TicTacToePlayer move)
		{
			SetIndexNoCheck(row, column, move);

			if (GameOver(row, column, out TicTacToePlayer winner))
			{
				//TODO: fix cast
				SigmaWindow window = (SigmaWindow)_monitor.Window;

				if (winner == TicTacToePlayer.None)
				{
					Task.Factory.StartNew(() => window.SnackbarMessageQueue.Enqueue("It's a tie!", "Got it", null));
				}
				else
				{
					Task.Factory.StartNew(() => window.SnackbarMessageQueue.Enqueue($"Player: {winner} has won the game!", "Got it", null));
				}

				_gameOver = true;

				foreach (TicTacToeButton ticTacToeButton in _buttons) { ticTacToeButton.IsEnabled = false; }
			}
		}

		public void InitGame()
		{
			_gameOver = false;
			foreach (TicTacToeButton ticTacToeButton in _buttons) { ticTacToeButton.IsEnabled = true; }
			NextTicTacToePlayer = StartTicTacToePlayer;
			InitField();
		}

		public bool CanPlace(int row, int column)
		{
			return !_gameOver && _field[row, column] == TicTacToePlayer.None;
		}

		public bool Place(int row, int column)
		{
			if (!CanPlace(row, column))
			{
				return false;
			}

			SetIndex(row, column, NextTicTacToePlayer);
			NextTicTacToePlayer = NextTicTacToePlayer.TogglePlayer();

			return true;
		}

		#region CheckGameOver

		public bool GameOver(int row, int column, out TicTacToePlayer winner)
		{
			if ((winner = CheckColumn(column)) == TicTacToePlayer.None)
			{
				if ((winner = CheckRow(row)) == TicTacToePlayer.None)
				{
					if ((winner = CheckDiagonal(row, column)) == TicTacToePlayer.None)
					{
						winner = CheckAntiDiagonal(row, column);
					}
				}
			}


			if (winner != TicTacToePlayer.None)
			{
				return true;
			}

			return !FieldContainsEmptyCell();
		}

		private TicTacToePlayer CheckColumn(int column)
		{
			for (int row = 1; row < _field.GetLength(0); row++)
			{
				if (_field[row - 1, column] != _field[row, column])
				{
					break;
				}

				if (row + 1 == _field.GetLength(0))
				{
					return _field[row, column];
				}
			}

			return TicTacToePlayer.None;
		}

		private TicTacToePlayer CheckRow(int row)
		{
			for (int col = 1; col < _field.GetLength(1); col++)
			{
				if (_field[row, col - 1] != _field[row, col])
				{
					break;
				}

				if (col + 1 == _field.GetLength(1))
				{
					return _field[row, col];
				}
			}

			return TicTacToePlayer.None;
		}

		private TicTacToePlayer CheckDiagonal(int row, int column)
		{
			if (row != column)
			{
				return TicTacToePlayer.None;
			}

			for (int i = 1; i < _field.GetLength(0); i++)
			{
				if (_field[i, i] != _field[i - 1, i - 1])
				{
					break;
				}
				if (i + 1 == _field.GetLength(0))
				{
					return _field[i, i];
				}
			}

			return TicTacToePlayer.None;
		}

		private TicTacToePlayer CheckAntiDiagonal(int row, int column)
		{
			int lengthMinus1 = _field.GetLength(0) - 1;
			for (int i = 1; i < _field.GetLength(0); i++)
			{
				if (_field[i, lengthMinus1 - i] != _field[i - 1, lengthMinus1 - (i - 1)])
				{
					break;
				}

				if (i == lengthMinus1)
				{
					return _field[i, lengthMinus1 - i];
				}
			}

			return TicTacToePlayer.None;
		}

		private bool FieldContainsEmptyCell()
		{
			return _field.Cast<TicTacToePlayer>().Any(ticTacToePlayer => ticTacToePlayer == TicTacToePlayer.None);
		}

		#endregion

		private void ButtonClick(int row, int column)
		{
			Place(row, column);
		}

		private void Reset_OnClick(object sender, RoutedEventArgs e)
		{
			InitGame();
		}

		#region Dependy Properties
		public bool Autoplay
		{
			get { return (bool)GetValue(AutoplayProperty); }
			set { SetValue(AutoplayProperty, value); }
		}

		// Using a DependencyProperty as the backing store for Autoplay.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty AutoplayProperty =
			DependencyProperty.Register("Autoplay", typeof(bool), typeof(TicTacToeField), new PropertyMetadata(true));
		#endregion

	}
}
