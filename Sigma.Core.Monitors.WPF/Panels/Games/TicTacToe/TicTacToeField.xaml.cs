using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.View.Windows;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

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
		public class TicTacToeMoveEventArgs : EventArgs
		{
			public bool IsReset { get; }
			public int Row { get; }
			public int Column { get; }

			public TicTacToePlayer Move { get; }

			public bool GameOver { get; }

			public TicTacToeMoveEventArgs()
			{
				IsReset = true;
			}

			public TicTacToeMoveEventArgs(int row, int column, TicTacToePlayer move, bool gameOver)
			{
				Row = row;
				Column = column;
				Move = move;
				GameOver = gameOver;
			}
		}

		private readonly WPFMonitor _monitor;

		/// <summary>
		/// The player that will start the game.
		/// </summary>
		public const TicTacToePlayer StartTicTacToePlayer = TicTacToePlayer.X;

		/// <summary>
		/// The real player. 
		/// </summary>
		public const TicTacToePlayer RealTicTacToePlayer = TicTacToePlayer.X;

		/// <summary>
		/// The computer player.
		/// </summary>
		public readonly TicTacToePlayer AiTicTacToePlayer = StartTicTacToePlayer.TogglePlayer();

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

		/// <summary>
		/// The current field as an INDArray (references).
		/// </summary>
		public INDArray Field { get; protected set; }

		/// <summary>
		/// A field changed event handler, that occurs every time the field changes. (e.g. place a new move / reset).
		/// </summary>
		public event EventHandler<TicTacToeMoveEventArgs> FieldChange;

		public event EventHandler AiMove;

		public TicTacToeField(IComputationHandler handler, WPFMonitor monitor)
		{
			_monitor = monitor;
			InitializeComponent();

			_buttons = new TicTacToeButton[3, 3];
			Field = handler.NDArray(_buttons.GetLength(0) * _buttons.GetLength(1));

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

		protected virtual int MapToInt(TicTacToePlayer player)
		{
			if (player == AiTicTacToePlayer) return 1;
			if (player == RealTicTacToePlayer) return -1;

			return 0;
		}

		protected void OnAiMove()
		{
			AiMove?.Invoke(this, new EventArgs());
		}

		protected void OnFieldChange()
		{
			FieldChange?.Invoke(this, new TicTacToeMoveEventArgs());
		}

		protected void OnFieldChange(int row, int column, TicTacToePlayer move, bool gameOver)
		{
			FieldChange?.Invoke(this, new TicTacToeMoveEventArgs(row, column, move, gameOver));
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
					SetIndexFast(i, j, TicTacToePlayer.None);
				}
			}

			for (int i = 0; i < _field.GetLength(0) * _field.GetLength(1); i++)
			{
				Field.SetValue(MapToInt(TicTacToePlayer.None), 0, i);
			}

			OnFieldChange();
		}

		private void SetIndexFast(int row, int column, TicTacToePlayer move)
		{
			_field[row, column] = move;
			_buttons[row, column].MainContent = move.GetPlayerText();
		}

		public void SetIndex(int row, int column, TicTacToePlayer move)
		{
			if (move == NextTicTacToePlayer)
			{
				SetIndexFast(row, column, move);

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

				NextTicTacToePlayer = NextTicTacToePlayer.TogglePlayer();

				Field.SetValue(MapToInt(move), 0, row * _field.GetLength(0) + column);
				OnFieldChange(row, column, move, _gameOver);

				if (Autoplay)
				{
					if (NextTicTacToePlayer == AiTicTacToePlayer)
					{
						OnAiMove();
					}
				}
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

		private void Move_OnClick(object sender, RoutedEventArgs e)
		{
			if (NextTicTacToePlayer == AiTicTacToePlayer)
			{
				OnAiMove();
			}
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