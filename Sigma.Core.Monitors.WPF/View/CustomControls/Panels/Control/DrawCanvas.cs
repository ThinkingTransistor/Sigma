using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Point = System.Windows.Point;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control
{
	public class DrawCanvas : Canvas, IDisposable
	{

		public delegate void InputChangedEventHandler(DrawCanvas canvas);

		public event InputChangedEventHandler InputChangedEvent;

		/// <summary>
		/// The colour that is used as drawing colour
		/// </summary>
		public Brush DrawColour
		{
			get { return (Brush) GetValue(DrawColourProperty); }
			set { SetValue(DrawColourProperty, value); }
		}

		/// <summary>
		/// The dependency property for <see ref="DrawColour"/>.
		/// </summary>
		public static readonly DependencyProperty DrawColourProperty =
			DependencyProperty.Register("DrawColour", typeof(Brush), typeof(DrawCanvas), new PropertyMetadata(Brushes.Black));

		public int GridWidth
		{
			get { return (int) GetValue(GridWidthProperty); }
			set
			{
				SetValue(GridWidthProperty, value);
				Width = value;
			}
		}

		// Using a DependencyProperty as the backing store for GridWidth.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty GridWidthProperty =
			DependencyProperty.Register("GridWidth", typeof(int), typeof(DrawCanvas), new PropertyMetadata(0));

		public int GridHeight
		{
			get { return (int) GetValue(GridHeightProperty); }
			set
			{
				SetValue(GridHeightProperty, value);
				Height = value;
			}
		}

		// Using a DependencyProperty as the backing store for GridHeight.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty GridHeightProperty =
			DependencyProperty.Register("GridHeight", typeof(int), typeof(DrawCanvas), new PropertyMetadata(0));

		public int PointSize
		{
			get { return (int) GetValue(PointSizeProperty); }
			set { SetValue(PointSizeProperty, value); }
		}

		// Using a DependencyProperty as the backing store for PointSize.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty PointSizeProperty =
			DependencyProperty.Register("PointSize", typeof(int), typeof(DrawCanvas), new PropertyMetadata(0));

		public bool SoftDrawing
		{
			get { return (bool) GetValue(SoftDrawingProperty); }
			set { SetValue(SoftDrawingProperty, value); }
		}

		// Using a DependencyProperty as the backing store for SoftDrawing.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty SoftDrawingProperty =
			DependencyProperty.Register("SoftDrawing", typeof(bool), typeof(DrawCanvas), new PropertyMetadata(true));

		public double SoftFactor
		{
			get { return (double) GetValue(SoftFactorProperty); }
			set { SetValue(SoftFactorProperty, value); }
		}

		// Using a DependencyProperty as the backing store for SoftFactor.  This enables animation, styling, binding, etc...
		public static readonly DependencyProperty SoftFactorProperty =
			DependencyProperty.Register("SoftFactor", typeof(double), typeof(DrawCanvas), new PropertyMetadata(0.125));

		private Point _currentPoint;

		private Rectangle[,] _rects;

		public DrawCanvas()
		{
			Background = Brushes.White;

			MouseDown += Canvas_MouseDown;
			MouseMove += Canvas_MouseMove;
		}

		#region RectangleBoundries

		private bool _drawing;

		private void Canvas_MouseDown(object sender, MouseButtonEventArgs e)
		{
			if (e.LeftButton == MouseButtonState.Pressed)
			{
				_currentPoint = e.GetPosition(this);
				_drawing = true;
			}
			else if (e.RightButton == MouseButtonState.Pressed)
			{
				_currentPoint = e.GetPosition(this);
				_drawing = false;
			}
		}

		private void Canvas_MouseMove(object sender, MouseEventArgs e)
		{
			if (e.LeftButton == MouseButtonState.Pressed || e.RightButton == MouseButtonState.Pressed)
			{
				int opacity = _drawing ? 1 : 0;
				int factor = _drawing ? 1 : -1;
				Brush fill = _drawing ? DrawColour : null;

				LineSegment line = new LineSegment
				{
					X1 = _currentPoint.X,
					Y1 = _currentPoint.Y,
					X2 = e.GetPosition(this).X,
					Y2 = e.GetPosition(this).Y
				};

				for (int row = 0; row < _rects.GetLength(0); row++)
				{
					for (int column = 0; column < _rects.GetLength(1); column++)
					{
						Rectangle rect = _rects[row, column];
						double x = GetLeft(rect);
						double y = GetTop(rect);

						if (RectIntersectsLine(new Rect(x, y, rect.Width, rect.Height), line))
						{
							rect.Opacity = opacity;
							rect.Fill = fill;
							if (SoftDrawing)
							{
								UpdateNeighbours(GetNeighbours(_rects, row, column), factor, fill);
							}
						}
					}

					InputChangedEvent?.Invoke(this);
				}

				_currentPoint = e.GetPosition(this);
			}
		}

		private Rectangle[] GetNeighbours(Rectangle[,] rectangles, int row, int column)
		{
			int height = rectangles.GetLength(0);
			int width = rectangles.GetLength(1);

			Rectangle[] rects = new Rectangle[8];

			// cells above
			if (row > 0)
			{
				// cell topleft
				if (column > 0)
				{
					rects[0] = rectangles[row - 1, column - 1];
				}

				// cell top
				rects[1] = rectangles[row - 1, column];

				// cell topright
				if (column + 1 < width)
				{
					rects[2] = rectangles[row - 1, column + 1];
				}
			}

			// cell right
			if (column + 1 < width)
			{
				rects[3] = rectangles[row, column + 1];
			}

			if (row + 1 < height)
			{
				// cell bottom right
				if (column + 1 < width)
				{
					rects[4] = rectangles[row + 1, column + 1];
				}

				// cell bottom
				rects[5] = rectangles[row + 1, column];

				if (column > 0)
				{
					rects[6] = rectangles[row + 1, column - 1];
				}
			}

			if (column > 0)
			{
				rects[7] = rectangles[row, column - 1];
			}

			return rects;
		}

		private void UpdateNeighbours(Rectangle[] neighbours, int factor, Brush fill)
		{
			if (neighbours.Length != 8) throw new ArgumentException("May only contain 8 elements.", nameof(neighbours));

			for (int i = 0; i < neighbours.Length; i++)
			{
				if (neighbours[i] != null)
				{
					double currentSoft = SoftFactor;

					neighbours[i].Fill = fill;

					if (i % 2 == 0)
					{
						currentSoft *= currentSoft;
					}

					neighbours[i].Opacity = Math.Max(0, Math.Min(neighbours[i].Opacity + currentSoft * factor, 1));
				}
			}
		}

		public void UpdateRects()
		{
			GenerateRects();
		}

		private void GenerateRects()
		{
			_rects = new Rectangle[GridHeight / PointSize, GridWidth / PointSize];
			// width
			for (int x = 0; x < GridWidth; x += PointSize)
			{
				// height
				for (int y = 0; y < GridHeight; y += PointSize)
				{
					Rectangle rect = new Rectangle
					{
						Opacity = 0,
						Width = Math.Min(PointSize, GridWidth - x),
						Height = Math.Min(PointSize, GridHeight - y),
					};

					SetLeft(rect, x);
					SetTop(rect, y);

					_rects[y / PointSize, x / PointSize] = rect;
					Children.Add(rect);
				}
			}
		}

		private class LineSegment
		{
			public double X1 { get; set; }
			public double X2 { get; set; }
			public double Y1 { get; set; }
			public double Y2 { get; set; }
		}

		private static bool SegmentsIntersect(LineSegment a, LineSegment b)
		{
			double x1 = a.X1, x2 = a.X2, x3 = b.X1, x4 = b.X2;
			double y1 = a.Y1, y2 = a.Y2, y3 = b.Y1, y4 = b.Y2;

			double denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);

			if (denominator == 0)
			{
				return false;
			}

			double ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator;
			double ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator;

			return (ua > 0 && ua < 1 && ub > 0 && ub < 1);
		}

		private static bool RectIntersectsLine(Rect a, LineSegment b)
		{
			return (SegmentsIntersect(b, new LineSegment { X1 = a.X, Y1 = a.Y, X2 = a.X, Y2 = a.Y + a.Height }) ||
				SegmentsIntersect(b, new LineSegment { X1 = a.X, Y1 = a.Y + a.Height, X2 = a.X + a.Width, Y2 = a.Y + a.Height }) ||
				SegmentsIntersect(b, new LineSegment { X1 = a.X + a.Width, Y1 = a.Y + a.Height, X2 = a.X + a.Width, Y2 = a.Y }) ||
				SegmentsIntersect(b, new LineSegment { X1 = a.X + a.Width, Y1 = a.Y, X2 = a.X, Y2 = a.Y }) ||
				RectContainsPoint(a, new Point(b.X1, b.Y1)) ||
				RectContainsPoint(a, new Point(b.X2, b.Y2)));
		}

		private static bool RectContainsPoint(Rect a, Point b)
		{
			return b.X > a.X && b.X < a.X + a.Width && b.Y > a.Y && b.Y < a.Y + a.Height;
		}

		#endregion RectangleBoundries

		public void Clear()
		{
			if (_rects != null)
			{
				foreach (Rectangle rectangle in _rects)
				{
					rectangle.Opacity = 0;
					rectangle.Fill = null;
				}
			}
		}

		/// <summary>
		/// The values of the panels normalised between [0;1]
		/// </summary>
		/// <returns>The normalised values.</returns>
		public double[,] GetValues()
		{
			if (_rects == null)
			{
				return null;
			}

			double[,] values = new double[_rects.GetLength(0), _rects.GetLength(1)];

			for (int i = 0; i < _rects.GetLength(0); i++)
			{
				for (int j = 0; j < _rects.GetLength(1); j++)
				{
					values[i, j] = _rects[i, j].Opacity;
				}
			}

			return values;
		}

		/// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
		public void Dispose()
		{
			MouseDown -= Canvas_MouseDown;
			MouseMove -= Canvas_MouseMove;
		}
	}
}