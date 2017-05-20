using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Shapes;

/// <summary>
/// A canvas that can visualise a 2d array of rectangles.
/// </summary>
public class RectangleCanvas : Canvas
{
	/// <summary>
	/// This variable specifies the width of the canvas and therefore the amount of rectangles that can be visualised. Once modified use update rects to apply the change.
	/// </summary>
	public int GridWidth
	{
		get => (int)GetValue(GridWidthProperty);
		set
		{
			SetValue(GridWidthProperty, value);
			Width = value;
		}
	}

	public static readonly DependencyProperty GridWidthProperty =
		DependencyProperty.Register("GridWidth", typeof(int), typeof(RectangleCanvas), new PropertyMetadata(0));

	/// <summary>
	/// This variable specifies the height of the canvas and therefore the amount of rectangles that can be visualised. Once modified use update rects to apply the change.
	/// </summary>
	public int GridHeight
	{
		get => (int)GetValue(GridHeightProperty);
		set
		{
			SetValue(GridHeightProperty, value);
			Height = value;
		}
	}

	public static readonly DependencyProperty GridHeightProperty =
		DependencyProperty.Register("GridHeight", typeof(int), typeof(RectangleCanvas), new PropertyMetadata(0));

	/// <summary>
	/// This variable specifies the pixel size of each rect. Once modified use update rects to apply the change.
	/// </summary>
	public int PointSize
	{
		get { return (int)GetValue(PointSizeProperty); }
		set { SetValue(PointSizeProperty, value); }
	}


	public static readonly DependencyProperty PointSizeProperty =
		DependencyProperty.Register("PointSize", typeof(int), typeof(RectangleCanvas), new PropertyMetadata(0));

	public Rectangle[,] Rectangles;

	/// <summary>
	/// This method has to be called in order to commit the change from a height / width change
	/// </summary>
	public void UpdateRects()
	{
		GenerateRects();
	}

	protected void GenerateRects()
	{
		Rectangles = new Rectangle[GridHeight / PointSize, GridWidth / PointSize];
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

				Rectangles[y / PointSize, x / PointSize] = rect;
				Children.Add(rect);
			}
		}
	}

	protected Rectangle[] GetNeighbours(Rectangle[,] rectangles, int row, int column)
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

	public void Clear()
	{
		if (Rectangles != null)
		{
			foreach (Rectangle rectangle in Rectangles)
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
		if (Rectangles == null)
		{
			return null;
		}

		double[,] values = new double[Rectangles.GetLength(0), Rectangles.GetLength(1)];

		for (int i = 0; i < Rectangles.GetLength(0); i++)
		{
			for (int j = 0; j < Rectangles.GetLength(1); j++)
			{
				values[i, j] = Rectangles[i, j].Opacity;
			}
		}

		return values;
	}
}