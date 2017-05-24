using log4net;
using Sigma.Core.Monitors.WPF.View.Windows;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	/// <summary>
	/// A panel that displays a byte[] as a image (bitmap). 
	/// </summary>
	public class BitmapPanel : GenericPanel<Image>
	{
		/// <summary>
		/// The bitmap that is being displayed by this panel. Use with caution and do not change the reference
		/// </summary>
		public WriteableBitmap Bitmap;

		/// <summary>
		/// The actions that will be invoked once the bitmap has been initialised.
		/// </summary>
		protected List<Action> BitmapInitListeners;

		/// <summary>
		/// A boolean that determines whether the bitmap is already initialised.
		/// </summary>
		protected bool Initialised;

		/// <summary>
		/// The window that will be used for the on loaded action.
		/// </summary>
		private Window _window;

		private readonly int _width, _height;

		/// <summary>
		/// The loger of this panel.
		/// </summary>
		private readonly ILog _logger = LogManager.GetLogger(typeof(BitmapPanel));

		/// <summary>
		///     Create a BitmapPanel with a given title, width, and height.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="width">The width of the bitmappanel (not the actual width but the width of the data grid).</param>
		/// <param name="height">The height of the bitmappanel (not the actual height but the height of the data grid).</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public BitmapPanel(string title, int width, int height, object headerContent = null) : base(title, headerContent)
		{
			Content = new Image();
			_width = width;
			_height = height;
		}

		/// <summary>
		/// This method is called once the bitmap has been successfully initialised.
		/// Use it if a initialisation is requried.
		/// </summary>
		protected virtual void OnBitmapInitialised()
		{
		}

		/// <summary>
		/// Add an action as listener that will be invoked once the bitmap has been initialised.
		/// </summary>
		/// <param name="action">The action that will be invoked.</param>
		public void OnBitmapInitialised(Action action)
		{
			if (Initialised)
			{
				action();
			}
			else
			{
				if (BitmapInitListeners == null)
				{
					BitmapInitListeners = new List<Action>();
				}
				BitmapInitListeners.Add(action);
			}
		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected override void OnInitialise(WPFWindow window)
		{
			// the presentationsource can only be received once loaded
			if (PresentationSource.FromVisual(window) == null)
			{
				window.Loaded += OnWindowLoaded;
				_window = window;
			}
			else
			{
				InitialiseBitmap(_width, _height);
			}
		}

		/// <summary>
		/// Initialise the bitmap and unassign from all listeners. Further after this method has finished execution, it will call all attached listeners
		/// </summary>
		protected void InitialiseBitmap(int width, int height)
		{
			PresentationSource source = PresentationSource.FromVisual(_window);

			double dpiX, dpiY;
			if (source?.CompositionTarget == null)
			{
				dpiX = dpiY = 96;
				_logger.Fatal("Cannot access presentation source or composition target - default dpi of 96 is assumed.");
			}
			else
			{

				dpiX = 96.0 * source.CompositionTarget.TransformToDevice.M11;
				dpiY = 96.0 * source.CompositionTarget.TransformToDevice.M22;
			}

			Bitmap = new WriteableBitmap(width, height, dpiX, dpiY, PixelFormats.Bgra32, null);

			DrawingVisual drawingVisual = new DrawingVisual();
			using (DrawingContext context = drawingVisual.RenderOpen())
			{
				context.DrawImage(Bitmap, new Rect(0, 0, width, height));
			}

			DrawingImage drawingImage = new DrawingImage(drawingVisual.Drawing);

			Content.Source = drawingImage;

			_window = null;
			OnBitmapInitialised();

			Initialised = true;
			if (BitmapInitListeners != null)
			{
				foreach (Action listener in BitmapInitListeners) { listener(); }
				BitmapInitListeners = null;
			}
		}

		/// <summary>
		/// This method is called once the root window has been succesfully loaded. 
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		protected void OnWindowLoaded(object sender, RoutedEventArgs e)
		{
			_window.Loaded -= OnWindowLoaded;
			InitialiseBitmap(_width, _height);
		}

		/// <summary>
		/// Check whether the bitmap is already initialised (equals null).
		/// </summary>
		/// <exception cref="InvalidOperationException">If the bitmap is not yet initialised.</exception>
		protected virtual void IsBitmapLoaded()
		{
			if (Bitmap == null) throw new InvalidOperationException("The bitmap is not yet initialised.");
		}

		/// <summary>
		/// Fill the content from a bitmap with a raw byte[]. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		public void RenderRaw(byte[] data)
		{
			IsBitmapLoaded();
			Bitmap.RenderRaw(data);
		}

		/// <summary>
		/// Fill the content from a bitmap with a data[]. The data[] can be of arbitrary format and will be mapped with the provided functions.
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="red">The function that defines the red value.</param>
		/// <param name="green">The function that defines the green value.</param>
		/// <param name="blue">The function that defines the blue value.</param>
		/// <param name="alpha">The function that defines the alpha value.</param>
		public void Render<T>(T[] data, Func<T, byte> red, Func<T, byte> green, Func<T, byte> blue, Func<T, byte> alpha)
		{
			IsBitmapLoaded();
			Bitmap.Render(data, blue, green, red, alpha);
		}

		/// <summary>
		/// Fill the content from a bitmap with a raw INDarray. The INDarray has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		public void RenderRaw(INDArray data)
		{
			IsBitmapLoaded();
			Bitmap.RenderRaw(data);
		}

		/// <summary>
		/// Fill the content from a bitmap with the data from an INDarray. The data can be of arbitrary format and will be mapped with the provided functions.
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="red">The function that defines the red value.</param>
		/// <param name="green">The function that defines the green value.</param>
		/// <param name="blue">The function that defines the blue value.</param>
		/// <param name="alpha">The function that defines the alpha value.</param>
		public void Render<T>(INDArray data, Func<T, byte> red, Func<T, byte> green, Func<T, byte> blue, Func<T, byte> alpha)
		{
			IsBitmapLoaded();
			Bitmap.Render(data, blue, green, red, alpha);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="width">The width of the data[].</param>
		/// <param name="height">The height of the data[].</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		public void RenderRectangleRaw(byte[] data, int width, int height, int xOffset, int yOffset)
		{
			IsBitmapLoaded();
			Bitmap.RenderRectangleRaw(data, width, height, xOffset, yOffset);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The data[] can be of arbitrary format and will be mapped with the provided functions.
		/// </summary>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="width">The width of the data[].</param>
		/// <param name="height">The height of the data[].</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>		
		/// <param name="red">The function that defines the red value.</param>
		/// <param name="green">The function that defines the green value.</param>
		/// <param name="blue">The function that defines the blue value.</param>
		/// <param name="alpha">The function that defines the alpha value.</param>
		public void RenderRectangle<T>(T[] data, Func<T, byte> red, Func<T, byte> green, Func<T, byte> blue, Func<T, byte> alpha, int width, int height, int xOffset, int yOffset)
		{
			IsBitmapLoaded();
			Bitmap.RenderRectangle(data, width, height, xOffset, yOffset, blue, green, red, alpha);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset, where the width and height is automatically taken from the INDarrays shape. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		public void RenderRectangleRaw(INDArray data, int xOffset, int yOffset)
		{
			IsBitmapLoaded();
			Bitmap.RenderRectangleRaw(data, xOffset, yOffset);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The data[] can be of arbitrary format and will be mapped with the provided functions.
		/// </summary>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		/// <param name="red">The function that defines the red value.</param>
		/// <param name="green">The function that defines the green value.</param>
		/// <param name="blue">The function that defines the blue value.</param>
		/// <param name="alpha">The function that defines the alpha value.</param>
		public void RenderRectangle<T>(INDArray data, Func<T, byte> red, Func<T, byte> green, Func<T, byte> blue, Func<T, byte> alpha, int xOffset, int yOffset)
		{
			IsBitmapLoaded();
			Bitmap.RenderRectangle(data, xOffset, yOffset, blue, green, red, alpha);
		}
	}
}