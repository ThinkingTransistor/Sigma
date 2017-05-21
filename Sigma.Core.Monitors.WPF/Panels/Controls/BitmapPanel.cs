
using log4net;
using Sigma.Core.Monitors.WPF.View.Windows;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

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
				InitBitmap();
			}
		}

		/// <summary>
		/// Initialise the bitmap and unassign from all listeners. Further after this method has finished execution, it will call all attached listeners
		/// </summary>
		protected void InitBitmap()
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

			Bitmap = new WriteableBitmap(_width, _height, dpiX, dpiY, PixelFormats.Bgra32, null);

			DrawingVisual drawingVisual = new DrawingVisual();
			using (DrawingContext context = drawingVisual.RenderOpen())
			{
				context.DrawImage(Bitmap, new Rect(0, 0, _width, _height));
			}

			DrawingImage drawingImage = new DrawingImage(drawingVisual.Drawing);

			Content.Source = drawingImage;

			_window = null;
			OnBitmapInitialised();

			Initialised = true;
			foreach (Action listener in BitmapInitListeners) { listener(); }
			BitmapInitListeners = null;
		}

		/// <summary>
		/// This method is called once the bitmap has been successfully initialised.
		/// Use it if a initialisation is requried.
		/// </summary>
		protected virtual void OnBitmapInitialised()
		{
			//byte[] pixels = new byte[Bitmap.PixelHeight * Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8];
			//for (int i = 0; i < 768; i++)
			//{
			//	int pos = i * 4;
			//	int pos2 = pos + 768 * 767 * 4;
			//	pixels[pos] = 0xff;
			//	pixels[pos + 3] = 0xff;

			//	pixels[pos2 + 2] = 0xff;
			//	pixels[pos2 + 3] = 0xff;
			//}
			//Random rand = new Random();
			//rand.NextBytes(pixels);
			//Render(pixels);
		}

		/// <summary>
		/// This method is called once the root window has been succesfully loaded. 
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		protected void OnWindowLoaded(object sender, RoutedEventArgs e)
		{
			_window.Loaded -= OnWindowLoaded;
			InitBitmap();
		}

		/// <summary>
		/// Render a given byte array of data (the bitmap has to be initialised).
		/// The data has to be of the length Bitmap.PixelHeight * Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8.
		/// </summary>
		/// <param name="data">The new data the image will contain.</param>
		public void Render(byte[] data)
		{
			if (Bitmap == null)
			{
				throw new InvalidOperationException("The bitmap is not yet initialised.");
			}
			if (data.Length != Bitmap.PixelHeight * Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8)
			{
				throw new ArgumentException(nameof(data));
			}

			Bitmap.WritePixels(new Int32Rect(0, 0, Bitmap.PixelWidth, Bitmap.PixelHeight), data, Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8, 0);
		}
	}
}