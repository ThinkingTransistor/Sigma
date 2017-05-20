
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	public class BitmapPanel : GenericPanel<Image>
	{
		protected readonly byte[] Pixels;

		public readonly WriteableBitmap Bitmap;

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public BitmapPanel(string title, int width, int height, object headerContent = null) : base(title, headerContent)
		{
			Content = new Image();
			Content.Loaded += OnImageLoaded;
			Bitmap = new WriteableBitmap(width, height, width, height, PixelFormats.Bgra32, null);
			Pixels = new byte[Bitmap.PixelHeight * Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8];

			Pixels[0] = 0xff;
			Pixels[1] = 0x00;
			Pixels[2] = 0x00;
			Pixels[3] = 0xff;

			//http://www.i-programmer.info/programming/wpf-workings/527-writeablebitmap.html?start=1
			Bitmap.WritePixels(new Int32Rect(0, 0, Bitmap.PixelWidth, Bitmap.PixelHeight), Pixels, Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8, 0);

			DrawingVisual drawingVisual = new DrawingVisual();
			using (DrawingContext context = drawingVisual.RenderOpen())
			{
				context.DrawImage(Bitmap, new Rect(0, 0, width, height));
			}

			DrawingImage drawingImage = new DrawingImage(drawingVisual.Drawing);

			Content.Source = drawingImage;
		}

		protected void OnImageLoaded(object sender, RoutedEventArgs e)
		{
			Content.Loaded -= OnImageLoaded;

			PresentationSource source = PresentationSource.FromVisual(Content);

			double dpiX, dpiY;
			if (source != null)
			{
				dpiX = 96.0 * source.CompositionTarget.TransformToDevice.M11;
				dpiY = 96.0 * source.CompositionTarget.TransformToDevice.M22;
			}
		}


		public void Render(byte[] data)
		{
			if (data.Length != Bitmap.PixelHeight * Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8)
			{
				throw new ArgumentException(nameof(data));
			}

			Bitmap.WritePixels(new Int32Rect(0, 0, Bitmap.PixelWidth, Bitmap.PixelHeight), data, Bitmap.PixelWidth * Bitmap.Format.BitsPerPixel / 8, 0);
		}
	}
}