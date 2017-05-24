using System;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Monitors.WPF.Utils
{
	/// <summary>
	/// An utils class that provides an easy way to draw to the bitmap. 
	/// </summary>
	public static class WriteableBitmapUtils
	{
		#region Render

		/// <summary>
		/// Render a stream of data with an xOffset. The data[] can be of arbitrary format and will be mapped with the provided functions. (see <see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void RenderStream<T>(this WriteableBitmap bitmap, T[] data, int xOffset, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.RenderStreamRaw(ApplyTransformation(data, transformationFuncs), xOffset);
		}

		/// <summary>
		/// Render a stream of data with an xOffset. The data can be of arbitrary format and will be mapped with the provided functions. (see <see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void RenderStream<T>(this WriteableBitmap bitmap, INDArray data, int xOffset, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.RenderStreamRaw(ApplyTransformation(data.GetDataAs<T>().ToArray(), transformationFuncs), xOffset);
		}

		/// <summary>
		/// Fill the content from a bitmap with a data[]. The data[] can be of arbitrary format and will be mapped with the provided functions. (see <see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void Render<T>(this WriteableBitmap bitmap, T[] data, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.RenderRaw(ApplyTransformation(data, transformationFuncs));
		}

		/// <summary>
		/// Fill the content from a bitmap with the data from an INDarray. The data can be of arbitrary format and will be mapped with the provided functions. (see <see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void Render<T>(this WriteableBitmap bitmap, INDArray data, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.Render(data.GetDataAs<T>().ToArray(), transformationFuncs);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The data[] can be of arbitrary format and will be mapped with the provided functions. (see<see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="width">The width of the data[].</param>
		/// <param name="height">The height of the data[].</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void RenderRectangle<T>(this WriteableBitmap bitmap, T[] data, int width, int height, int xOffset, int yOffset, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.RenderRectangleRaw(ApplyTransformation(data, transformationFuncs), width, height, xOffset, yOffset);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The data[] can be of arbitrary format and will be mapped with the provided functions. (see<see cref="ApplyTransformation{T}"/> for mor details).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		/// <param name="transformationFuncs">The functions that map the data to bytes.</param>
		public static void RenderRectangle<T>(this WriteableBitmap bitmap, INDArray data, int xOffset, int yOffset, params Func<T, byte>[] transformationFuncs)
		{
			bitmap.RenderRectangle(data.GetDataAs<T>().ToArray(), (int)data.Shape[1], (int)data.Shape[0], xOffset, yOffset, transformationFuncs);
		}

		#endregion Render

		#region Raw

		/// <summary>
		/// Render a stream of bytes with an xOffset.
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="xOffset">The x-Offset</param>
		public static void RenderStreamRaw(this WriteableBitmap bitmap, byte[] data, int xOffset)
		{
			if (data == null) throw new ArgumentNullException(nameof(data));

			bitmap.WritePixels(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight), data, bitmap.PixelWidth * bitmap.Format.BitsPerPixel / 8, xOffset);
		}

		/// <summary>
		/// Render a stream of bytes with an xOffset.
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		/// <param name="xOffset">The x-Offset</param>
		public static void RenderStreamRaw(this WriteableBitmap bitmap, INDArray data, int xOffset)
		{
			bitmap.RenderStreamRaw(data.GetDataAs<byte>().ToArray(), xOffset);
		}

		/// <summary>
		/// Fill the content from a bitmap with a raw byte[]. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		public static void RenderRaw(this WriteableBitmap bitmap, byte[] data)
		{
			bitmap.RenderRectangleRaw(data, bitmap.PixelWidth, bitmap.PixelHeight, 0, 0);
		}

		/// <summary>
		/// Fill the content from a bitmap with a raw INDarray. The INDarray has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be written to the new bitmap.</param>
		public static void RenderRaw(this WriteableBitmap bitmap, INDArray data)
		{
			bitmap.RenderRaw(data.GetDataAs<byte>().ToArray());
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset and a given width and height. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="width">The width of the data[].</param>
		/// <param name="height">The height of the data[].</param>
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		public static void RenderRectangleRaw(this WriteableBitmap bitmap, byte[] data, int width, int height, int xOffset, int yOffset)
		{
			if (bitmap == null) throw new ArgumentNullException(nameof(bitmap));

			bitmap.WritePixels(new Int32Rect(xOffset, yOffset, width, height), data, width * bitmap.Format.BitsPerPixel / 8, 0);
		}

		/// <summary>
		/// Render bytes as a rectangle with a x- and y-offset, where the width and height is automatically taken from the INDarrays shape. The byte[] has to contain all data (so rgb etc).
		/// </summary>
		/// <param name="bitmap">The bitmap that will be written to.</param>
		/// <param name="data">The data that will be rendered.</param>		
		/// <param name="xOffset">The x-offset.</param>
		/// <param name="yOffset">The y-offste.</param>
		public static void RenderRectangleRaw(this WriteableBitmap bitmap, INDArray data, int xOffset, int yOffset)
		{
			bitmap.RenderRectangleRaw(data.GetDataAs<byte>().ToArray(), (int)data.Shape[1], (int)data.Shape[0], xOffset, yOffset);
		}

		/// <summary>
		/// This method transforms a given data array to a byte array. Each passed data element will be split into transformationFuncs.Length amount of bytes. 
		/// </summary>
		/// <typeparam name="T">The type of the data.</typeparam>
		/// <param name="data">The data that will be transformed.</param>
		/// <param name="transformationFuncs">The function that transforms the data.</param>
		/// <exception cref="ArgumentNullException">If any of the passed arguments is null.</exception>
		/// <returns>The resulting byte[] (transformation).</returns>
		public static byte[] ApplyTransformation<T>(T[] data, params Func<T, byte>[] transformationFuncs)
		{
			if (data == null) throw new ArgumentNullException(nameof(data));
			if (transformationFuncs == null) throw new ArgumentNullException(nameof(transformationFuncs));
			if (transformationFuncs.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(transformationFuncs));

			byte[] transformedData = new byte[data.Length * transformationFuncs.Length];
			for (int i = 0; i < data.Length; i++)
			{
				int pos = i * transformationFuncs.Length;
				for (int j = 0; j < transformationFuncs.Length; j++)
				{
					transformedData[pos + j] = transformationFuncs[j](data[i]);
				}
			}

			return transformedData;
		}

		#endregion Raw
	}
}