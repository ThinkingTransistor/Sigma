/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Math;
using System;
using System.Collections.Generic;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of array utility functions. 
	/// </summary>
	public static class ArrayUtils
	{
		/// <summary>
		/// The product of an integer array (i.e. all values multiplied with each other).
		/// </summary>
		/// <param name="array">The array.</param>
		/// <returns>The product of the given array.</returns>
		public static long Product(params long[] array)
		{
			long product = 1L;

			for (int i = 0; i < array.Length; i++)
			{
				product *= array[i];
			}

			return product;
		}

		/// <summary>
		/// Add two arrays of equal length and get the resulting summed array of same length.
		/// </summary>
		/// <param name="a">The first array.</param>
		/// <param name="b">The second array.</param>
		/// <param name="c">The optional result array which will be returned. If null or not large enough, a new array will be created.</param>
		/// <returns>An array representing the sum of a and b.</returns>
		public static long[] Add(long[] a, long[] b, long[] c = null)
		{
			if (a.Length != b.Length)
			{
				throw new ArgumentException($"Arrays to be added must be of same length, but first array a was of size {a.Length} and b {b.Length}.");
			}

			if (c == null || c.Length < a.Length)
			{
				c = new long[a.Length];
			}

			for (int i = 0; i < a.Length; i++)
			{
				c[i] = a[i] + b[i];
			}

			return c;
		}

		/// <summary>
		/// Get a certain range of values with a certain step size. 
		/// Note: Ranges work both ways; start can be larger than end and the range will be returned in reverse. 
		/// </summary>
		/// <param name="start">The start of the range.</param>
		/// <param name="end">The end of the range.</param>
		/// <param name="stepSize">The step size (the gap between values).</param>
		/// <returns>An array of values over a given range and given step size.</returns>
		public static int[] Range(int start, int end, int stepSize = 1)
		{
			if (stepSize <= 0)
			{
				throw new ArgumentException($"Step size must be >= 1, but step size was {stepSize} (swap start and end for a reversed range).");
			}

			int span = System.Math.Abs(end - start) + 1;
			int length = (int) System.Math.Ceiling((double) span / stepSize);
			int[] result = new int[length];

			if (end > start)
			{
				int i = 0;
				for (int value = start; value <= end && i < length; value += stepSize)
				{
					result[i++] = value;
				}
			}
			else if (end < start)
			{
				int i = 0;
				for (int value = start; value >= end && i < length; value -= stepSize)
				{
					result[i++] = value;
				}
			}

			return result;
		}

		/// <summary>
		/// Get a permuted COPY of an array according to the rearranged dimension array (the permutation array).
		/// </summary>
		/// <param name="array">The array to get a permuted copy of.</param>
		/// <param name="rearrangedDimensions">The permutation array, how each dimension should be rearranged.</param>
		/// <returns>A permuted copy of the given array according to the permutation array (rearrangedDimensions).</returns>
		public static long[] PermuteArray(long[] array, int[] rearrangedDimensions)
		{
			int length = array.Length;
			long[] result = new long[length];

			for (int i = 0; i < length; i++)
			{
				result[i] = array[rearrangedDimensions[i]];
			}

			return result;
		}

		/// <summary>
		/// Get the flattened column mappings corresponding to a certain higher level column mapping.
		/// </summary>
		/// <param name="columnMappings">The higher level column mappings.</param>
		/// <returns>The corresponding flattened column mappings.</returns>
		public static Dictionary<string, IList<int>> GetFlatColumnMappings(Dictionary<string, int[][]> columnMappings)
		{
			if (columnMappings == null)
			{
				throw new ArgumentNullException("Column mappings cannot be null.");
			}

			Dictionary<string, IList<int>> flatMappings = new Dictionary<string, IList<int>>();

			foreach (string key in columnMappings.Keys)
			{
				flatMappings.Add(key, GetFlatColumnMappings(columnMappings[key]));
			}

			return flatMappings;
		}

		/// <summary>
		/// Get the flattened column mappings corresponding to a certain higher level column mapping.
		/// </summary>
		/// <param name="columnMappings">The higher level column mappings.</param>
		/// <returns>The corresponding flattened column mappings.</returns>
		public static IList<int> GetFlatColumnMappings(int[][] columnMappings)
		{
			if (columnMappings == null)
			{
				throw new ArgumentNullException("Column mappings cannot be null.");
			}

			IList<int> individualMappings = new List<int>();

			for (int i = 0; i < columnMappings.Length; i++)
			{
				if (columnMappings[i].Length == 0)
				{
					throw new ArgumentException($"Column mappings cannot be empty, but mapping at index {i} was empty (length: 0).");
				}

				if (columnMappings[i].Length == 1)
				{
					individualMappings.Add(columnMappings[i][0]);
				}
				else if (columnMappings[i].Length > 2)
				{
					throw new ArgumentException($"Column mappings can only be assigned in pairs, but mapping at index{1} was of length {columnMappings[i].Length}");
				}
				else
				{
					foreach (int y in ArrayUtils.Range(columnMappings[i][0], columnMappings[i][1]))
					{
						individualMappings.Add(y);
					}
				}
			}

			return individualMappings;
		}

		/// <summary>
		/// Maps a list of values into a dictionary, where each value is mapped to the order in the given list.
		/// </summary>
		/// <typeparam name="T">The value type.</typeparam>
		/// <param name="values">The values.</param>
		/// <returns>A dictionary mapping of values and their order in the given values list.</returns>
		public static Dictionary<T, object> MapToOrder<T>(IEnumerable<T> values)
		{
			Dictionary<T, object> mapping = new Dictionary<T, object>();

			int index = 0;
			foreach (T value in values)
			{
				mapping.Add(value, index++);
			}

			return mapping;
		}

		/// <summary>
		/// Get a sub array from a certain array within a certain range.
		/// </summary>
		/// <typeparam name="T">The type of the array elements.</typeparam>
		/// <param name="array">The array to slice.</param>
		/// <param name="startIndex">The start index of the subarray in the given array.</param>
		/// <param name="length">The length of the sub array, starting at the start index, in the given array.</param>
		/// <returns>A sub array of the given array within the given range of the given type.</returns>
		public static T[] SubArray<T>(this T[] array, int startIndex, int length)
		{
			T[] subArray = new T[length];

			Array.Copy(array, startIndex, subArray, 0, length);

			return subArray;
		}

		/// <summary>
		/// A helper function to call the toString method of the default NDArray<T> implementation.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="array"></param>
		/// <param name="toStringElement"></param>
		/// <param name="maxDimensionNewLine"></param>
		/// <returns></returns>
		public static string ToString<T>(INDArray array, NDArray<T>.ToStringElement toStringElement = null, int maxDimensionNewLine = 1)
		{
			return ((NDArray<T>) array).ToString(toStringElement, maxDimensionNewLine);
		}

		/// <summary>
		/// Extension ToString method for all enumerables, because it for something reason is not included in the default base class.
		/// </summary>
		/// <param name="array">The enumerable to string.</param>
		/// <returns>A string representing the enumerable.</returns>
		public static string ToString<T>(this IEnumerable<T> array)
		{
			return "[" + string.Join(", ", array) + "]";
		}

		/// <summary>
		/// Extended ToString method for two-dimensional arrays of any type.
		/// </summary>
		/// <typeparam name="T">The type.</typeparam>
		/// <param name="array">The two-dimensional array.</param>
		/// <returns>A string representing the contents of the given array.</returns>
		public static string ToString<T>(T[][] array)
		{
			string[] subarrays = new string[array.Length];

			for (int i = 0; i < subarrays.Length; i++)
			{
				subarrays[i] = "[" + string.Join(", ", array[i]) + "]";
			}

			return "[" + string.Join(",\n", subarrays) + "]";
		}
	}
}
