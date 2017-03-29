/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility methods for using registries and registry resolvers. 
	/// </summary>
	public static class RegistryUtils
	{
		/// <summary>
		/// Compare two identifiers and get an integer indicating the difference in their specificity. 
		/// The comparison is in ascending order of specificity, least specific come first, most specific come last.
		/// Note:   On the behaviour of the comparison function with wildcards and tagging: 
		/// 		- Deeper, more nested identifiers are considered more specific. 
		///			- Identifiers containing wildcards are considered less specific.
		///			- Identifiers containing tags are considered more specific than untagged ones.
		///			- Identifiers with wildcards and different amounts of tags or different individual tags are considered equal.
		/// </summary>
		/// <param name="identifier1">The first identifier.</param>
		/// <param name="identifier2">The first identifier.</param>
		/// <returns>An integer indicating the difference in the specificity of two identifiers.</returns>
		public static int CompareIdentifierSpecificityAscending(string identifier1, string identifier2)
		{
			if (identifier1 == null) throw new ArgumentNullException(nameof(identifier1));
			if (identifier2 == null) throw new ArgumentNullException(nameof(identifier2));

			int nestedDepth1 = _InternalGetCharCountInString(identifier1, '.');
			int nestedDepth2 = _InternalGetCharCountInString(identifier2, '.');

			if (nestedDepth1 != nestedDepth2)
			{
				return nestedDepth1 - nestedDepth2;
			}

			string[] parts1 = identifier1.Split('.'), parts2 = identifier2.Split('.');

			for (int i = 0; i < parts1.Length; i++)
			{
				int numSpecificChars1 = parts1[i].Replace("*", "").Length, numSpecificChars2 = parts2[i].Replace("*", "").Length;

				if (numSpecificChars1 != numSpecificChars2)
				{
					return numSpecificChars1 - numSpecificChars2;
				}
			}

			return 0;
		}

		private static int _InternalGetCharCountInString(string @string, char character)
		{
			int count = 0;

			for (int i = 0; i < @string.Length; i++)
			{
				if (@string[i] == character)
				{
					count++;
				}
			}

			return count;
		}

		/// <summary>
		/// Get the deepest copy of a given value (order: <see cref="IDeepCopyable.DeepCopy"/> => <see cref="ICloneable.Clone"/>).
		/// If the value cannot be copied, the original value is returned.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <returns>The deepest available copy of the given value.</returns>
		public static object DeepestCopy(object value)
		{
			object copiedValue;
			IDeepCopyable deepCopyableValue = value as IDeepCopyable;

			if (deepCopyableValue == null)
			{
				copiedValue = (value as ICloneable)?.Clone() ?? value;
			}
			else
			{
				copiedValue = deepCopyableValue.DeepCopy();
			}

			return copiedValue;
		}
	}
}
