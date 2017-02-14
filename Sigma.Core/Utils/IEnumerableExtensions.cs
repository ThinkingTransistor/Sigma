using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Utils
{
    // ReSharper disable once InconsistentNaming
    public static class IEnumerableExtensions
    {
        public static int IndexOf<T>(this IEnumerable<T> source, T item)
        {
            var entry = source
                        .Select((x, i) => new { Value = x, Index = i })
                        .FirstOrDefault(x => Equals(x.Value, item));
            return entry?.Index ?? -1;
        }

        public static void CopyTo<T>(this IEnumerable<T> source, T[] array, int startIndex)
        {
            int lowerBound = array.GetLowerBound(0);
            int upperBound = array.GetUpperBound(0);

            if (startIndex < lowerBound)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "The start index must be greater than or equal to the array lower bound");
            if (startIndex > upperBound)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "The start index must be less than or equal to the array upper bound");

            int i = 0;
            foreach (var item in source)
            {
                if (startIndex + i > upperBound)
                    throw new ArgumentException("The array capacity is insufficient to copy all items from the source sequence");
                array[startIndex + i] = item;
                i++;
            }
        }
    }
}