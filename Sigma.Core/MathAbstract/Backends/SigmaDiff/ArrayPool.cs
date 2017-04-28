/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff
{
    /// <summary>
    /// An array pool for pooling arrays.
    /// Duh.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ArrayPool<T>
    {
        private readonly IDictionary<int, IList<T[]>> _availableArrays;

        /// <summary>
        /// Create a new array pool.
        /// </summary>
        public ArrayPool()
        {
            _availableArrays = new Dictionary<int, IList<T[]>>();
        }

        /// <summary>
        /// Allocate an array of a certain size from this array pool.
        /// </summary>
        /// <param name="arraySize">The array size.</param>
        /// <returns>An array of the given size.</returns>
        public T[] Allocate(int arraySize)
        {
            if (!_availableArrays.ContainsKey(arraySize))
            {
                return new T[arraySize];
            }

            IList<T[]> pooledArrays = _availableArrays[arraySize];
            int lastIndex = pooledArrays.Count - 1;

            T[] lastPooledArray = pooledArrays[lastIndex];

            pooledArrays.RemoveAt(lastIndex);

            return lastPooledArray;
        }

        /// <summary>
        /// Free a specific array allocated with this 
        /// </summary>
        /// <param name="array"></param>
        public void Free(T[] array)
        {
            // TODO what happens if the same array is freed multiple times? check list? but that's slower than checking a set... but a set doesn't have an item order...
            _availableArrays.TryGetValue(array.Length, () => new List<T[]>()).Add(array);
        }

        /// <summary>
        /// Free all pooled arrays.
        /// </summary>
        public void FreeAll()
        {
            _availableArrays.Clear();
            // now get to work GC
        }
    }
}
