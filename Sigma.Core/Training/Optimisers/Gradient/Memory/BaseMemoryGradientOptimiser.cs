/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;

namespace Sigma.Core.Training.Optimisers.Gradient.Memory
{
    /// <summary>
    /// A base memory gradient optimiser for implementing gradient optimisers that require some form of memory (e.g. over past gradients, param updates).
    /// </summary>
    /// <typeparam name="TMemory">The type of the value that should be memorised.</typeparam>
    [Serializable]
    public abstract class BaseMemoryGradientOptimiser<TMemory> : BaseGradientOptimiser
    {
        private readonly string _memoryIdentifier;

        /// <summary>
        /// Create a base memory gradient optimiser with an optional external output cost alias to use. 
        /// </summary>
        /// <param name="memoryIdentifier">The memory identifier (useful for parametrisation, so be somewhat descriptive).</param>
        /// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
        protected BaseMemoryGradientOptimiser(string memoryIdentifier, string externalCostAlias = "external_cost") : base(externalCostAlias)
        {
            if (memoryIdentifier == null) throw new ArgumentNullException(nameof(memoryIdentifier));

            _memoryIdentifier = memoryIdentifier;

            var memory = new Dictionary<string, TMemory>();
            Registry.Set(memoryIdentifier, memory, memory.GetType());
        }

        /// <summary>
        /// Get the memory entry for a certain parameter identifier, use default value if it doesn't exist.
        /// </summary>
        /// <param name="paramIdentifier">The parameter identifier.</param>
        /// <param name="defaultValue">The default value.</param>
        /// <returns>The memorised value with the given param identifier or the given default value if it isn't memorised.</returns>
        protected TMemory GetMemory(string paramIdentifier, TMemory defaultValue)
        {
            return !IsInMemory(paramIdentifier) ? defaultValue : Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier)[paramIdentifier];
        }

        /// <summary>
        /// Get the memory entry for a certain parameter identifier, get default value from function if it doesn't exist.
        /// </summary>
        /// <param name="paramIdentifier">The parameter identifier.</param>
        /// <param name="defaultValueFunction">The default value function.</param>
        /// <returns>The memorised value with the given param identifier or the given default value if it isn't memorised.</returns>
        protected TMemory GetMemory(string paramIdentifier, Func<TMemory> defaultValueFunction)
        {
            return !IsInMemory(paramIdentifier) ? defaultValueFunction.Invoke() : Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier)[paramIdentifier];
        }
        /// <summary>
        /// Set the memory entry for a certain parameter identifier.
        /// </summary>
        /// <param name="paramIdentifier">The parameter identifier.</param>
        /// <param name="value">The memory entry to set.</param>
        /// <returns>The given memory entry (for convenience).</returns>
        protected TMemory SetMemory(string paramIdentifier, TMemory value)
        {
            if (!IsInMemory(paramIdentifier))
            {
                Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier).Add(paramIdentifier, value);
            }
            else
            {
                Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier)[paramIdentifier] = value;
            }

            return value;
        }

        /// <summary>
        /// Get a boolean indicating whether a certain param identifier is in memory.
        /// </summary>
        /// <param name="paramIdentifier">The parameter identifier.</param>
        /// <returns>A boolean indicating whether a certain param identifier is in memory.</returns>
        protected bool IsInMemory(string paramIdentifier)
        {
            return Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier).ContainsKey(paramIdentifier);
        }

        /// <summary>
        /// Clear this optimisers' memory (e.g. to reset gradient operations).
        /// </summary>
        public void ClearMemory()
        {
            Registry.Get<Dictionary<string, TMemory>>(_memoryIdentifier).Clear();
        }
    }
}
