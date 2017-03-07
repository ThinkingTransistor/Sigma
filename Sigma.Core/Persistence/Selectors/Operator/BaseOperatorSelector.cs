/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Training.Operators;
using System.Linq;
using Sigma.Core.Training.Hooks;

namespace Sigma.Core.Persistence.Selectors.Operator
{
	/// <summary>
	/// A base oeprator selector that handles basic <see cref="NetworkComponent"/> selection independent of actual network type.
	/// </summary>
	/// <typeparam name="TOperator"></typeparam>
	public abstract class BaseOperatorSelector<TOperator> : IOperatorSelector<TOperator> where TOperator : IOperator
	{
		/// <summary>
		/// The current result object of type <see cref="TOperator"/>.
		/// </summary>
		public TOperator Result { get; }

		/// <summary>
		/// Get the "emptiest" available version of this <see cref="TOperator"/> while still retaining a legal object state for type <see cref="TOperator"/>.
		/// Note: Any parameters that are fixed per-object (such as unique name) must be retained.
		/// </summary>
		/// <returns>An empty version of the object.</returns>
		public ISelector<TOperator> Empty()
		{
			return Keep(OperatorComponent.None);
		}

		/// <summary>
		/// Get an uninitialised version of this <see cref="TOperator"/> without any runtime information, but ready to be re-initialised.
		/// </summary>
		/// <returns></returns>
		public ISelector<TOperator> Uninitialised()
		{
			return Keep(OperatorComponent.None);
		}

		/// <summary>
		/// Keep a set of network components, discard all other components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new network with the given component(s) retained.</returns>
		public ISelector<TOperator> Keep(params OperatorComponent[] components)
		{
		    TOperator @operator;

			if (components.Contains(OperatorComponent.Everything) || components.Contains(OperatorComponent.RuntimeState))
			{
			    @operator = (TOperator) Result.ShallowCopy();
			}
			else
			{
			    @operator = CreateOperator();
			}

			return CreateSelector(@operator);
		}

		/// <summary>
		/// Discard specified network components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to discard.</param>
		/// <returns>A selector for a new network with the given component(s) discarded.</returns>
		public ISelector<TOperator> Discard(params OperatorComponent[] components)
		{
			if (components.Contains(OperatorComponent.Everything))
			{
				return Keep(OperatorComponent.None);
			}

			if (components.Contains(OperatorComponent.RuntimeState))
			{
				throw new InvalidOperationException($"Cannot only discard runtime state and keep everything, that does not make sense.");
			}

			throw new InvalidOperationException($"Cannot discard given components {components}, discard is invalid and probably does not make sense.");
		}

		/// <summary>
		/// Create an operator of this operator selectors appropriate type.
		/// </summary>
		/// <returns>The operator.</returns>
		public abstract TOperator CreateOperator();

		/// <summary>
		/// Create an operator selector with a certain operator.
		/// </summary>
		/// <param name="operator">The operator</param>
		/// <returns>An operator selector with the given operator.</returns>
		public abstract IOperatorSelector<TOperator> CreateSelector(TOperator @operator);
	}
}
