/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Training.Operators;
using System.Linq;
using Sigma.Core.Utils;

namespace Sigma.Core.Persistence.Selectors.Operator
{
	/// <summary>
	/// A base oeprator selector that handles basic <see cref="NetworkComponent"/> selection independent of actual network type.
	/// Note:   There is no default operator selector using reflection because operators do not have a common constructor type (it could be anything).
	///         Essentially, creating operators via activator will probably break things and not having a default selector makes people take care of their own constructors as they see fit.
	/// </summary>
	/// <typeparam name="TOperator"></typeparam>
	public abstract class BaseOperatorSelector<TOperator> : IOperatorSelector<TOperator> where TOperator : IOperator
	{
		/// <summary>
		/// The current result object of type <see cref="TOperator"/>.
		/// </summary>
		public TOperator Result { get; }

		/// <summary>
		/// Create a base operator selector with a certain operator.
		/// </summary>
		/// <param name="operator">The operator.</param>
		protected BaseOperatorSelector(TOperator @operator)
		{
			Result = @operator;
		}

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

			if (components.ContainsFlag(OperatorComponent.Everything) || components.ContainsFlag(OperatorComponent.RuntimeState))
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
			if (components.ContainsFlag(OperatorComponent.Everything))
			{
				return Keep(OperatorComponent.None);
			}

			if (components.ContainsFlag(OperatorComponent.RuntimeState))
			{
				throw new InvalidOperationException($"Cannot only discard runtime state and keep everything, that does not make sense.");
			}

			throw new InvalidOperationException($"Cannot discard given components {components}, discard is invalid and probably does not make sense.");
		}

		/// <summary>
		/// Create an operator of this operator selectors appropriate type (take necessary constructor arguments from the current <see cref="Result"/> operator).
		/// </summary>
		/// <returns>The operator.</returns>
		protected abstract TOperator CreateOperator();

		/// <summary>
		/// Create an operator selector with a certain operator.
		/// </summary>
		/// <param name="operator">The operator</param>
		/// <returns>An operator selector with the given operator.</returns>
		protected abstract IOperatorSelector<TOperator> CreateSelector(TOperator @operator);
	}
}
