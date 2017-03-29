/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Training.Operators;

namespace Sigma.Core.Persistence.Selectors
{
	/// <summary>
	/// An operator selector to selectively keep and discard <see cref="IOperator"/> data.
	/// </summary>
	/// <typeparam name="TOperator"></typeparam>
	public interface IOperatorSelector<out TOperator> : ISelector<TOperator> where TOperator : IOperator
	{
		/// <summary>
		/// Keep a set of network components, discard all other components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new network with the given component(s) retained.</returns>
		ISelector<TOperator> Keep(params OperatorComponent[] components);

		/// <summary>
		/// Discard specified network components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to discard.</param>
		/// <returns>A selector for a new network with the given component(s) discarded.</returns>
		ISelector<TOperator> Discard(params OperatorComponent[] components);
	}

	/// <summary>
	/// An individual operator component. 
	/// </summary>
	public class OperatorComponent : SelectorComponent
	{
		/// <summary>
		/// Create a selector component with a certain id. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		public OperatorComponent(int id) : base(id)
		{
		}

		/// <summary>
		/// Create a selector component with a certain id and certain sub-components. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		/// <param name="subComponents">The selected sub components.</param>
		public OperatorComponent(int id, params SelectorComponent[] subComponents) : base(id, subComponents)
		{
		}

		/// <summary>
		/// Nothing (except the operator type and constructor settings, which is the minimum state and included by default).
		/// </summary>
		public static readonly OperatorComponent None = new OperatorComponent(0);

		/// <summary>
		/// The runtime state of this operator (trainer state, attached hooks, invocation caches). 
		/// </summary>
		public static readonly OperatorComponent RuntimeState = new OperatorComponent(1 << 0);

		/// <summary>
		/// Everything (in addition to <see cref="RuntimeState"/> also the attached Sigma environment).
		/// </summary>
		public static readonly OperatorComponent Everything = new OperatorComponent(int.MaxValue);
	}
}
