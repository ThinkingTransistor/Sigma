/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Training;

namespace Sigma.Core.Persistence.Selectors
{
	/// <summary>
	/// A trainer selector to selectively keep and discard <see cref="ITrainer"/> data.
	/// </summary>
	/// <typeparam name="T">The trainer type.</typeparam>
	public interface ITrainerSelector<out T> : ISelector<T> where T : ITrainer
	{
		/// <summary>
		/// Keep a set of trainer components, discard all other components in a new trainer.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new trainer with the given component(s) retained.</returns>
		ISelector<T> Keep(params TrainerComponent[] components);

		/// <summary>
		/// Discard specified trainer components in a new trainer.
		/// </summary>
		/// <param name="componentsToDiscard">The component(s) to discard.</param>
		/// <returns>A selector for a new trainer with the given component(s) discarded.</returns>
		ISelector<T> Discard(params TrainerComponent[] componentsToDiscard);
	}

	/// <summary>
	/// An individual trainer component for <see cref="ITrainer"/> data selection (keep / discard).
	/// </summary>
	public class TrainerComponent : SelectorComponent
	{
		/// <summary>
		/// Nothing (except the trainer name, which is the minimum state and included by default).
		/// </summary>
		public static readonly TrainerComponent None = new TrainerComponent(0);

		/// <summary>
		/// The network model in this trainer.
		/// </summary>
		public static TrainerComponent Network(params NetworkComponent[] components)
		{
			return new TrainerComponent(1 << 0, components);
		}

		/// <summary>
		/// The attached initialisers.
		/// </summary>
		public static readonly TrainerComponent Initialisers = new TrainerComponent(1 << 1);

		/// <summary>
		/// The attached value modifiers.
		/// </summary>
		public static readonly TrainerComponent ValueModifiers = new TrainerComponent(1 << 2);

		/// <summary>
		/// The attached optimiser.
		/// </summary>
		public static readonly TrainerComponent Optimiser = new TrainerComponent(1 << 3);

		/// <summary>
		/// The attached operator.
		/// </summary>
		public static TrainerComponent Operator(params OperatorComponent[] components)
		{
			return new TrainerComponent(1 << 4, components);
		}

		/// <summary>
		/// The attached data provider.
		/// </summary>
		public static readonly TrainerComponent DataProvider = new TrainerComponent(1 << 5);

		/// <summary>
		/// The attached data iterators.
		/// </summary>
		public static readonly TrainerComponent DataIterators = new TrainerComponent(1 << 6);

		/// <summary>
		/// The attached hooks.
		/// </summary>
		public static readonly TrainerComponent Hooks = new TrainerComponent(1 << 7);

		/// <summary>
		/// Everything.
		/// </summary>
		public static readonly TrainerComponent All = new TrainerComponent(int.MaxValue);

		/// <summary>
		/// Create a selector component with a certain id. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		public TrainerComponent(int id) : base(id)
		{
		}

		/// <summary>
		/// Create a selector component with a certain id and certain sub-components. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		/// <param name="subComponents">The selected sub components.</param>
		protected TrainerComponent(int id, params SelectorComponent[] subComponents) : base(id, subComponents)
		{
		}
	}
}
