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
	/// <typeparam name="T"></typeparam>
	public interface ITrainerSelector<T> : ISelector<T> where T : ITrainer
	{
		/// <summary>
		/// Keep a set of trainer components, discard all other components in a new trainer.
		/// </summary>
		/// <param name="component">The component(s) to keep.</param>
		/// <returns>A selector for a new trainer with the given component(s) retained.</returns>
		ISelector<T> Keep(TrainerComponent component);

		/// <summary>
		/// Discard specified trainer components in a new trainer.
		/// </summary>
		/// <param name="component">The component(s) to discard.</param>
		/// <returns>A selector for a new trainer with the given component(s) discarded.</returns>
		ISelector<T> Discard(TrainerComponent component);
	}

	/// <summary>
	/// An individual trainer component for <see cref="ITrainer"/> data selection (keep / discard).
	/// </summary>
	[Flags]
	public enum TrainerComponent
	{
		/// <summary>
		/// Nothing (except the trainer name, which is the minimum state and included by default).
		/// </summary>
		None            = 0,

		/// <summary>
		/// The network model in this trainer.
		/// </summary>
		Network         = 1 << 0,

		/// <summary>
		/// The attached initialisers.
		/// </summary>
		Initialisers    = 1 << 1,

		/// <summary>
		/// The attached value modifiers.
		/// </summary>
		ValueModifiers  = 1 << 2,

		/// <summary>
		/// The attached optimiser.
		/// </summary>
		Optimiser       = 1 << 3,

		/// <summary>
		/// The attached operator.
		/// </summary>
		Operator        = 1 << 4,

		/// <summary>
		/// The attached data provider.
		/// </summary>
		DataProvider    = 1 << 5,

		/// <summary>
		/// The attached data iterators.
		/// </summary>
		DataIterators   = 1 << 6,

		/// <summary>
		/// The attached hooks.
		/// </summary>
		Hooks       	= 1 << 7,

		/// <summary>
		/// Everything.
		/// </summary>
		All				= Int32.MaxValue
	}
}
