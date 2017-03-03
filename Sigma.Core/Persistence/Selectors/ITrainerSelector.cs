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
		None             = 0,
		Sigma            = 1 << 0,
		Network          = 1 << 1,
		Initialisers     = 1 << 2,
		ValueModifiers   = 1 << 3,
		Optimiser        = 1 << 4,
		Operator         = 1 << 5,
		DataProvider     = 1 << 6,
		DataIterators    = 1 << 7,
		Hooks            = 1 << 8,
	}
}
