/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Training;

namespace Sigma.Core.Persistence.Selectors
{
	/// <summary>
	/// A sigma environment selector for selecting specific parts of an entire environment.
	/// </summary>
	public interface IEnvironmentSelector : ISelector<SigmaEnvironment>
	{
		/// <summary>
		/// Keep a set of trainer components, discard all other components in a new trainer.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new trainer with the given component(s) retained.</returns>
		ISelector<SigmaEnvironment> Keep(params TrainerComponent[] components);

		/// <summary>
		/// Discard specified trainer components in a new trainer.
		/// </summary>
		/// <param name="components">The component(s) to discard.</param>
		/// <returns>A selector for a new trainer with the given component(s) discarded.</returns>
		ISelector<SigmaEnvironment> Discard(params TrainerComponent[] components);
	}

	/// <summary>
	/// An individual trainer component for <see cref="ITrainer"/> data selection (keep / discard).
	/// </summary>
	public class EnvironmentComponent : SelectorComponent
	{
		/// <summary>
		/// Nothing (except the environment name, which is the minimum state and included by default).
		/// </summary>
		public static readonly EnvironmentComponent None = new EnvironmentComponent(0);

		/// <summary>
		/// All attached monitors.
		/// </summary>
		public static readonly EnvironmentComponent Monitors = new EnvironmentComponent(1 << 0);

		/// <summary>
		/// All attached trainers.
		/// </summary>
		public static EnvironmentComponent Trainers(TrainerComponent component)
		{
			return new EnvironmentComponent(1 << 1, component);
		}

		/// <summary>
		/// All attached trainers, monitors and additional runtime data (pending requests, hook queues, execution states).
		/// </summary>
		public static readonly EnvironmentComponent RuntimeState = new EnvironmentComponent(Monitors.Id | Trainers(TrainerComponent.Everything).Id);

		/// <summary>
		/// Everything.
		/// </summary>
		public static readonly EnvironmentComponent All = new EnvironmentComponent(int.MaxValue);

		/// <summary>
		/// Create a selector component with a certain id. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different subflags).
		/// </summary>
		/// <param name="id">The id.</param>
		protected EnvironmentComponent(int id) : base(id)
		{
		}

		/// <summary>
		/// Create a selector component with a certain id and certain sub-components. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		/// <param name="subComponents">The selected sub components.</param>
		protected EnvironmentComponent(int id, params SelectorComponent[] subComponents) : base(id, subComponents)
		{
		}
	}
}
