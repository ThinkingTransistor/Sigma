/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Training.Operators;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks
{

	/// <summary>
	/// The base implementation of the <see cref="IHook"/> interface.
	/// Represents a hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before <see cref="Invoke"/> is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	public abstract class BaseHook : IHook
	{
		/// <summary>
		/// The time step at which to execute this hook.
		/// </summary>
		public ITimeStep TimeStep { get; }

		/// <summary>
		/// The global registry entries required for the execution of this hook.
		/// </summary>
		public ISet<string> RequiredRegistryEntries { get; }

		/// <summary>
		/// Flag whether this hook should be invoked by the owner (worker/operator) or in a separate background task.
		/// By default this is set to false (no background execution). Hooks that are not invoked in the background should execute as fast as possible. Otherwise...
		/// Set this to true if performance intensive operations (e.g. storing to disk, processing large arrays) are used in this hook.
		/// Note: When invoked in background, hooks received a complete copy of all required registry entries and can therefore not directly modify the parameters of a worker/operator.
		/// </summary>
		public bool InvokeInBackground { get; } = false;

		/// <summary>
		/// The operator that owns this hook and dispatched it for execution. 
		/// </summary>
		public IOperator Operator { get; set; }

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseHook(ITimeStep timestep, params string[] requiredRegistryEntries) : this(timestep, new HashSet<string>(requiredRegistryEntries))
		{
		}

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseHook(ITimeStep timestep, ISet<string> requiredRegistryEntries)
		{
			if (timestep == null)
			{
				throw new ArgumentNullException(nameof(timestep));
			}

			if (requiredRegistryEntries == null)
			{
				throw new ArgumentNullException(nameof(requiredRegistryEntries));
			}

			TimeStep = timestep;
			RequiredRegistryEntries = requiredRegistryEntries;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		public abstract void Invoke(IRegistry registry);
	}

	/// <summary>
	/// The base implementation of the <see cref="IActiveHook"/> interface.
	/// An active hook that is within each worker's frame of reference (per worker).
	/// Note: The distinction between "active" and "passive" depends on the kind of hook.
	///		  Active hooks are invoked within each worker on the worker's time scale, passive hooks are invoked by the operator on the operator's time scale (e.g. all worker's are at epoch x, iteration y).
	///		  For example, a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public abstract class BaseActiveHook : BaseHook, IActiveHook
	{
		/// <summary>
		/// Create an active hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		protected BaseActiveHook(ITimeStep timestep, params string[] requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}

		/// <summary>
		/// Create an active hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		protected BaseActiveHook(ITimeStep timestep, ISet<string> requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}
	}

	/// <summary>
	/// The base implementation of the <see cref="IPassiveHook"/> interface.
	/// A passive hook that is within the operators frame of reference (shared).
	/// Note: The distinction between "active" and "passive" depends on the kind of hook.
	///		  Active hooks are invoked within each worker on the worker's time scale, passive hooks are invoked by the operator on the operator's time scale (e.g. all worker's are at epoch x, iteration y).
	///		  For example, a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public abstract class BasePassiveHook : BaseHook, IPassiveHook
	{
		/// <summary>
		/// A complete and local copy of the global registry with the parameters required for this hook for asynchronous execution.
		/// </summary>
		public IRegistry RegistryCopy { get; set; }

		/// <summary>
		/// Create a passive hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		protected BasePassiveHook(ITimeStep timestep, params string[] requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}

		/// <summary>
		/// Create a passive hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		protected BasePassiveHook(ITimeStep timestep, ISet<string> requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}
	}

	/// <summary>
	/// The base implementation of the <see cref="ICommand"/> interface.
	/// An active hook that is only invoked one time on the operator, regardless of TimeStep (though live time should be 1 for consistency).
	/// </summary>
	public abstract class BaseCommand : BaseActiveHook, ICommand
	{
		/// <summary>
		/// The time step for commands (intermediate time scale, interval and live time are 1 for single invocation).
		/// </summary>
		protected static readonly TimeStep TimeStepCommand = new TimeStep(TimeScale.Indeterminate, 1, 1);

		/// <summary>
		/// Create a command with a set of required global registry entries.
		/// </summary>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseCommand(params string[] requiredRegistryEntries) : base(TimeStepCommand, requiredRegistryEntries)
		{
		}

		/// <summary>
		/// Create a command with a set of required global registry entries.
		/// </summary>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseCommand(ISet<string> requiredRegistryEntries) : base(TimeStepCommand, requiredRegistryEntries)
		{
		}
	}
}
