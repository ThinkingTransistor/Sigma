/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks
{

	/// <summary>
	/// The base implementation of the <see cref="IHook"/> interface.
	/// Represents a hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before Execute is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	public abstract class Hook : IHook
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
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected Hook(ITimeStep timestep, params string[] requiredRegistryEntries) : this(timestep, new HashSet<string>(requiredRegistryEntries))
		{
		}

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected Hook(ITimeStep timestep, ISet<string> requiredRegistryEntries)
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
	/// Represents an active hook that actively influences the calling operator by modifying the given parameters.
	/// </summary>
	public abstract class BaseActiveHook : Hook, IActiveHook
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
	/// A passive hook that passively executes code independent from the operand (without modifying the given parameters or dependency on the operator). 
	/// </summary>
	public abstract class BasePassiveHook : Hook, IPassiveHook
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
