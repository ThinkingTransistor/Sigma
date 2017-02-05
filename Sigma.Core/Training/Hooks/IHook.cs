/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System.Collections.Generic;
using Sigma.Core.Training.Operators;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// Represents a hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before <see cref="Invoke"/> is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	public interface IHook
	{
		/// <summary>
		/// The time step at which to execute this hook.
		/// </summary>
		ITimeStep TimeStep { get; }

		/// <summary>
		/// The global registry entries required for the execution of this hook.
		/// </summary>
		ISet<string> RequiredRegistryEntries { get; }

		/// <summary>
		/// Flag whether this hook should be invoked by the owner (worker/operator) or in a separate background thread.
		/// Note: When invoked in background, hooks received a complete copy of all required registry entries and can therefore not directly modify the parameters of a worker/operator.
		/// </summary>
		bool InvokeInBackground { get; }

		/// <summary>
		/// A complete and local copy of the global registry with the parameters required for this hook for asynchronous execution.
		/// Note: This field is only populated if <see cref="InvokeInBackground"/> is set to true.
		/// </summary>
		IRegistry RegistryCopy { get; set; }

		/// <summary>
		/// The operator that owns this hook and dispatched it for execution. 
		/// </summary>
		IOperator Operator { get; set; }

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		void Invoke(IRegistry registry);		
	}

	/// <summary>
	/// An active hook that is within each worker's frame of reference (per worker).
	/// Note: The distinction between "active" and "passive" depends on the kind of hook.
	///		  Active hooks are invoked within each worker on the worker's time scale, passive hooks are invoked by the operator on the operator's time scale (e.g. all worker's are at epoch x, iteration y).
	///		  For example, a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IActiveHook : IHook
	{
	}

	/// <summary>
	/// A passive hook that is within the operators frame of reference (shared).
	/// Note: The distinction between "active" and "passive" depends on the kind of hook.
	///		  Active hooks are invoked within each worker on the worker's time scale, passive hooks are invoked by the operator on the operator's time scale (e.g. all worker's are at epoch x, iteration y).
	///		  For example, a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IPassiveHook : IHook
	{
	}

	/// <summary>
	/// An active hook that is only invoked one time on the operator, regardless of TimeStep (though live time should be 1 for consistency).
	/// </summary>
	public interface ICommand : IActiveHook
	{
	}
}