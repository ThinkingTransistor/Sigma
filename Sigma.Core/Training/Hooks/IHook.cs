/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System.Collections.Generic;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// A hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before Execute is first called (so that the operator can fetch the requested parameters).
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
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		void Invoke(IRegistry registry);		
	}

	/// <summary>
	/// An active hook that actively influences the calling operator by modifying the given parameters.
	/// Note: The distinction between "active" and "passive" depends on the kind of hook. 
	///		  For example, a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IActiveHook : IHook
	{
	}

	/// <summary>
	/// A passive hook that passively executes code independent from the operand (without modifying the given parameters or dependency on the operator). 
	/// Note: The distinction between "active" and "passive" depends on the kind of hook. 
	///		  For example a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IPassiveHook : IHook
	{
		/// <summary>
		/// A complete and local copy of the global registry with the parameters required for this hook for asynchronous execution.
		/// </summary>
		IRegistry RegistryCopy { get; set; }
	}

	/// <summary>
	/// An active hook that is only invoked one time on the operator, regardless of TimeStep (though live time should be 1 for consistency).
	/// </summary>
	public interface ICommand : IActiveHook
	{
	}
}