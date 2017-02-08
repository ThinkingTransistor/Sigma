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
		IReadOnlyCollection<string> RequiredRegistryEntries { get; }

		/// <summary>
		/// The hooks that are required for this hook (i.e. the hooks this hook depends on).
		/// Required hooks are prioritised and executed before the dependent hook.
		/// If multiple required hooks are functionally equivalent, only one will be invoked. 
		/// </summary>
		IReadOnlyCollection<IHook> RequiredHooks { get; }

		/// <summary>
		/// Flag whether this hook should be invoked by the owner (worker/operator) or in a separate background thread.
		/// Note: When invoked in background, hooks received a complete copy of all required registry entries and can therefore not directly modify the parameters of a worker/operator.
		/// </summary>
		bool InvokeInBackground { get; }

		/// <summary>
		/// The operator that owns this hook and dispatched it for execution. 
		/// </summary>
		IOperator Operator { get; set; }

		/// <summary>
		/// Invoke this hook with a certain parameter registry (and the corresponding resolver).
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">The to the registry corresponding registry resolver for easier access to nested / tagged / filtered elements.</param>
		void Invoke(IRegistry registry, IRegistryResolver resolver);

		/// <summary>
		/// Check if this hook's functionality is equal to that of another. 
		/// Used when deciding which hooks can be omitted (optimised out).
		/// Note: Different parameters typically infer different functionalities.
		///		  If your custom hook requires any external parameters that alter its behaviour reflect that in this method.
		/// </summary>
		/// <param name="other">The hook to check.</param>
		/// <returns>A boolean indicating whether or not the other hook does the same that this one does.</returns>
		bool FunctionallyEquals(IHook other);
	}

	/// <summary>
	/// A hook that is only invoked one time on global and local scope(s), regardless of TimeStep (though live time should be 1 for consistency).
	/// Used for operator-wide parameter changes that should be applied to all local scopes. 
	/// </summary>
	public interface ICommand : IHook
	{
	}
}