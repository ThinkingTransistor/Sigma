/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// A utility lambda hook for miscellaneous things and debugging.
	/// </summary>
	public class LambdaHook : BaseHook
	{
		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="invokeAction">The action that will be invoked in <see cref="Invoke"/>.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		public LambdaHook(ITimeStep timestep, Action<IRegistry, IRegistryResolver> invokeAction, params string[] requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
			ParameterRegistry["invoke_action"] = invokeAction;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			var action = ParameterRegistry.Get<Action<IRegistry, IRegistryResolver>>("invoke_action");
			action.Invoke(registry, resolver);
		}
	}
}
