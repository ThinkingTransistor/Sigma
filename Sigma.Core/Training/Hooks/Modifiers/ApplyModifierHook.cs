/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Modifiers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Modifiers
{
	/// <summary>
	/// An apply modifier hook that applies a certain <see cref="IValueModifier"/> to a certain, single parameter.
	/// </summary>
	[Serializable]
	public class ApplyModifierHook : BaseHook
	{
		public ApplyModifierHook(string parameter, IValueModifier modifier, TimeStep timeStep) : base(timeStep, parameter)
		{
			if (modifier == null) throw new ArgumentNullException(nameof(modifier));

			ParameterRegistry.Set("parameter_identifier", parameter, typeof(string));
			ParameterRegistry.Set("modifier", modifier, typeof(string));
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			IValueModifier modifier = ParameterRegistry.Get<IValueModifier>("modifier");
			string identifier = ParameterRegistry.Get<string>("parameter_identifier");
			object parameter = resolver.ResolveGetSingle<object>(identifier);

			INumber asNumber = parameter as INumber;
			INDArray asArray = parameter as INDArray;

			if (asNumber != null)
			{
				parameter = modifier.Modify(identifier, asNumber, asNumber.AssociatedHandler);
			}
			else if (asArray != null)
			{
				parameter = modifier.Modify(identifier, asArray, asArray.AssociatedHandler);
			}
			else
			{
				throw new InvalidOperationException($"Cannot apply modifier {modifier} to parameter \"{identifier}\" with value {parameter}, " +
				                                    $"parameter is neither {nameof(INumber)} nor {nameof(INDArray)}.");
			}

			resolver.ResolveSet(identifier, parameter);
		}
	}
}
