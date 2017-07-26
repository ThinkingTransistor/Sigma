/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Hooks.Processors;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A hook that reports the results of a target maximisation hook, which attempts to find an input that causes the network to output targets close to the desired targets.
	/// </summary>
	[Serializable]
	public class TargetMaximisationReporter : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="desiredTargets">The desired targets.</param>
		public TargetMaximisationReporter(INDArray desiredTargets, ITimeStep timestep) : this(desiredTargets, 0.05, timestep)
		{
		}

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="desiredTargets">The desired targets.</param>
		/// <param name="desiredCost">The desired cost.</param>
		public TargetMaximisationReporter(INDArray desiredTargets, double desiredCost, ITimeStep timestep) : base(timestep)
		{
			int uid = GetHashCode();
			ParameterRegistry["uid"] = uid;
			ParameterRegistry["desired_targets"] = desiredTargets;

			RequireHook(new TargetMaximisationHook(timestep, desiredTargets, $"shared.target_maximisation_result_{uid}_success", $"shared.target_maximisation_result_{uid}_input", desiredCost));

			InvokePriority = 10000;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			INDArray desiredTargets = ParameterRegistry.Get<INDArray>("desired_targets");
			int uid = ParameterRegistry.Get<int>("uid");

			bool success = resolver.ResolveGetSingleWithDefault($"shared.target_maximisation_result_{uid}_success", false);

			if (!success)
			{
				_logger.Warn($"Failed target maximisation for {desiredTargets}, nothing to print.");
			}
			else
			{
				IComputationHandler handler = Operator.Handler;
				INDArray inputs = resolver.ResolveGetSingle<INDArray>($"shared.target_maximisation_result_{uid}_input");

				OnTargetMaximisationSuccess(handler, inputs, desiredTargets);
			}
		}

		/// <summary>
		/// Handle a successful maximisation.
		/// </summary>
		/// <param name="handler">The computation handler.</param>
		/// <param name="inputs">The inputs.</param>
		/// <param name="desiredTargets">The desired targets.</param>
		protected virtual void OnTargetMaximisationSuccess(IComputationHandler handler, INDArray inputs, INDArray desiredTargets)
		{
			char[] palette = PrintUtils.AsciiGreyscalePalette;

			float min = handler.Min(inputs).GetValueAs<float>(), max = handler.Max(inputs).GetValueAs<float>();
			float range = max - min;

			_logger.Info($"Successfully completed target maximisation for {desiredTargets}: \n" +
						ArrayUtils.ToString<float>(inputs, e => palette[(int)(Math.Pow((e - min) / range, 1.9) * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false));
		}
	}
}
