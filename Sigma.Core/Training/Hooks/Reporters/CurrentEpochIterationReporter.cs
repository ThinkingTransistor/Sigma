/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	public class CurrentEpochIterationReporter : BaseHook
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="format">The format string used (arg 0 is epoch, arg 1 is iteration).</param>
		public CurrentEpochIterationReporter(ITimeStep timestep, string format = "Epoch: {0} / Iteration: {1}") : base(timestep, "epoch", "iteration")
		{
			InvokePriority = -10000; // typically this should be invoked first
			ParameterRegistry["format_string"] = format;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			int epoch = registry.Get<int>("epoch");
			int iteration = registry.Get<int>("iteration");

			Report(epoch, iteration);
		}

		protected virtual void Report(int epoch, int iteration)
		{
			_logger.Info(string.Format(ParameterRegistry.Get<string>("format_string"), epoch, iteration));		
		}
	}
}
