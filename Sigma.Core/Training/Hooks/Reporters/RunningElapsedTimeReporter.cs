/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
    [Serializable]
    public class RunningElapsedTimeReporter : BaseHook
    {
        [NonSerialized]
        private readonly ILog _logger = LogManager.GetLogger(typeof(DebugHandler));

        /// <summary>
        /// Create a hook with a certain time step and a set of required global registry entries. 
        /// </summary>
        /// <param name="timestep">The time step.</param>
        public RunningElapsedTimeReporter(ITimeStep timestep) : base(timestep)
        {
            DefaultTargetMode = TargetMode.Global;
        }

        /// <summary>
        /// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
        /// </summary>
        /// <param name="registry">The registry containing the required values for this hook's execution.</param>
        /// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
        public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
        {
            if (ParameterRegistry.ContainsKey("last_time"))
            {
                long lastTime = ParameterRegistry.Get<long>("last_time");
                long currentTime = Operator.RunningTimeMilliseconds;
                long elapsedTime = currentTime - lastTime;

                Report(lastTime, currentTime, elapsedTime);
            }

            ParameterRegistry["last_time"] = Operator.RunningTimeMilliseconds;
        }

        protected virtual void Report(long lastTime, long currentTime, long elapsedTime)
        {
            _logger.Info($"Elapsed time since last {TimeStep}: {elapsedTime}ms");
        }
    }
}
