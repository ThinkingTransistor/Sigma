/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Processors
{
    [Serializable]
    public class MetricProcessorHook<T> : BaseHook where T : class
    {
        public MetricProcessorHook(string registryEntryToProcess, Func<T, IComputationHandler, INumber> metricFunction, string metricSharedResultEntry) : this(Utils.TimeStep.Every(1, TimeScale.Iteration), registryEntryToProcess, metricFunction, metricSharedResultEntry)
        {
        }

        public MetricProcessorHook(ITimeStep timestep, string registryEntryToProcess, Func<T, IComputationHandler, INumber> metricFunction, string metricSharedResultIdentifier) : base(timestep, registryEntryToProcess)
        {
            if (registryEntryToProcess == null) throw new ArgumentNullException(nameof(registryEntryToProcess));
            if (metricFunction == null) throw new ArgumentNullException(nameof(metricFunction));
            if (metricSharedResultIdentifier == null) throw new ArgumentNullException(nameof(metricSharedResultIdentifier));

            InvokePriority = -1000;
            ParameterRegistry["registry_entry_to_process"] = registryEntryToProcess;
            ParameterRegistry["metric_function"] = metricFunction;
            ParameterRegistry["metric_shared_result_identifier"] = metricSharedResultIdentifier;
        }

        /// <inheritdoc />
        public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
        {
            IComputationHandler handler = Operator.Handler;

            string registryEntryToProcess = ParameterRegistry.Get<string>("registry_entry_to_process");
            Func<T, IComputationHandler, INumber> metricFunction = ParameterRegistry.Get<Func<T, IComputationHandler, INumber>>("metric_function");
            string metricSharedResultIdentifier = ParameterRegistry.Get<string>("metric_shared_result_identifier");

            object[] entries = resolver.ResolveGet<object>(registryEntryToProcess);

            double totalMetric = 0.0;
            int count = 0;

            foreach (object entry in entries)
            {
                T entryAsT = entry as T;
                IEnumerable<T> entryAsEnumerable = entry as IEnumerable<T>;
                IDictionary<string, T> entryAsDictionary = entry as IDictionary<string, T>;

                if (entryAsDictionary != null)
                {
                    entryAsEnumerable = entryAsDictionary.Values;
                }

                if (entryAsT != null)
                {
                    totalMetric += metricFunction.Invoke(entryAsT, handler).GetValueAs<double>();
                    count++;
                }
                else if (entryAsEnumerable != null)
                {
                    foreach (T value in entryAsEnumerable)
                    {
                        totalMetric += metricFunction.Invoke(value, handler).GetValueAs<double>();
                        count++;
                    }
                }
                else
                {
                    throw new InvalidOperationException($"Cannot process metric for entry of type {entry.GetType()} with identifier \"{registryEntryToProcess}\", must be {typeof(T)} or enumerable thereof.");
                }
            }

            double resultMetric = totalMetric / count;

            resolver.ResolveSet(metricSharedResultIdentifier, resultMetric, addIdentifierIfNotExists: true);
        }
    }
}
