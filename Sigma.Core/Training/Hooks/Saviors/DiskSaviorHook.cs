/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using log4net;
using log4net.Core;
using Sigma.Core.Persistence;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Saviors
{
    [Serializable]
    public class DiskSaviorHook<T> : BaseHook
    {
        [NonSerialized]
        private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileName">The file name to store to disk as.</param>
        /// <param name="selectFunction">The select function to apply.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(ITimeStep timestep, string registryEntryToSave, string fileName, bool verbose = true) : this(timestep, registryEntryToSave, fileName, o => o, verbose)
        {
        }

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileName">The file name to store to disk as.</param>
        /// <param name="selectFunction">The select function to apply.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(ITimeStep timestep, string registryEntryToSave, string fileName, Func<T, T> selectFunction, bool verbose = true) : base(timestep, registryEntryToSave)
        {
            ParameterRegistry["registry_entry_to_save"] = registryEntryToSave ?? throw new ArgumentNullException(nameof(registryEntryToSave));
            ParameterRegistry["file_name"] = fileName ?? throw new ArgumentNullException(nameof(fileName));
            ParameterRegistry["select_function"] = selectFunction ?? throw new ArgumentNullException(nameof(selectFunction));
            ParameterRegistry["verbose"] = verbose;

            DefaultTargetMode = TargetMode.Global;
        }

        /// <summary>
        /// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
        /// </summary>
        /// <param name="registry">The registry containing the required values for this hook's execution.</param>
        /// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
        public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
        {
            string registryEntryToSave = ParameterRegistry.Get<string>("registry_entry_to_save");
            string fileName = ParameterRegistry.Get<string>("file_name");
            object toSerialise = resolver.ResolveGetSingle<object>(registryEntryToSave);
            bool verbose = ParameterRegistry.Get<bool>("verbose");
            Func<T, T> selectFunction = ParameterRegistry.Get<Func<T, T>>("select_function");

            toSerialise = selectFunction.Invoke((T) toSerialise);

            Serialisation.WriteBinaryFile(toSerialise, fileName, verbose: false);

            if (verbose)
            {
                _logger.Info($"Saved \"{registryEntryToSave}\" to \"{SigmaEnvironment.Globals.Get<string>("storage_path")}/{fileName}\".");
            }
        }
    }
}
