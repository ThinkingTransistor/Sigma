/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Reflection;
using log4net;
using Sigma.Core.Persistence;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Saviors
{
    /// <summary>
    /// A disk savior hook for selectively storing certain objects on disk on certain conditions / at certain intervals.
    /// </summary>
    /// <typeparam name="T">The type of object to store.</typeparam>
    [Serializable]
    public class DiskSaviorHook<T> : BaseHook
    {
        [NonSerialized]
        private readonly ILog _logger = LogManager.GetLogger(Assembly.GetCallingAssembly(), typeof(DiskSaviorHook<T>).Namespace + "." + typeof(DiskSaviorHook<T>).Name);


        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileName">The file namer to store to disk as.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(string registryEntryToSave, string fileName, bool verbose = true) : this(Utils.TimeStep.Every(1, TimeScale.Iteration), registryEntryToSave, fileName, verbose)
        {
        }

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileName">The file namer to store to disk as.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(ITimeStep timestep, string registryEntryToSave, string fileName, bool verbose = true) : this(timestep, registryEntryToSave, Namers.Static(fileName), verbose)
        {
        }

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileNamer">The file namer to store to disk as.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(string registryEntryToSave, INamer fileNamer, bool verbose = true) : this(Utils.TimeStep.Every(1, TimeScale.Iteration), registryEntryToSave, fileNamer, verbose)
        {
        }

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileNamer">The file namer to store to disk as.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(ITimeStep timestep, string registryEntryToSave, INamer fileNamer, bool verbose = true) : this(timestep, registryEntryToSave, fileNamer, o => o, verbose)
        {
        }

        /// <summary>
        /// Create a savior hook that will automatically serialise a certain registry entry.
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="registryEntryToSave"></param>
        /// <param name="fileNamer">The file namer to store to disk as.</param>
        /// <param name="selectFunction">The select function to apply.</param>
        /// <param name="verbose">Indicate whether or not to report when the specified object was serialised.</param>
        public DiskSaviorHook(ITimeStep timestep, string registryEntryToSave, INamer fileNamer, Func<T, T> selectFunction, bool verbose = true) : base(timestep, registryEntryToSave)
        {
            if (registryEntryToSave == null) throw new ArgumentNullException(nameof(registryEntryToSave));
            if (fileNamer == null) throw new ArgumentNullException(nameof(fileNamer));
            if (selectFunction == null) throw new ArgumentNullException(nameof(selectFunction));

            ParameterRegistry["registry_entry_to_save"] = registryEntryToSave;
            ParameterRegistry["file_namer"] = fileNamer;
            ParameterRegistry["select_function"] = selectFunction;
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
            INamer fileNamer = ParameterRegistry.Get<INamer>("file_namer");
            object toSerialise = resolver.ResolveGetSingle<object>(registryEntryToSave);
            bool verbose = ParameterRegistry.Get<bool>("verbose");
            Func<T, T> selectFunction = ParameterRegistry.Get<Func<T, T>>("select_function");

            toSerialise = selectFunction.Invoke((T) toSerialise);

            lock (fileNamer)
            {
                Serialisation.WriteBinaryFile(toSerialise, fileNamer.GetName(registry, resolver, this), verbose: false);
            }

            if (verbose)
            {
                _logger.Info($"Saved \"{registryEntryToSave}\" to \"{SigmaEnvironment.Globals.Get<string>("storage_path")}{fileNamer}\".");
            }
        }
    }
}
