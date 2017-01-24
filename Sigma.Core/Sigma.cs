/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using log4net;
using log4net.Config;
using Sigma.Core.Monitors;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Operators;
using Sigma.Core.Utils;

namespace Sigma.Core
{
	/// <summary>
	/// A sigma environment, where all the magic happens.
	/// </summary>
	public class SigmaEnvironment
	{
		/// <summary>
		/// If the <see cref="SigmaEnvironment"/> is currently running. 
		/// </summary>
		public bool Running { get; private set; }

		private readonly ISet<IMonitor> _monitors;
		private readonly IDictionary<string, ITrainer> _trainersByName;
		private readonly ISet<IOperator> _runningOperators;
		private readonly ConcurrentQueue<IPassiveHook> _hooksToExecute;
		private readonly ConcurrentQueue<KeyValuePair<IHook, IOperator>> _hooksToAttach;
		private bool _requestedStop;

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// The unique name of this environment. 
		/// </summary>
		public string Name
		{
			get;
		}

		/// <summary>
		/// The root registry of this environment where all exposed parameters are stored hierarchically.
		/// </summary>
		public IRegistry Registry { get; }

		/// <summary>
		/// The registry resolver corresponding to the registry used in this environment. 
		/// For easier notation and faster access you can retrieve and using regex-style registry names and dot notation.
		/// </summary>
		public IRegistryResolver RegistryResolver { get; }

		/// <summary>
		/// The random number generator to use for randomised operations for reproducibility. 
		/// </summary>
		public Random Random
		{
			get; private set;
		}

		private SigmaEnvironment(string name)
		{
			Name = name;
			Registry = new Registry();
			RegistryResolver = new RegistryResolver(Registry);
			Random = new Random();
			_monitors = new HashSet<IMonitor>();
			_hooksToExecute = new ConcurrentQueue<IPassiveHook>();
			_hooksToAttach = new ConcurrentQueue<KeyValuePair<IHook, IOperator>>();
			_runningOperators = new HashSet<IOperator>();
			_trainersByName = new ConcurrentDictionary<string, ITrainer>();
		}

		/// <summary>
		/// Set the random seed used for all randomised computations for reproducibility.
		/// </summary>
		/// <param name="seed"></param>
		public void SetRandomSeed(int seed)
		{
			Random = new Random(seed);
		}

		/// <summary>
		/// Add a monitor to this environment.
		/// </summary>
		/// <typeparam name="TMonitor">The type of the monitor to be added.</typeparam>
		/// <param name="monitor">The monitor to add.</param>
		/// <returns>The monitor given (for convenience).</returns>
		public TMonitor AddMonitor<TMonitor>(TMonitor monitor) where TMonitor : IMonitor
		{
			monitor.Sigma = this;
			monitor.Registry = new Registry(Registry);

			monitor.Initialise();
			_monitors.Add(monitor);

			_logger.Debug($"Added monitor {monitor} to sigma environment \"{Name}\".");

			return monitor;
		}

		/// <summary>
		/// Prepare this environment for execution. Start all monitors.
		/// </summary>
		public void Prepare()
		{
			foreach (IMonitor monitor in _monitors)
			{
				monitor.Start();
			}
		}

		/// <summary>
		/// Create a trainer with a certain unique name and add it to this environment.
		/// </summary>
		/// <param name="name">The trainer name.</param>
		/// <returns>A new trainer with the given name.</returns>
		public ITrainer CreateTrainer(string name)
		{
			return AddTrainer(new Trainer(name));
		}

		/// <summary>
		/// Add a trainer to this environment.
		/// </summary>
		/// <typeparam name="TTrainer">The type of the trainer to add.</typeparam>
		/// <param name="trainer">The trainer to add.</param>
		/// <returns>The given trainer (for convenience).</returns>
		public TTrainer AddTrainer<TTrainer>(TTrainer trainer) where TTrainer : ITrainer
		{
			if (_trainersByName.ContainsKey(trainer.Name))
			{
				throw new InvalidOperationException($"Trainer with name \"{trainer.Name}\" is already registered in this environment (\"{Name}\").");
			}

			_trainersByName.Add(trainer.Name, trainer);
			trainer.Sigma = this;

			return trainer;
		}

		/// <summary>
		/// Run this environment. Execute all registered options until stop is requested.
		/// Note: This method is blocking and executes in the calling thread. 
		/// </summary>
		public void Run()
		{
			_logger.Info($"Starting sigma environment \"{Name}\"...");

			bool shouldRun = true;

			Running = true;

			InitialiseTrainers();
			FetchRunningOperators();
			StartRunningOperators();

			_logger.Info($"Started sigma environment \"{Name}\".");

			while (shouldRun)
			{
				ProcessHooksToAttach();
				ProcessHooksToExecute();

				if (_requestedStop)
				{
					_logger.Info($"Stopping sigma environment \"{Name}\" as request stop flag was set...");

					shouldRun = false;
				}
			}

			StopRunningOperators();

			Running = false;

			_logger.Info($"Stopped sigma environment \"{Name}\".");
		}

		private void InitialiseTrainers()
		{
			foreach (ITrainer trainer in _trainersByName.Values)
			{
				trainer.Initialise(trainer.Operator.Handler);
			}
		}

		protected void FetchRunningOperators()
		{
			if (_trainersByName.Count == 0)
			{
				_logger.Info($"No trainers attached to this environment ({Name}).");
			}

			_logger.Debug($"Fetching operators from {_trainersByName.Count} trainers in environment \"{Name}\"...");

			foreach (ITrainer trainer in _trainersByName.Values)
			{
				_runningOperators.Add(trainer.Operator);

				trainer.Operator.Sigma = this;
			}
		}

		private void StartRunningOperators()
		{
			_logger.Debug($"Starting operators from {_trainersByName.Count} trainers in environment \"{Name}\"...");

			foreach (IOperator op in _runningOperators)
			{
				op.Start();
			}
		}

		private void StopRunningOperators()
		{
			_logger.Debug($"Stopping operators from {_trainersByName.Count} trainers in environment \"{Name}\"...");

			foreach (IOperator op in _runningOperators)
			{
				op.SignalStop();
			}
		}

		/// <summary>
		/// Signal this environment to stop execution as soon as possible (if running).
		/// </summary>
		public void SignalStop()
		{
			if (!Running)
			{
				return;
			}

			_logger.Debug($"Stop signal received in environment \"{Name}\".");

			_requestedStop = true;
		}

		private void ProcessHooksToExecute()
		{
			while (!_hooksToExecute.IsEmpty)
			{
				IPassiveHook hook;

				if (_hooksToExecute.TryDequeue(out hook))
				{
					new Task(() => hook.Invoke(hook.RegistryCopy)).Start();
				}
			}
		}

		private void ProcessHooksToAttach()
		{
			while (!_hooksToAttach.IsEmpty)
			{
				KeyValuePair<IHook, IOperator> hookPair;

				if (_hooksToAttach.TryDequeue(out hookPair))
				{
					if (hookPair.Key is ActiveHook)
					{
						hookPair.Value.AttachHook((ActiveHook) hookPair.Key);
					}
					else if (hookPair.Key is PassiveHook)
					{
						hookPair.Value.AttachHook((PassiveHook) hookPair.Key);
					}
					else
					{
						_logger.Warn($"Unable to attach hook {hookPair.Key} to operator {hookPair.Value}, hook is neither active nor passive hook.");
					}
				}
			}
		}

		/// <summary>
		/// Request for a hook to be attached to a certain trainer's operator (identified by its name).
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		/// <param name="trainerName">The trainer name whose trainer's operator the hook should be attached to.</param>
		public void RequestAttachHook(IHook hook, string trainerName)
		{
			if (!_trainersByName.ContainsKey((trainerName)))
			{
				throw new ArgumentException($"Trainer with name {trainerName} is not registered in this environment ({Name}).");
			}

			RequestAttachHook(hook, _trainersByName[trainerName]);
		}

		/// <summary>
		/// Request for a hook to be attached to a certain trainer's operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		/// <param name="trainer">The trainer whose operator the hook should be attached to.</param>
		public void RequestAttachHook(IHook hook, ITrainer trainer)
		{
			RequestAttachHook(hook, trainer.Operator);
		}

		/// <summary>
		/// Request for a hook to be attached to a certain operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		/// <param name="operatorToAttachTo">The operator to attach to.</param>
		public void RequestAttachHook(IHook hook, IOperator operatorToAttachTo)
		{
			_hooksToAttach.Enqueue(new KeyValuePair<IHook, IOperator>(hook, operatorToAttachTo));
		}

		/// <summary>
		/// Request the asynchronous execution of a passive hook.
		/// Note: The required parameter registry must already be set in the given hook.
		/// </summary>
		/// <param name="hook">The hook to execute.</param>
		public void RequestExecuteHookAsync(IPassiveHook hook)
		{
			_hooksToExecute.Enqueue(hook);
		}

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="fullMatchedIdentifierArray">The fully matched identifiers corresponding to the given match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		public T[] ResolveGet<T>(string matchIdentifier, out string[] fullMatchedIdentifierArray, T[] values = null)
		{
			return RegistryResolver.ResolveGet(matchIdentifier, out fullMatchedIdentifierArray, values);
		}

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		public T[] ResolveGet<T>(string matchIdentifier, T[] values = null)
		{
			return RegistryResolver.ResolveGet(matchIdentifier, values);
		}

		/// <summary>
		/// Set a single given value of a certain type to all matching identifiers. For the detailed supported syntax <see cref="IRegistryResolver"/>
		/// Note: The individual registries might throw an exception if a type-protected value is set to the wrong type.
		/// </summary>
		/// <typeparam name="T">The type of the value.</typeparam>
		/// <param name="matchIdentifier">The full match identifier. </param>
		/// <param name="value"></param>
		/// <param name="associatedType">Optionally set the associated type (<see cref="IRegistry"/>)</param>
		/// <returns>A list of fully qualified matches to the match identifier.</returns>
		public string[] ResolveSet<T>(string matchIdentifier, T value, Type associatedType = null)
		{
			return RegistryResolver.ResolveSet(matchIdentifier, value, associatedType);
		}

		// static part of SigmaEnvironment

		/// <summary>
		/// The task manager for this environment.
		/// </summary>
		public static ITaskManager TaskManager
		{
			get; internal set;
		}

		internal static readonly CultureInfo DefaultCultureInfo = new CultureInfo("en-GB");

		internal static IRegistry ActiveSigmaEnvironments;
		private static readonly ILog ClazzLogger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		static SigmaEnvironment()
		{
			// logging not initialised
			SetDefaultCulture(DefaultCultureInfo);

			ActiveSigmaEnvironments = new Registry();
			TaskManager = new TaskManager();

			Globals = new Registry();
			RegisterGlobals();
		}

		/// <summary>
		/// This method sets the default culture. 
		/// </summary>
		/// <param name="culture">The culture that will be the new default. </param>
		// This method also exists in BaseLocaleTest.
		private static void SetDefaultCulture(CultureInfo culture)
		{
			Thread.CurrentThread.CurrentCulture = culture;
			CultureInfo.DefaultThreadCurrentCulture = culture;
		}


		/// <summary>
		/// A global variable pool for globally relevant constants (e.g. workspace path).
		/// </summary>
		public static IRegistry Globals { get; }

		/// <summary>
		/// Register all global parameters with an initial value and required associated type. 
		/// </summary>
		private static void RegisterGlobals()
		{
			Globals.Set("workspace_path", "workspace/", typeof(string));
			Globals.Set("cache", Globals.Get<string>("workspace_path") + "cache/", typeof(string));
			Globals.Set("datasets", Globals.Get<string>("workspace_path") + "datasets/", typeof(string));
			Globals.Set("web_proxy", WebRequest.DefaultWebProxy, typeof(IWebProxy));
		}

		/// <summary>
		/// Loads the log4net configuration from the corresponding xml file. See log4net for more details.
		/// </summary>
		public static void EnableLogging()
		{
			XmlConfigurator.Configure();
		}

		/// <summary>
		/// Create an environment with a certain name.
		/// </summary>
		/// <param name="environmentName"></param>
		/// <returns>A new environment with the given name.</returns>
		public static SigmaEnvironment Create(string environmentName)
		{
			if (Exists(environmentName))
			{
				throw new ArgumentException($"Cannot create environment, environment {environmentName} already exists.");
			}

			SigmaEnvironment environment = new SigmaEnvironment(environmentName);

			//do environment initialisation and registration

			ActiveSigmaEnvironments.Set(environmentName, environment);

			ClazzLogger.Info($"Created and registered sigma environment \"{environmentName}\".");

			return environment;
		}

		/// <summary>
		/// Get environment if it already exists, create and return new one if it does not. 
		/// </summary>
		/// <param name="environmentName"></param>
		/// <returns>A new environment with the given name or the environment already associated with the name.</returns>
		public static SigmaEnvironment GetOrCreate(string environmentName)
		{
			return Exists(environmentName) ? Get(environmentName) : Create(environmentName);
		}

		/// <summary>
		/// Gets an environment with a given name, if previously created (null otherwise).
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>The existing with the given name or null.</returns>
		public static SigmaEnvironment Get(string environmentName)
		{
			return ActiveSigmaEnvironments.Get<SigmaEnvironment>(environmentName);
		}

		/// <summary>
		/// Checks whether an environment exists with the given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>A boolean indicating if an environment with the given name exists.</returns>
		public static bool Exists(string environmentName)
		{
			return ActiveSigmaEnvironments.ContainsKey(environmentName);
		}

		/// <summary>
		/// Removes an environment with a given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		public static void Remove(string environmentName)
		{
			ActiveSigmaEnvironments.Remove(environmentName);
		}

		/// <summary>
		/// Removes all active environments.
		/// </summary>
		public static void Clear()
		{
			ActiveSigmaEnvironments.Clear();
		}
	}
}