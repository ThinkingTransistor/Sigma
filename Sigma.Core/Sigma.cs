/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Monitors;
using Sigma.Core.Utils;
using System;
using System.Net;

namespace Sigma.Core
{
	/// <summary>
	/// A sigma environment, where all the magic happens.
	/// </summary>
	public class SigmaEnvironment
	{
		private IRegistry rootRegistry;
		private IRegistryResolver rootRegistryResolver;

		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// The unique name of this environment. 
		/// </summary>
		public string Name
		{
			get; internal set;
		}

		/// <summary>
		/// The root registry of this environment where all exposed parameters are stored hierarchically.
		/// </summary>
		public IRegistry Registry
		{
			get { return rootRegistry; }
		}

		/// <summary>
		/// The registry resolver corresponding to the registry used in this environment. 
		/// For easier notation and faster access you can retrieve and using regex-style registry names and dot notation.
		/// </summary>
		public IRegistryResolver RegistryResolver
		{
			get { return rootRegistryResolver; }
		}

		private SigmaEnvironment(string name)
		{
			this.Name = name;
			this.rootRegistry = new Registry();
			this.rootRegistryResolver = new RegistryResolver(this.rootRegistry);
		}

		public TMonitor AddMonitor<TMonitor>(TMonitor monitor) where TMonitor : IMonitor
		{
			//TODO: 
			monitor.Sigma = this;


			monitor.Initialise();

			return monitor;
		}

		public void Prepare()
		{
			//TODO
		}

		public void Run()
		{
			//TODO
		}

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		T[] ResolveGet<T>(string matchIdentifier, ref string[] fullMatchedIdentifierArray, T[] values = null)
		{
			return rootRegistryResolver.ResolveGet<T>(matchIdentifier, ref fullMatchedIdentifierArray, values);
		}

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="fullMatchedIdentifierArray">A list of fully qualified matches to the match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		T[] ResolveGet<T>(string matchIdentifier, T[] values = null)
		{
			return rootRegistryResolver.ResolveGet<T>(matchIdentifier, values);
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
		public string[] ResolveSet<T>(string matchIdentifier, T value, System.Type associatedType = null)
		{
			return rootRegistryResolver.ResolveSet<T>(matchIdentifier, value, associatedType);
		}

		// static part of SigmaEnvironment

		/// <summary>
		/// The task manager for this environment.
		/// </summary>
		public static ITaskManager TaskManager
		{
			get; internal set;
		}

		internal static IRegistry activeSigmaEnvironments;
		private static readonly ILog clazzLogger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		static SigmaEnvironment()
		{ 
			activeSigmaEnvironments = new Registry();
			TaskManager = new TaskManager();

			Globals = new Registry();
			RegisterGlobals();
		}

		/// <summary>
		/// A global variable pool for globally relevant constants (e.g. workspace path).
		/// </summary>
		public static IRegistry Globals { get; private set; }


		/// <summary>
		/// Register all globals with an initial value and required associated type. 
		/// </summary>
		private static void RegisterGlobals()
		{
			Globals.Set("workspacePath", "workspace/", typeof(string));
			Globals.Set("cache", Globals.Get<string>("workspacePath") + "cache/", typeof(string));
			Globals.Set("datasets", Globals.Get<string>("workspacePath") + "datasets/", typeof(string));
			Globals.Set("webProxy", System.Net.WebRequest.DefaultWebProxy, typeof(IWebProxy));
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

			activeSigmaEnvironments.Set(environmentName, environment);

			clazzLogger.Info($"Created and registered sigma environment \"{environmentName}\"");

			return environment;
		}

		/// <summary>
		/// Get environment if it already exists, create and return new one if it does not. 
		/// </summary>
		/// <param name="environmentName"></param>
		/// <returns>A new environment with the given name or the environment already associated with the name.</returns>
		public static SigmaEnvironment GetOrCreate(string environmentName)
		{
			if (!Exists(environmentName))
			{
				return Create(environmentName);
			}

			return Get(environmentName);
		}

		/// <summary>
		/// Gets an environment with a given name, if previously created (null otherwise).
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>The existing with the given name or null.</returns>
		public static SigmaEnvironment Get(string environmentName)
		{
			return activeSigmaEnvironments.Get<SigmaEnvironment>(environmentName);
		}

		/// <summary>
		/// Checks whether an environment exists with the given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>A boolean indicating if an environment with the given name exists.</returns>
		public static bool Exists(string environmentName)
		{
			return activeSigmaEnvironments.ContainsKey(environmentName);
		}

		/// <summary>
		/// Removes an environment with a given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		public static void Remove(string environmentName)
		{
			activeSigmaEnvironments.Remove(environmentName);
		}

		/// <summary>
		/// Removes all active environments.
		/// </summary>
		public static void Clear()
		{
			activeSigmaEnvironments.Clear();
		}
	}
}