/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Monitors;
using Sigma.Core.Utils;

namespace Sigma.Core
{
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

		public T AddMonitor<T>(T monitor) where T : IMonitor
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

		internal static IRegistry activeSigmaEnvironments;
		private static ILog clazzLogger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		static SigmaEnvironment()
		{
			activeSigmaEnvironments = new Registry();
		}

		/// <summary>
		/// Create an environment with the given name.
		/// </summary>
		/// <param name="environmentName"></param>
		/// <returns></returns>
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
		/// Checks whether an environment exists with the given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>A boolean indicating if an environment with the given name exists.</returns>
		public static bool Exists(string environmentName)
		{
			return activeSigmaEnvironments.ContainsKey(environmentName);
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