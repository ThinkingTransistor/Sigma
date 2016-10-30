using log4net;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core
{
	public partial class Sigma
	{
		internal IRegistry rootRegistry;
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public String Name
		{
			get; internal set;
		}

		private Sigma(String name)
		{
			this.Name = name;
		}

		public void Prepare()
		{
			//TODO
		}

		public void Run()
		{
			//TODO
		}

		internal static IRegistry activeSigmaEnvironments;
		private static ILog clazzLogger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		static Sigma()
		{ 
			
		}

		/// <summary>
		/// Create an environment with the given name.
		/// </summary>
		/// <param name="environmentName"></param>
		/// <returns></returns>
		public static Sigma Create(string environmentName)
		{
			if (Exists(environmentName))
			{
				throw new ArgumentException($"Cannot create environment, environment {environmentName} already exists.");
			}

			Sigma environment = new Sigma(environmentName);

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
		public static Sigma GetOrCreate(string environmentName)
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
		public static bool Exists(String environmentName)
		{
			return activeSigmaEnvironments.Contains(environmentName);
		}

		/// <summary>
		/// Gets an environment with a given name, if previously created (null otherwise).
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		/// <returns>The existing with the given name or null.</returns>
		public static Sigma Get(string environmentName)
		{
			return activeSigmaEnvironments.Get<Sigma>(environmentName);
		}

		/// <summary>
		/// Removes an environment with a given name.
		/// </summary>
		/// <param name="environmentName">The environment name.</param>
		public static void Remove(string environmentName)
		{
			activeSigmaEnvironments.Remove(environmentName);
		}
	}
}