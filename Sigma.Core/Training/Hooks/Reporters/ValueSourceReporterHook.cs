using System.Collections.Generic;
using Sigma.Core.Utils;
using Sigma.Core.Monitors.Synchronisation;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A hook that stores given values and can provide them to a <see cref="ISynchronisationHandler"/> as a source.
	/// </summary>
	public class ValueSourceReporterHook : BaseHook, ISynchronisationSource
	{
		private const string ValueIdentifier = "values";
		//private readonly IDictionary<string, object> _values = new Dictionary<string, object>();

		/// <summary>
		///	Create a hook that fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueSourceReporterHook(TimeStep timestep, string valueIdentifier) : base(timestep, valueIdentifier)
		{
			Initialise(valueIdentifier);
		}


		/// <summary>
		///	Create a hook that fetches a given amount of values (i.e. registry identifiers) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifiers">The values that will be fetched (i.e. registry identifiers). E.g. <c>"optimiser.cost_total"</c>, ...</param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueSourceReporterHook(ITimeStep timestep, params  string[] valueIdentifiers) : base(timestep, valueIdentifiers)
		{
			Initialise(valueIdentifiers);
		}

		private void Initialise()
		{
			ParameterRegistry.Add(ValueIdentifier, new Dictionary<string, object>());
		}

		/// <summary>
		/// Initialise the dictionary containing the values with given <see ref="valueIdentifier"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched.</param>
		protected void Initialise(string valueIdentifier)
		{
			Initialise();
			IDictionary<string, object> values = (IDictionary<string, object>) ParameterRegistry[ValueIdentifier];
			values.Add(valueIdentifier, null);
		}

		/// <summary>
		/// Initialise the dictionary containing the values with given <see ref="valueIdentifiers"/>.
		/// </summary>
		/// <param name="valueIdentifiers">The values that will be fetched.</param>
		protected void Initialise(string[] valueIdentifiers)
		{
			Initialise();
			IDictionary<string, object> values = (IDictionary<string, object>) ParameterRegistry[ValueIdentifier];

			foreach (string identifier in valueIdentifiers)
			{
				values.Add(identifier, null);
			}
		}


		/// <summary>
		/// Try to retrieve a value from this source (if existent).
		/// </summary>
		/// <typeparam name="T">The type of the value that will be retrieved.</typeparam>
		/// <param name="key">The key of the value.</param>
		/// <param name="val">The value itself that will be assigned if it could be retrieved, <c>null</c> otherwise.</param>
		/// <returns><c>True</c> if the source could retrieve given key, <c>false</c> otherwise.</returns>
		public bool TryGet<T>(string key, out T val)
		{
			IDictionary<string, object> values = (IDictionary<string, object>) ParameterRegistry[ValueIdentifier];

			//TODO: validate lock requirement, probably it is required
			lock (values)
			{
				object oVal;

				if (values.TryGetValue(key, out oVal))
				{
					if (oVal == null)
					{
						val = default(T);
					}
					else
					{
						val = (T) oVal;
					}

					return true;
				}

				val = default(T);
				return false;
			}
		}

		/// <summary>
		/// No set supported in this observative hook. 
		/// </summary>
		/// <typeparam name="T">The type of the value that will be set.</typeparam>
		/// <param name="key">The key of the value.</param>
		/// <param name="val">The value itself that will be assigned if it applicable.</param>
		/// <returns><c>True</c> if the source could set given key, <c>false</c> otherwise.</returns>
		public bool TrySet<T>(string key, T val)
		{
			// a set is not supported
			return false;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			IDictionary<string, object> values = (IDictionary<string, object>) ParameterRegistry[ValueIdentifier];

			//TODO: validate lock requirement, probably it is required
			lock (values)
			{
				foreach (KeyValuePair<string, object> valuePair in registry)
				{
					values[valuePair.Key] = valuePair.Value;
				}
			}
		}
	}
}