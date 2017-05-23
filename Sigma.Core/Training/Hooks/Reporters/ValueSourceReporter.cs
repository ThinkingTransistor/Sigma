using System.Collections.Generic;
using Sigma.Core.Utils;
using Sigma.Core.Monitors.Synchronisation;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A hook that stores given values and can provide them to a <see cref="ISynchronisationHandler"/> as a source.
	/// </summary>
	public class ValueSourceReporter : BaseHook, ISynchronisationSource
	{
		private const string ValueIdentifier = "values";
		private const string RegistryResolver = "resolver";

		//private readonly IDictionary<string, object> _values = new Dictionary<string, object>();

		/// <summary>
		///	Create a hook that fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueSourceReporter(TimeStep timestep, string valueIdentifier) : base(timestep, valueIdentifier)
		{
			Initialise(valueIdentifier);
		}


		/// <summary>
		///	Create a hook that fetches a given amount of values (i.e. registry identifiers) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifiers">The values that will be fetched (i.e. registry identifiers). E.g. <c>"optimiser.cost_total"</c>, ...</param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueSourceReporter(ITimeStep timestep, params string[] valueIdentifiers) : base(timestep, valueIdentifiers)
		{
			Initialise(valueIdentifiers);
		}

		private void Initialise()
		{
			IRegistry reg = new Registry();
			reg.Add(RegistryResolver, new RegistryResolver(reg));

			ParameterRegistry.Add(ValueIdentifier, reg);
		}

		/// <summary>
		/// Initialise the dictionary containing the values with given <see ref="valueIdentifier"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched.</param>
		protected void Initialise(string valueIdentifier)
		{
			Initialise();
			IRegistry values = (IRegistry) ParameterRegistry[ValueIdentifier];
			values.Add(valueIdentifier, null);
			Keys = new[] {valueIdentifier};
		}

		/// <summary>
		/// Initialise the dictionary containing the values with given <see ref="valueIdentifiers"/>.
		/// </summary>
		/// <param name="valueIdentifiers">The values that will be fetched.</param>
		protected void Initialise(string[] valueIdentifiers)
		{
			Initialise();
			IRegistry values = (IRegistry) ParameterRegistry[ValueIdentifier];

			foreach (string identifier in valueIdentifiers)
			{
				values.Add(identifier, null);
			}
			Keys = valueIdentifiers;
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
			IRegistry values = (IRegistry) ParameterRegistry[ValueIdentifier];
			IRegistryResolver resolver = values.Get<IRegistryResolver>(RegistryResolver);

			//TODO: validate lock requirement, probably it is required
			lock (values)
			{
				T[] vals = resolver.ResolveGet<T>(key);

				if (vals.Length > 0)
				{
					val = vals[0];
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

		/// <summary>
		/// Determine whether a given key is contained / manged by this source.
		/// </summary>
		/// <param name="key">The key that will be checked.</param>
		/// <returns><c>True</c> if given key can be accessed with get / set, <c>false</c> otherwise.</returns>
		public bool Contains(string key)
		{
			IDictionary<string, object> values = (IDictionary<string, object>) ParameterRegistry[ValueIdentifier];

			return values.ContainsKey(key);
		}

		/// <summary>
		/// This is a list of keys this source provides. It is <b>completely</b> optional, although it is recommended to implement it.
		/// 
		/// Once a new source is added, the keys of the sources are checked against to determine double entries which makes debugging for users easier (as log entries are produced autoamtically).
		/// </summary>
		public string[] Keys { get; private set; }
	}
}