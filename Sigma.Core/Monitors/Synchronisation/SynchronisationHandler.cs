/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Training.Operators;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.Synchronisation
{
	/// <summary>
	/// The default synchronisation handler for Sigma. It is responsible for syncing values between monitors 
	/// and the environment itself.
	/// </summary>
	public class SynchronisationHandler : ISynchronisationHandler
	{
		/// <summary>
		/// The environment this handler is associated with.
		/// </summary>
		public SigmaEnvironment Sigma { get; }

		/// <summary>
		/// Map every registry to a resolver for that registry. 
		/// </summary>
		protected Dictionary<IRegistry, IRegistryResolver> RegistryResolvers { get; }

		/// <summary>
		/// Default constructor for <see cref="ISynchronisationHandler"/>. 
		/// </summary>
		/// <param name="sigma">The <see cref="SigmaEnvironment"/> this <see cref="ISynchronisationHandler"/> is
		/// associated with. May not be <c>null</c>.</param>
		public SynchronisationHandler(SigmaEnvironment sigma)
		{
			if (sigma == null) throw new ArgumentNullException(nameof(sigma));

			RegistryResolvers = new Dictionary<IRegistry, IRegistryResolver>();

			Sigma = sigma;
		}

		/// <inheritdoc />
		public virtual void SynchroniseSet<T>(IRegistry registry, string key, T val, Action<T> onSuccess = null, Action<Exception> onError = null)
		{
			// check if the registry is from an operator
			foreach (IOperator op in Sigma.RunningOperatorsByTrainer.Values)
			{
				if (ReferenceEquals(op.Registry, registry))
				{
					//TODO: test if callback is called
					op.InvokeCommand(new SetValueCommand<T>(key, val, () => onSuccess?.Invoke(val)));

					return;
				}
			}

			IRegistryResolver resolver = RegistryResolvers.TryGetValue(registry, () => new RegistryResolver(registry));

			// check if at least one value has been set
			if (resolver.ResolveSet(key, val, true, typeof(T)).Length > 0)
			{
				onSuccess?.Invoke(val);
			}
			else
			{
				onError?.Invoke(new KeyNotFoundException($"{key} was not found in {registry} and could not be created."));
			}
		}

		/// <inheritdoc />
		public virtual T SynchroniseGet<T>(IRegistry registry, string key)
		{
			IRegistryResolver resolver = RegistryResolvers.TryGetValue(registry, () => new RegistryResolver(registry));
			return resolver.ResolveGetSingleWithDefault(key, default(T));
		}
	}
}