/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using DiffSharp.Config;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	public class SigmaDiffSharpBackendProvider : IBackendProvider
	{
		private static SigmaDiffSharpBackendProvider _instance;
		public static SigmaDiffSharpBackendProvider Instance => _instance ?? (_instance = new SigmaDiffSharpBackendProvider());

		public static void AssignToDiffSharpGlobal()
		{
			DiffSharp.Config.GlobalConfig.BackendProvider = Instance;
		}

		private readonly Dictionary<long, object> _registeredBackendConfigs;

		public long Register<T>(BackendConfig<T> backendConfig)
		{
			if (backendConfig == null) throw new ArgumentNullException(nameof(backendConfig));

			long maxTag = -1;

			foreach (long existingTag in _registeredBackendConfigs.Keys)
			{
				if (existingTag > maxTag)
				{
					maxTag = existingTag;
				}
			}

			long tag = maxTag + 1;

			_registeredBackendConfigs.Add(tag, backendConfig);

			return tag;
		}

		public SigmaDiffSharpBackendProvider()
		{
			_registeredBackendConfigs = new Dictionary<long, object>();
		}

		public BackendConfig<T> GetBackend<T>(object obj)
		{
			long tag;

			if (obj is DiffSharp.Interop.Float32.DNDArray)
			{
				tag = ((DiffSharp.Interop.Float32.DNDArray) obj).BackendTag;
			}
			else if (obj is DiffSharp.Interop.Float32.DNumber)
			{
				tag = ((DiffSharp.Interop.Float32.DNumber) obj).BackendTag;
			}
			else if (obj is DiffSharp.Interop.Float64.DNDArray)
			{
				tag = ((DiffSharp.Interop.Float64.DNDArray) obj).BackendTag;
			}
			else if (obj is DiffSharp.Interop.Float64.DNumber)
			{
				tag = ((DiffSharp.Interop.Float64.DNumber) obj).BackendTag;
			}
			else
			{
				throw new NotSupportedException($"Cannot fetch backend for unknown object {obj}.");
			}

			if (!_registeredBackendConfigs.ContainsKey(tag))
			{
				throw new InvalidOperationException($"Cannot fetch backend for tag {tag}, tag is not registered with any backend in this provider.");
			}

			return (BackendConfig<T>) _registeredBackendConfigs[tag];
		}
	}
}
