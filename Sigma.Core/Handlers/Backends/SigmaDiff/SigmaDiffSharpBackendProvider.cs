/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using DiffSharp;
using DiffSharp.Config;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// A DiffSharp backend provider as passed to the global Sigma.DiffSharp configuration to dynamically select a backend. 
	/// </summary>
	public class SigmaDiffSharpBackendProvider : IBackendProvider
	{
		private static SigmaDiffSharpBackendProvider _instance;
		public static SigmaDiffSharpBackendProvider Instance => _instance ?? (_instance = new SigmaDiffSharpBackendProvider());

		public static void AssignToDiffSharpGlobal()
		{
			DiffSharp.Config.GlobalConfig.BackendProvider = Instance;
		}

		private readonly Dictionary<long, object> _registeredBackendConfigs;

		public SigmaDiffSharpBackendProvider()
		{
			_registeredBackendConfigs = new Dictionary<long, object>();
		}

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

		public BackendConfig<T> GetBackend<T>(long backendTag)
		{
			return (BackendConfig<T>)_registeredBackendConfigs[backendTag];
		}

		public BackendConfig<T> GetBackend<T>(object obj)
		{
			if (obj is DiffSharp.AD.Float32.DNDArray)
			{
				return (BackendConfig<T>) _registeredBackendConfigs[((SigmaDiffDataBuffer<float>) ((DiffSharp.AD.Float32.DNDArray) obj).Buffer.DataBuffer).BackendTag];
			}
			else if (obj is Util.ShapedDataBufferView<T>)
			{
				return (BackendConfig<T>) _registeredBackendConfigs[((SigmaDiffDataBuffer<T>) ((Util.ShapedDataBufferView<T>) obj).DataBuffer).BackendTag];
			}
			else if (obj is SigmaDiffDataBuffer<T>)
			{
				return (BackendConfig<T>) _registeredBackendConfigs[((SigmaDiffDataBuffer<T>) obj).BackendTag];
			}

			throw new InvalidOperationException($"Cannot get backend for unknown object {obj} of type {obj.GetType()}, object is neither a known type nor a backend mapped type.");
		}
	}
}
