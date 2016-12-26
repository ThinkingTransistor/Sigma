/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using DiffSharp;
using DiffSharp.Config;
using Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu;

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
		private readonly Dictionary<object, long> _backendMappedValues;

		public SigmaDiffSharpBackendProvider()
		{
			_registeredBackendConfigs = new Dictionary<long, object>();
			_backendMappedValues = new Dictionary<object, long>();
		}

		public T MapToBackend<T>(T value, long backendTag)
		{
			_backendMappedValues.Add(value, backendTag);

			return value;
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

		public BackendConfig<T> GetBackend<T>(object obj)
		{
			if (obj is ADNDFloat32Array)
			{
				return (BackendConfig<T>) _registeredBackendConfigs[((SigmaDiffDataBuffer<float>) ((ADNDFloat32Array) obj).Data).BackendTag];
			}
			else if (obj is Util.ShapedDataBufferView<T>)
			{
				return (BackendConfig<T>) _registeredBackendConfigs[((SigmaDiffDataBuffer<T>) ((Util.ShapedDataBufferView<T>) obj).DataBuffer).BackendTag];
			}
			else if (_backendMappedValues.ContainsKey(obj))
			{
				return (BackendConfig<T>) _registeredBackendConfigs[_backendMappedValues[obj]];
			}

			throw new InvalidOperationException($"Cannot get backend for unknown object {obj} of type {obj.GetType()}, object is neither a known type nor a backend mapped type.");
		}
	}
}
