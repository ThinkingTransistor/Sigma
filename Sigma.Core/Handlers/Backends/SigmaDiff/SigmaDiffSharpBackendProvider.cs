using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DiffSharp.Config;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	public class SigmaDiffSharpBackendProvider : IBackendProvider
	{
		private Dictionary<long, object> _registeredBackends;

		public SigmaDiffSharpBackendProvider(Dictionary<long, object> registeredBackends)
		{
			if (registeredBackends == null) throw new ArgumentNullException(nameof(registeredBackends));

			_registeredBackends = registeredBackends;
		}

		public BackendConfig<T> GetBackend<T>(object obj)
		{
			long tag = -1;

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

			if (!_registeredBackends.ContainsKey(tag))
			{
				throw new InvalidOperationException($"Cannot fetch backend for tag {tag}, tag is not registered with any backend in this provider.");
			}

			return (BackendConfig<T>) _registeredBackends[tag];
		}
	}
}
