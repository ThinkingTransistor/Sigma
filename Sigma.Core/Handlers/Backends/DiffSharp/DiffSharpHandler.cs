/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Handlers.Backends.DiffSharp
{
	public abstract class DiffSharpHandler<T> 
	{
		public IBlasBackend BlasBackend { get; }
		public ILapackBackend LapackBackend { get; }
	}
}
