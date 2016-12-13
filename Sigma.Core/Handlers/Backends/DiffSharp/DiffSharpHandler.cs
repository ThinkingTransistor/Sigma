/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using log4net;

namespace Sigma.Core.Handlers.Backends.DiffSharp
{
	public abstract class DiffSharpHandler<T> 
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public IBlasBackend BlasBackend { get; }
		public ILapackBackend LapackBackend { get; }

		static DiffSharpHandler()
		{
			PlatformDependentDllUtils.EnsureSetPlatformDependentDllDirectory();
		}
	}
}
