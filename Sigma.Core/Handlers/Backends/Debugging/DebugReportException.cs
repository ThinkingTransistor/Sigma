/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Handlers.Backends.Debugging
{
	public class DebugReportException : Exception
	{
		public object[] BadValues { get; }

		public DebugReportException(string message, params object[] badValues) : base(message)
		{
			BadValues = badValues;
		}

		public DebugReportException(string message, Exception innerException, params object[] badValues) : base(message, innerException)
		{
			BadValues = badValues;
		}
	}
}
