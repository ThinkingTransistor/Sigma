/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// The general base class for all Sigma-specific exceptions.
	/// </summary>
	public class SigmaException : Exception
	{
		public SigmaException()
		{
		}

		public SigmaException(string message) : base(message)
		{
		}

		public SigmaException(string message, Exception innerException) : base(message, innerException)
		{
		}
	}
}
