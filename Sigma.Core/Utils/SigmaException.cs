/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// The general base class for all Sigma-specific exceptions.
	/// </summary>
	public class SigmaException : Exception
	{
		public SigmaException() : base()
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
