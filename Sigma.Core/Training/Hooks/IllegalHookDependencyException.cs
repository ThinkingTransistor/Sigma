/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// An exception that occurs if a hook dependency is not legal and the hook manager there does not know what to do.
	/// </summary>
	public class IllegalHookDependencyException : Exception
	{
		public IllegalHookDependencyException(string message) : base(message)
		{
		}

		public IllegalHookDependencyException(string message, Exception innerException) : base(message, innerException)
		{
		}
	}
}
