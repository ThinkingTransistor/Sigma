/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Runtime.Serialization;

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// An exception if a networks architecture is invalid. Yup.
	/// </summary>
	public class InvalidNetworkArchitectureException : Exception
	{
		public InvalidNetworkArchitectureException(string message) : base(message)
		{
		}

		public InvalidNetworkArchitectureException(string message, Exception innerException) : base(message, innerException)
		{
		}

		protected InvalidNetworkArchitectureException(SerializationInfo info, StreamingContext context) : base(info, context)
		{
		}
	}
}
