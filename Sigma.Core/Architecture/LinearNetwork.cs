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

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A linear neural network with linear acyclic connections between layers. 
	/// </summary>
	public class LinearNetwork : INetwork
	{
		public INetworkArchitecture Architecture { get; set; }
	}
}
