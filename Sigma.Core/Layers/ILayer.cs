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
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A layer in a neural network model.
	/// </summary>
	public interface ILayer
	{
		/// <summary>
		/// The unique name of this layer. 
		/// </summary>
		string Name { get; }

		IRegistry Parameters { get; }

		void Run(IRegistry inputs, IRegistry parameters, IRegistry outputs);
	}

	/// <summary>
	/// A layer construct representing a certain named layer construct, where all parameters are stored in a parameter registry.
	/// </summary>
	public interface ILayerConstruct
	{
		/// <summary>
		/// The unique name of the layer created in this layer construct.
		/// </summary>
		string Name { get; set; }

		/// <summary>
		/// The parameters of the layer created in this layer construct.
		/// </summary>
		IRegistry Parameters { get; }

		ILayer CreateLayer(IComputationHandler handler);
	}
}
