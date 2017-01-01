/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A layer buffer containing all relevant inputs, outputs and parameters for a layer to work with in an iteration. 
	/// </summary>
	public interface ILayerBuffer
	{
		/// <summary>
		/// The actual layer instance this buffer is for. 
		/// </summary>
		ILayer Layer { get; }

		/// <summary>
		/// The parameters for a layer to work with.
		/// </summary>
		IRegistry Parameters { get; }

		/// <summary>
		/// Indicate the alias names of the input layers to be supplied externally.
		/// </summary>
		string[] ExternalInputs { get; }

		/// <summary>
		/// Indicate the alias names of the output layers to be supplied externally.
		/// </summary>
		string[] ExternalOutputs { get; }

		/// <summary>
		/// The alias-named inputs for a layer.
		/// </summary>
		IDictionary<string, IRegistry> Inputs { get; }

		/// <summary>
		/// The alias-named outputs for a layer.
		/// </summary>
		IDictionary<string, IRegistry> Outputs { get; }
	}
}
