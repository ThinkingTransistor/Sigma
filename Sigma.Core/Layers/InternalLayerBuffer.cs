/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A default implementation of the <see cref="ILayerBuffer"/> interface for internal use.
	/// </summary>
	[Serializable]
	public class InternalLayerBuffer : ILayerBuffer
	{
		public ILayer Layer { get; }
		public IRegistry Parameters { get; }
		public string[] ExternalInputs { get; }
		public string[] ExternalOutputs { get; }
		public IReadOnlyDictionary<string, IRegistry> Inputs { get; }
		public IReadOnlyDictionary<string, IRegistry> Outputs { get; }

		/// <summary>
		/// Create a layer buffer for a certain layer, parameters, inputs, and outputs.
		/// </summary>
		/// <param name="layer">The layer.</param>
		/// <param name="parameters">The parameters.</param>
		/// <param name="inputs">The inputs.</param>
		/// <param name="outputs">The outputs.</param>
		/// <param name="externalInputs">Indicate the inputs that are external.</param>
		/// <param name="externalOutputs">Indicate if outputs are external.</param>
		public InternalLayerBuffer(ILayer layer, IRegistry parameters, IDictionary<string, IRegistry> inputs, IDictionary<string, IRegistry> outputs,
									string[] externalInputs, string[] externalOutputs)
		{
			if (layer == null) throw new ArgumentNullException(nameof(layer));
			if (parameters == null) throw new ArgumentNullException(nameof(parameters));
			if (inputs == null) throw new ArgumentNullException(nameof(inputs));
			if (outputs == null) throw new ArgumentNullException(nameof(outputs));
			if (externalInputs == null) throw new ArgumentNullException(nameof(externalInputs));
			if (externalOutputs == null) throw new ArgumentNullException(nameof(externalOutputs));

			Parameters = parameters;
			Inputs = new ReadOnlyDictionary<string, IRegistry>(inputs);
			Outputs = new ReadOnlyDictionary<string, IRegistry>(outputs);
			Layer = layer;
			ExternalInputs = externalInputs;
			ExternalOutputs = externalOutputs;
		}
	}
}
