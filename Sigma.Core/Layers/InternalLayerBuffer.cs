﻿/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A default implementation of the <see cref="ILayerBuffer"/> interface for internal use.
	/// </summary>
	public class InternalLayerBuffer : ILayerBuffer
	{
		public ILayer Layer { get; }
		public IRegistry Parameters { get; }
		public bool InputsExternal { get; }
		public bool OutputsExternal { get; }
		public IDictionary<string, IRegistry> Inputs { get; }
		public IDictionary<string, IRegistry> Outputs { get; }

		/// <summary>
		/// Create a layer buffer for a certain layer, parameters, inputs, and outputs.
		/// </summary>
		/// <param name="layer">The layer.</param>
		/// <param name="parameters">The parameters.</param>
		/// <param name="inputs">The inputs.</param>
		/// <param name="outputs">The outputs.</param>
		/// <param name="inputsExternal">Indicate if inputs are external.</param>
		/// <param name="outputsExternal">Indicate if outputs are external.</param>
		public InternalLayerBuffer(ILayer layer, IRegistry parameters, IDictionary<string, IRegistry> inputs, IDictionary<string, IRegistry> outputs,
									bool inputsExternal, bool outputsExternal)
		{
			if (layer == null) throw new ArgumentNullException(nameof(layer));
			if (parameters == null) throw new ArgumentNullException(nameof(parameters));
			if (inputs == null) throw new ArgumentNullException(nameof(inputs));
			if (outputs == null) throw new ArgumentNullException(nameof(outputs));

			Parameters = parameters;
			Inputs = inputs;
			Outputs = outputs;
			Layer = layer;
			InputsExternal = inputsExternal;
			OutputsExternal = outputsExternal;
		}
	}
}