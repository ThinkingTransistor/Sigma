/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers.Recurrent
{
	public class RecurrentLayer : BaseLayer
	{
		/// <summary>
		/// Create a base layer with a certain unique name.
		/// Note: Do NOT change the signature of this constructor, it's used to dynamically instantiate layers.
		/// </summary>
		/// <param name="name">The unique name of this layer.</param>
		/// <param name="parameters">The parameters to this layer.</param>
		/// <param name="handler">The handler to use for ndarray parameter creation.</param>
		public RecurrentLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
		}

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry. Each input and each output registry represents one connected layer.
		/// </summary>
		/// <param name="buffer">The buffer containing the inputs, parameters and outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		/// <param name="trainingPass">Indicate whether this is run is part of a training pass.</param>
		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			throw new NotImplementedException();
		}
	}
}
