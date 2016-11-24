/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A neural layer in a neural network model with some [parameters].
	/// </summary>
	public interface ILayer
	{
		/// <summary>
		/// The unique name of this layer. 
		/// </summary>
		string Name { get; }

		/// <summary>
		/// The trainable parameters of this layer, e.g. "weights".
		/// </summary>
		string[] TrainableParameters { get; }

		/// <summary>
		/// The parameters of this layer, e.g. "weights", "size", "dropout_probability".
		/// </summary>
		IRegistry Parameters { get; }

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry.
		/// </summary>
		/// <param name="inputs">The inputs respective to this layer.</param>
		/// <param name="parameters">The parameters of this layer.</param>
		/// <param name="outputs">The outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		void Run(IRegistry inputs, IRegistry parameters, IRegistry outputs, IComputationHandler handler);
	}
}
