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
		/// The expected input names, e.g. "default" or "inputs_0", "inputs_1". 
		/// Note: These names do not represent the actual contents of the input, but the expected amount and names of connected inputs. 
		/// </summary>
		string[] ExpectedInputs { get; }

		/// <summary>
		/// The expected output names, e.g. "default" or "outputs_0", "outputs_1".
		/// Note: These names do not represent the actual contents of the input, but the expected amount and names of connected inputs. 
		/// </summary>
		string[] ExpectedOutputs { get; }

		/// <summary>
		/// The trainable parameters of this layer, e.g. "weights".
		/// </summary>
		string[] TrainableParameters { get; }

		/// <summary>
		/// The parameters of this layer, e.g. "weights", "size", "dropout_probability".
		/// </summary>
		IRegistry Parameters { get; }

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry. Each input and each output registry represents one connected layer.
		/// </summary>
		/// <param name="inputs">The inputs respective to this layer.</param>
		/// <param name="parameters">The parameters of this layer.</param>
		/// <param name="outputs">The outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		/// <param name="trainingPass">Indicate whether this is run is part of a training pass.</param>
		void Run(AliasRegistry inputs, IRegistry parameters, AliasRegistry outputs, IComputationHandler handler, bool trainingPass = true);
	}
}
