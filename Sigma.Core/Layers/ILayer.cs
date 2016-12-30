/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
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
		/// The expected input alias names, e.g. "default" or "inputs_0", "inputs_1". 
		/// Note: These names do not represent the actual contents of the input, but the expected amount and alias names of connected inputs. 
		/// </summary>
		string[] ExpectedInputs { get; }

		/// <summary>
		/// The expected output names, e.g. "default" or "outputs_0", "outputs_1".
		/// Note: These names do not represent the actual contents of the input, but the expected amount and alias names of connected inputs. 
		/// </summary>
		string[] ExpectedOutputs { get; }

		/// <summary>
		/// The trainable parameters of this layer, e.g. "weights".
		/// Note that only <see cref="ITraceable"/> parameters can be marked as trainable.
		/// </summary>
		string[] TrainableParameters { get; }

		/// <summary>
		/// The constant, variable and trainable parameters of this layer, e.g. "size", "dropout_probability", or "weights".
		/// </summary>
		IRegistry Parameters { get; }

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry. Each input and each output registry represents one connected layer.
		/// </summary>
		/// <param name="buffer">The buffer containing the inputs, parameters and outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		/// <param name="trainingPass">Indicate whether this is run is part of a training pass.</param>
		void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass);
	}
}
