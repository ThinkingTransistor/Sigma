/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
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
			int size = parameters.Get<int>("size");
			int inputSize = parameters.Get<int>("default_input_size");

			parameters["weights"] = handler.NDArray(inputSize, size);
			parameters["biases"] = handler.NDArray(size);

			TrainableParameters = new[] { "weights", "biases" };
		}

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry. Each input and each output registry represents one connected layer.
		/// </summary>
		/// <param name="buffer">The buffer containing the inputs, parameters and outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		/// <param name="trainingPass">Indicate whether this is run is part of a training pass.</param>
		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			INDArray inputs = buffer.Inputs["default"].Get<INDArray>("activations");
			INDArray weights = buffer.Parameters.Get<INDArray>("weights");
			INDArray biases = buffer.Parameters.Get<INDArray>("biases");
			string activation = buffer.Parameters.Get<string>("activation");
			long batches = inputs.Shape[0];
			int inputSize = Parameters.Get<int>("default_input_size"), size = Parameters.Get<int>("size");

			INDArray activations = handler.PermuteBatchAndTime(inputs); // BatchTimeFeatures ordering by default, needs to be TimeBatchFeatures for layers operating on the time dimension
			activations = activations.Reshape(activations.Shape[0], activations.Shape[1] * ArrayUtils.Product(2, activations.Shape));
			activations = handler.RowWise(activations, timeSlice =>
			{
				timeSlice = timeSlice.Reshape(inputs.Shape[0], inputSize);
				timeSlice = handler.Dot(timeSlice, weights);
				timeSlice = handler.RowWise(timeSlice, row => handler.Add(row, biases));
				timeSlice = handler.Activation(activation, timeSlice);

				return timeSlice.Reshape(1L, batches * size);
			});

			activations = activations.Reshape(activations.Shape[0], batches, size);
			buffer.Outputs["default"]["activations"] = handler.PermuteBatchAndTime(activations); // TODO are those the right dimensions? they should be...
		}

		public static LayerConstruct Construct(int size, string activation = "sigmoid", string name = "#-fullyconnected")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(RecurrentLayer));

			construct.Parameters["size"] = size;
			construct.Parameters["activation"] = activation;

			// input size is required for instantiation but not known at construction time, so update before instantiation
			construct.UpdateBeforeInstantiationEvent +=
				(sender, args) => args.Self.Parameters["default_input_size"] = args.Self.Inputs["default"].Parameters["size"];

			return construct;
		}
	}
}
