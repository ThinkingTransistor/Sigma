/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Layers.Regularisation
{
	/// <summary>
	/// A standard dropout layer applying a random probability mask with a given dropout probability during training.
	/// </summary>
	[Serializable]
	public class DropoutLayer : BaseLayer
	{
		/// <summary>
		/// Create a base layer with a certain unique name.
		/// </summary>
		/// <param name="name">The unique name of this layer.</param>
		/// <param name="parameters">The parameters to this layer.</param>
		/// <param name="handler">The handler to use for ndarray parameter creation.</param>
		public DropoutLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			parameters["dropout_mask"] = handler.NDArray(parameters.Get<int>("size"));
		}

		/// <summary>
		/// Run this layer. Take relevant input values from inputs and put relevant output values in outputs registry. Each input and each output registry represents one connected layer.
		/// </summary>
		/// <param name="buffer">The buffer containing the inputs, parameters and outputs respective to this layer.</param>
		/// <param name="handler">The computation handler to use for computations (duh).</param>
		/// <param name="trainingPass">Indicate whether this is run is part of a training pass.</param>
		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			if (trainingPass)
			{
				INDArray inputs = buffer.Inputs["default"].Get<INDArray>("activations");
				INDArray activations = handler.FlattenTimeAndFeatures(inputs);
				INDArray dropoutMask = Parameters.Get<INDArray>("dropout_mask");

				activations = handler.RowWise(activations, row =>
				{
					handler.FillWithProbabilityMask(dropoutMask, 1.0 - Parameters.Get<double>("dropout_probability"));

					return handler.Multiply(row, dropoutMask);
				});

				buffer.Outputs["default"]["activations"] = activations.Reshape((long[]) inputs.Shape.Clone());
			}
			else
			{
				buffer.Outputs["default"]["activations"] = buffer.Inputs["default"]["activations"];
			}
		}

		public static LayerConstruct Construct(double dropoutProbability, string name = "#-dropout")
		{
			if (dropoutProbability < 0.0 || dropoutProbability >= 1.0)
			{
				throw new ArgumentException($"Dropout probability must be in range 0.0 <= x < 1.0 but was {dropoutProbability}.");
			}

			LayerConstruct construct = new LayerConstruct(name, typeof(DropoutLayer));

			construct.Parameters["dropout_probability"] = dropoutProbability;

			construct.UpdateBeforeInstantiationEvent +=
				(sender, args) => args.Self.Parameters["size"] = args.Self.Inputs["default"].Parameters["size"];

			return construct;
		}
	}
}
