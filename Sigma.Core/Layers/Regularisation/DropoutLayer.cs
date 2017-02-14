using System;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers.Regularisation
{
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
			throw new NotImplementedException();

			if (trainingPass)
			{
				INDArray activations = buffer.Inputs["default"].Get<INDArray>("activations");

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

			construct.Parameters["drop_probability"] = dropoutProbability;

			construct.UpdateBeforeInstantiationEvent +=
				(sender, args) => args.Self.Parameters["size"] = args.Self.Inputs["default"].Parameters["size"];

			return construct;
		}
	}
}
