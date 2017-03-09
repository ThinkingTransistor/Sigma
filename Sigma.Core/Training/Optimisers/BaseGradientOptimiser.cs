/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Optimisers
{
	/// <summary>
	/// A base class for gradient based optimisers for easier implementation (all parameters treated as ndarray and passed with identifiers). 
	/// Provides a default implementation of the cost calculation algorithm. 
	/// </summary>
	[Serializable]
	public abstract class BaseGradientOptimiser : IOptimiser
	{
		/// <summary>
		/// The registry containing data about this optimiser and its last run.
		/// </summary>
		public IRegistry Registry { get; }

		protected readonly string ExternalCostAlias;

		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private bool _prepared;
		private uint _traceTag;

		/// <summary>
		/// Create a base gradient optimiser with an optional external output cost alias to use. 
		/// </summary>
		/// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
		protected BaseGradientOptimiser(string externalCostAlias = "external_cost")
		{
			if (externalCostAlias == null) throw new ArgumentNullException(nameof(externalCostAlias));

			ExternalCostAlias = externalCostAlias;
			Registry = new Registry(tags: "optimiser");
		}

		/// <summary>
		/// Prepare for a single iteration of the network (model) optimisation process (<see cref="IOptimiser.Run"/>).
		/// Typically used to trace trainable parameters to retrieve the derivatives in <see cref="IOptimiser.Run"/>.
		/// </summary>
		/// <param name="network">The network to prepare for optimisation.</param>
		/// <param name="handler">THe handler to use.</param>
		public void PrepareRun(INetwork network, IComputationHandler handler)
		{
			_traceTag = handler.BeginTrace();

			foreach (ILayerBuffer layerBuffer in network.YieldLayerBuffersOrdered())
			{
				foreach (string identifier in layerBuffer.Layer.TrainableParameters)
				{
					layerBuffer.Layer.Parameters[identifier] = handler.Trace(layerBuffer.Layer.Parameters.Get<ITraceable>(identifier), _traceTag);
				}
			}

			_prepared = true;
		}

		public void Run(INetwork network, IComputationHandler handler)
		{
			if (network == null) throw new ArgumentNullException(nameof(network));
			if (handler == null) throw new ArgumentNullException(nameof(handler));

			if (!_prepared)
			{
				throw new InvalidOperationException($"Cannot run network optimisation on network {network} in optimiser {this} before {nameof(PrepareRun)} is called.");
			}

			IRegistry costRegistry = new Registry(Registry, tags: "costs");
			Registry["cost_partials"] = costRegistry;

			IRegistry gradientRegistry = new Registry(Registry, tags: "gradients");
			Registry["gradients"] = gradientRegistry;

			INumber cost = GetTotalCost(network, handler, costRegistry);
			Registry["cost_total"] = cost.GetValueAs<double>();

			handler.ComputeDerivativesTo(cost);

			foreach (ILayerBuffer layerBuffer in network.YieldLayerBuffersOrdered())
			{
				string layerIdentifier = layerBuffer.Layer.Name;

				foreach (string trainableParameter in layerBuffer.Layer.TrainableParameters)
				{
					object parameter = layerBuffer.Parameters[trainableParameter];
					string parameterIdentifier = layerIdentifier + "." + trainableParameter;

					INumber asNumber = parameter as INumber;
					INDArray asArray = parameter as INDArray;

					if (asNumber == null && asArray == null)
					{
						throw new InvalidOperationException($"Cannot optimise non-ndarray and non-number parameter \"{parameter}\" (identifier \"{parameterIdentifier}\"" +
									$" in layer \"{layerBuffer.Layer.Name}\") but it is marked as trainable.");
					}

					if (asNumber != null)
					{
						// boxing the numbers to ndarrays is easier to work with and the performance difference is completely negligible
						//  (especially considering that ndarrays are far more common as trainable parameters).
						//  if you think otherwise, implement your own gradient optimiser and do it your way
						INDArray convertedNumber = handler.ClearTrace(handler.AsNDArray(asNumber));
						INDArray convertedGradient = handler.ClearTrace(handler.AsNDArray(handler.GetDerivative(asNumber)));

						gradientRegistry[parameterIdentifier] = convertedGradient;

						layerBuffer.Parameters[trainableParameter] = handler.AsNumber(Optimise(parameterIdentifier, convertedNumber, convertedGradient, handler), 0, 0);
					}
					else
					{
						INDArray gradient = handler.ClearTrace(handler.GetDerivative(asArray));

						gradientRegistry[parameterIdentifier] = gradient;

						layerBuffer.Parameters[trainableParameter] = Optimise(parameterIdentifier, handler.ClearTrace(asArray), gradient, handler);
					}

					layerBuffer.Parameters[trainableParameter] = handler.ClearTrace(layerBuffer.Parameters.Get<ITraceable>(trainableParameter));
				}
			}
		}

		/// <summary>
		/// Get the total cost from a certain network using a certain computation handler and put the relevant information in the cost registry (total, partial, importances).
		/// </summary>
		/// <param name="network">The network to get the costs from.</param>
		/// <param name="handler">The handler to use.</param>
		/// <param name="costRegistry">The registry to put relevant information in.</param>
		/// <returns>The total cost of the given network (0.0 if none).</returns>
		protected virtual INumber GetTotalCost(INetwork network, IComputationHandler handler, IRegistry costRegistry)
		{
			INumber totalCost = handler.Number(0);

			foreach (ILayerBuffer layerBuffer in network.YieldExternalOutputsLayerBuffers())
			{
				if (layerBuffer.Outputs.ContainsKey(ExternalCostAlias))
				{
					IRegistry externalCostRegistry = layerBuffer.Outputs[ExternalCostAlias];

					INumber partialCost = externalCostRegistry.Get<INumber>("cost");
					double partialImportance = externalCostRegistry.Get<double>("importance");

					costRegistry["partial_" + layerBuffer.Layer.Name] = partialCost.GetValueAs<double>();
					costRegistry["partial_" + layerBuffer.Layer.Name + "_importance"] = partialImportance;

					totalCost = handler.Add(totalCost, handler.Multiply(partialCost, partialImportance));
				}
			}

			costRegistry["total"] = totalCost.GetValueAs<double>();

			return totalCost;
		}

		/// <summary>
		/// Optimise a certain parameter given a certain gradient using a certain computation handler.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier (e.g. "3-elementwise.weights").</param>
		/// <param name="parameter">The parameter to optimise.</param>
		/// <param name="gradient">The gradient of the parameter respective to the total cost.</param>
		/// <param name="handler">The handler to use.</param>
		/// <returns>The optimised parameter.</returns>
		protected abstract INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler);

		/// <summary>
		/// Deep copy this object.
		/// </summary>
		/// <returns>A deep copy of this object.</returns>
		public abstract object DeepCopy();
	}
}
