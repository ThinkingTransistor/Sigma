/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
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
	public abstract class BaseGradientOptimiser : IOptimiser
	{
		private readonly string _externalCostAlias;

		protected BaseGradientOptimiser(string externalCostAlias = "external_cost")
		{
			if (externalCostAlias == null) throw new ArgumentNullException(nameof(externalCostAlias));

			_externalCostAlias = externalCostAlias;
		}

		public void Run(INetwork network, IComputationHandler handler, IRegistry registry)
		{
			if (network == null) throw new ArgumentNullException(nameof(network));
			if (handler == null) throw new ArgumentNullException(nameof(handler));
			if (registry == null) throw new ArgumentNullException(nameof(registry));

			IRegistry costRegistry = new Registry(registry, tags: "cost");
			registry["costs"] = costRegistry;

			INumber cost = GetTotalCost(network, handler, costRegistry);

			handler.ComputeDerivativesTo(cost);

			foreach (ILayerBuffer layerBuffer in network.YieldLayerBuffersOrdered())
			{
				string layerIdentifier = network.Name + "." + layerBuffer.Layer.Name;

				foreach (string trainableParameter in layerBuffer.Layer.TrainableParameters)
				{
					object parameter = layerBuffer.Parameters[trainableParameter];
					string parameterIdentifier = layerIdentifier + "." + trainableParameter;

					INumber asNumber = parameter as INumber;

					if (asNumber != null)
					{
						// boxing the numbers to ndarrays is easier to work with and the performance difference is completely negligible 
						//  (especially considering that ndarrays are far more common as trainable parameters).
						INDArray convertedNumber = handler.AsNDArray(asNumber);
						INDArray convertedGradient = handler.AsNDArray(handler.GetDerivative(asNumber));

						layerBuffer.Parameters[trainableParameter] = handler.AsNumber(Optimise(parameterIdentifier, convertedNumber, convertedGradient, handler), 0, 0);
					}
					else
					{
						INDArray asArray = parameter as INDArray;

						if (asArray != null)
						{
							layerBuffer.Parameters[trainableParameter] = Optimise(parameterIdentifier, asArray, handler.GetDerivative(asArray), handler);
						}
						else
						{
							throw new InvalidOperationException($"Cannot optimise non-ndarray and non-number parameter \"{parameter}\" (identifier \"{parameterIdentifier}\"" +
																$" in layer \"{layerBuffer.Layer.Name}\") but it is marked as trainable.");
						}
					}
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
				if (layerBuffer.Outputs.ContainsKey(_externalCostAlias))
				{
					IRegistry externalCostRegistry = layerBuffer.Outputs[_externalCostAlias];

					INumber partialCost = externalCostRegistry.Get<INumber>("cost");
					float partialImportance = externalCostRegistry.Get<float>("importance");

					costRegistry["partial_" + layerBuffer.Layer.Name] = partialCost;
					costRegistry["partial_" + layerBuffer.Layer.Name + "_importance"] = partialImportance;

					totalCost = handler.Add(totalCost, handler.Multiply(partialCost, partialImportance));
				}
			}

			costRegistry["total"] = totalCost;

			return totalCost;
		}

		/// <summary>
		/// Optimise a certain parameter given a certain gradient using a certain computation handler.
		/// </summary>
		/// <param name="parameterIdentifier">The parameter identifier </param>
		/// <param name="parameter"></param>
		/// <param name="gradient"></param>
		/// <param name="handler"></param>
		/// <returns></returns>
		protected abstract INDArray Optimise(string parameterIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler);
	}
}
