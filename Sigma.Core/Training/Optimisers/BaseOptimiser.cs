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

namespace Sigma.Core.Training.Optimisers
{
	/// <summary>
	/// A base class for optimisers for easier implementation (all parameters treated as ndarray and passed with identifiers). 
	/// </summary>
	public abstract class BaseOptimiser : IOptimiser
	{
		public void Run(INetwork network, IComputationHandler handler)
		{
			if (network == null) throw new ArgumentNullException(nameof(network));
			if (handler == null) throw new ArgumentNullException(nameof(handler));

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
