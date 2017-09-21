/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Optimisers.Gradient
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

		/// <summary>
		/// The external cost alias to use.
		/// </summary>
		protected readonly string ExternalCostAlias;

		private readonly ISet<Regex> _internalFilterMasks;
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
			Registry["updates"] = new Dictionary<string, INDArray>();
			Registry["filter_masks"] = new HashSet<string>();
			Registry["filtered_identifiers"] = new HashSet<string>();
			Registry["self"] = this;

			_internalFilterMasks = new HashSet<Regex>();
		}

		/// <summary>
		/// Prepare for a single iteration of the network (model) optimisation process (<see cref="IOptimiser.Run"/>).
		/// Typically used to trace trainable parameters to retrieve the derivatives in <see cref="IOptimiser.Run"/>.
		/// </summary>
		/// <param name="network">The network to prepare for optimisation.</param>
		/// <param name="handler">THe handler to use.</param>
		public void PrepareRun(INetwork network, IComputationHandler handler)
		{
			ISet<string> filterMasks = Registry.Get<ISet<string>>("filter_masks");
			ISet<string> filteredIdentifiers = Registry.Get<ISet<string>>("filtered_identifiers");

			filteredIdentifiers.Clear();
			_internalFilterMasks.Clear();

			foreach (string filterMask in filterMasks)
			{
				_internalFilterMasks.Add(new Regex(filterMask));
			}

			_traceTag = handler.BeginTrace();

			foreach (ILayerBuffer layerBuffer in network.YieldLayerBuffersOrdered())
			{
				string layerIdentifier = layerBuffer.Layer.Name;

				foreach (string trainableParameter in layerBuffer.Layer.TrainableParameters)
				{
					string parameterIdentifier = layerIdentifier + "." + trainableParameter;

					bool anyFilterApplies = false;

					foreach (Regex filterMask in _internalFilterMasks)
					{
						if (filterMask.IsMatch(parameterIdentifier))
						{
							anyFilterApplies = true;
							break;
						}
					}

					if (!anyFilterApplies)
					{
						layerBuffer.Layer.Parameters[trainableParameter] = handler.Trace(layerBuffer.Layer.Parameters.Get<ITraceable>(trainableParameter), _traceTag);
					}
					else
					{
						filteredIdentifiers.Add(parameterIdentifier);
					}
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

			ISet<string> filteredIdentifiers = Registry.Get<ISet<string>>("filtered_identifiers");
			IRegistry costRegistry = new Registry(Registry, tags: "costs");
			IRegistry gradientRegistry = new Registry(Registry, tags: "gradients");
			INumber cost = GetTotalCost(network, handler, costRegistry);

			Registry["cost_total"] = cost.GetValueAs<double>();
			Registry["cost_partials"] = costRegistry;
			Registry["gradients"] = gradientRegistry;

			handler.ComputeDerivativesTo(cost);

			foreach (ILayerBuffer layerBuffer in network.YieldLayerBuffersOrdered())
			{
				string layerIdentifier = layerBuffer.Layer.Name;

				foreach (string trainableParameter in layerBuffer.Layer.TrainableParameters)
				{
					object parameter = layerBuffer.Parameters[trainableParameter];
					string parameterIdentifier = layerIdentifier + "." + trainableParameter;

					if (filteredIdentifiers.Contains(parameterIdentifier))
					{
						continue;
					}

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

						handler.FreeLimbo(asArray);

						INDArray newParameter = Optimise(parameterIdentifier, handler.ClearTrace(asArray), gradient, handler);

						handler.MarkLimbo(newParameter);

						layerBuffer.Parameters[trainableParameter] = newParameter;
					}

					layerBuffer.Parameters[trainableParameter] = handler.ClearTrace(layerBuffer.Parameters.Get<ITraceable>(trainableParameter));
				}

				// outputs might have a trace as well, clear everything
				_InternalClearAllTraces(layerBuffer.Inputs, handler);
				_InternalClearAllTraces(layerBuffer.Outputs, handler);
			}
		}

		/// <summary>
		/// Add a specific filter mask ("freeze" a specific part of the model).
		/// Filter masks are registry resolve strings for the model, e.g. layer1.*, *.weights.
		/// </summary>
		/// <param name="filterMask">The filter mask to add ("freeze").</param>
		public void AddFilter(string filterMask)
		{
			Registry.Get<ISet<string>>("filtered_identifiers").Add(filterMask);
		}

		/// <summary>
		/// Remove a specific filter mask ("unfreeze" a specific part of the model).
		/// </summary>
		/// <param name="filterMask">The filter mask to remove ("unfreeze").</param>
		public void RemoveFilter(string filterMask)
		{
			Registry.Get<ISet<string>>("filtered_identifiers").Remove(filterMask);
		}

		/// <summary>
		/// Clear all existing filter masks ("unfreeze" the entire model).
		/// </summary>
		public void ClearFilters()
		{
			Registry.Get<ISet<string>>("filtered_identifiers").Clear();
		}

		/// <summary>
		/// Get a shallow copy of this optimiser with the same parameters (etc. learning / decay rates, filters).		
		/// </summary>
		/// <returns>A shallow copy of this optimiser.</returns>
		public IOptimiser ShallowCopy()
		{
			IOptimiser copy = ShallowCopyParameters();
			ISet<string> filterMasks = Registry.Get<ISet<string>>("filter_masks");

			foreach (string filterMask in filterMasks)
			{
				copy.AddFilter(filterMask);
			}

			return copy;
		}

		/// <summary>
		/// Get a shallow copy of this optimiser with the same specific parameters (all parameters not specified in the <see cref="BaseGradientOptimiser"/>).
		/// </summary>
		/// <returns>A shallow copy of this optimiser with the same specific parameters.</returns>
		protected abstract BaseGradientOptimiser ShallowCopyParameters();

		private static void _InternalClearAllTraces(IReadOnlyDictionary<string, IRegistry> layerExternalBuffer, IComputationHandler handler)
		{
			foreach (string output in layerExternalBuffer.Keys.ToArray())
			{
				IRegistry registry = layerExternalBuffer[output];

				foreach (string parameter in registry.Keys.ToArray())
				{
					ITraceable traceable = registry[parameter] as ITraceable;

					if (traceable != null)
					{
						registry[parameter] = handler.ClearTrace(traceable);
					}
				}
			}
		}

		/// <summary>
		/// Expose a parameter update to the outside through the gradient optimiser utilities.
		/// </summary>
		/// <param name="parameterIdentifier">The parameter identifier.</param>
		/// <param name="update">The update.</param>
		protected void ExposeParameterUpdate(string parameterIdentifier, INDArray update)
		{
			if (parameterIdentifier == null) throw new ArgumentNullException(nameof(parameterIdentifier));
			if (update == null) throw new ArgumentNullException(nameof(update));

			Registry.Get<IDictionary<string, INDArray>>("updates")[parameterIdentifier] = update;
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
		internal abstract INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler);

		/// <summary>
		/// Deep copy this object.
		/// </summary>
		/// <returns>A deep copy of this object.</returns>
		public object DeepCopy()
		{
			IOptimiser copy = ShallowCopy();

			copy.Registry.AddAll((IDictionary<string, object>) copy.Registry.DeepCopy());

			return copy;
		}
	}
}
