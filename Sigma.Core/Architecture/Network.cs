/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.Utils;

namespace Sigma.Core.Architecture
{
	public class Network : INetwork
	{
		public INetworkArchitecture Architecture { get; set; }
		public string Name { get; }
		public IRegistry Registry { get; }

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private readonly List<InternalLayerBuffer> _orderedLayerBuffers;
		private readonly List<InternalLayerBuffer> _externalInputsLayerBuffers;
		private readonly List<InternalLayerBuffer> _externalOutputsLayerBuffers;
		private List<ILayer> _orderedLayers;

		public Network(string name = "unnamed")
		{
			if (name == null) throw new ArgumentNullException(nameof(name));

			Name = name;
			Registry = new Registry(tags: "network");
			_orderedLayerBuffers = new List<InternalLayerBuffer>();
			_externalInputsLayerBuffers = new List<InternalLayerBuffer>();
			_externalOutputsLayerBuffers = new List<InternalLayerBuffer>();
		}

		public void Validate()
		{
			if (Architecture == null)
			{
				throw new InvalidOperationException("Cannot validate network before assigning a network architecture.");
			}

			Architecture.Validate();
		}

		public void Initialise(IComputationHandler handler)
		{
			if (Architecture == null)
			{
				throw new InvalidOperationException("Cannot initialise network before assigning a network architecture.");
			}

			_logger.Info($"Initialising network \"{Name}\" for handler {handler} containing {Architecture.LayerCount} layers...");

			ITaskObserver prepareTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare);

			Architecture.ResolveAllNames();

			_orderedLayerBuffers.Clear();
			_externalInputsLayerBuffers.Clear();
			_externalOutputsLayerBuffers.Clear();

			Registry.Clear();

			Registry layersRegistry = new Registry(Registry);
			Registry["layers"] = layersRegistry;

			Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry> mappedRegistriesByInOutputs = new Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry>();

			foreach (LayerConstruct layerConstruct in Architecture.YieldLayerConstructsOrdered())
			{
				ILayer layer = layerConstruct.InstantiateLayer(handler);

				layersRegistry[layer.Name] = layerConstruct.Parameters.DeepCopy();

				Dictionary<string, IRegistry> inputs = new Dictionary<string, IRegistry>();

				foreach (string externalInputAlias in layerConstruct.ExternalInputs)
				{
					inputs[externalInputAlias] = new Registry(tags: "external_input");
				}

				foreach (string inputAlias in layerConstruct.Inputs.Keys)
				{
					inputs[inputAlias] = mappedRegistriesByInOutputs[new Tuple<LayerConstruct, LayerConstruct>(layerConstruct.Inputs[inputAlias], layerConstruct)];
				}

				Dictionary<string, IRegistry> outputs = new Dictionary<string, IRegistry>();

				foreach (string externalOutputAlias in layerConstruct.ExternalOutputs)
				{
					inputs[externalOutputAlias] = new Registry(tags: "external_output");
				}

				foreach (string outputAlias in layerConstruct.Outputs.Keys)
				{
					LayerConstruct outputConstruct = layerConstruct.Outputs[outputAlias];

					Tuple<LayerConstruct, LayerConstruct> inOuTuple = new Tuple<LayerConstruct, LayerConstruct>(layerConstruct, outputConstruct);

					Registry outRegistry = new Registry(tags: "internal");

					mappedRegistriesByInOutputs.Add(inOuTuple, outRegistry);

					outputs[outputAlias] = outRegistry;
				}

				InternalLayerBuffer layerBuffer = new InternalLayerBuffer(layer, layerConstruct.Parameters, inputs, outputs,
					layerConstruct.ExternalInputs, layerConstruct.ExternalOutputs);

				_orderedLayerBuffers.Add(layerBuffer);

				if (layerConstruct.ExternalInputs.Length > 0)
				{
					_externalInputsLayerBuffers.Add(layerBuffer);
				}

				if (layerConstruct.ExternalOutputs.Length > 0)
				{
					_externalOutputsLayerBuffers.Add(layerBuffer);
				}
			}

			_orderedLayers = _orderedLayerBuffers.ConvertAll(buffer => buffer.Layer);

			SigmaEnvironment.TaskManager.EndTask(prepareTask);

			_logger.Info($"Done initialising network \"{Name}\" for handler {handler} containing {Architecture.LayerCount} layers.");
		}

		public IEnumerable<ILayer> YieldLayersOrdered()
		{
			return _orderedLayers;
		}

		public IEnumerable<ILayerBuffer> YieldLayerBuffersOrdered()
		{
			return _orderedLayerBuffers;
		}

		public IEnumerable<ILayerBuffer> YieldExternalInputsLayerBuffers()
		{
			return _externalInputsLayerBuffers;
		}

		public IEnumerable<ILayerBuffer> YieldExternalOutputsLayerBuffers()
		{
			return _externalOutputsLayerBuffers;
		}
	}
}
