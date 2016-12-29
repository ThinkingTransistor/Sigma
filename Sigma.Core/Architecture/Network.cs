/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
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

		private readonly List<InternalLayerBuffer> _orderedLayerBuffers;
		private List<ILayer> _orderedLayers;

		public Network(string name = "unnamed")
		{
			if (name == null) throw new ArgumentNullException(nameof(name));

			Name = name;
			Registry = new Registry();
			_orderedLayerBuffers = new List<InternalLayerBuffer>();
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

			Architecture.ResolveAllNames();

			_orderedLayerBuffers.Clear();
			Registry.Clear();
			Registry layersRegistry = new Registry(Registry);
			Registry["layers"] = layersRegistry;

			Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry> mappedRegistriesByInOutputs = new Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry>();

			foreach (LayerConstruct layerConstruct in Architecture.YieldLayerConstructsOrdered())
			{
				ILayer layer = layerConstruct.InstantiateLayer(handler);

				layersRegistry[layer.Name] = layerConstruct.Parameters;

				Dictionary<string, IRegistry> inputs = new Dictionary<string, IRegistry>();

				if (layerConstruct.InputsExternal)
				{
					inputs["external"] = new Registry(tags: "external");
				}
				else
				{
					foreach (string inputAlias in layerConstruct.Inputs.Keys)
					{
						inputs[inputAlias] = mappedRegistriesByInOutputs[new Tuple<LayerConstruct, LayerConstruct>(layerConstruct.Inputs[inputAlias], layerConstruct)];
					}
				}

				Dictionary<string, IRegistry> outputs = new Dictionary<string, IRegistry>();

				if (layerConstruct.OutputsExternal)
				{
					outputs["external"] = new Registry(tags: "external");
				}
				else
				{
					foreach (string outputAlias in layerConstruct.Outputs.Keys)
					{
						LayerConstruct outputConstruct = layerConstruct.Outputs[outputAlias];

						Tuple<LayerConstruct, LayerConstruct> inOuTuple = new Tuple<LayerConstruct, LayerConstruct>(layerConstruct, outputConstruct);

						Registry outRegistry = new Registry(tags: "internal");

						mappedRegistriesByInOutputs.Add(inOuTuple, outRegistry);

						outputs[outputAlias] = outRegistry;
					}
				}

				_orderedLayerBuffers.Add(new InternalLayerBuffer(layer, layerConstruct.Parameters, inputs, outputs));
			}

			_orderedLayers = _orderedLayerBuffers.ConvertAll(buffer => buffer.Layer);
		}

		public IEnumerable<ILayer> YieldLayersOrdered()
		{
			return _orderedLayers;
		}

		public IEnumerable<ILayerBuffer> YieldLayerBuffersOrdered()
		{
			return _orderedLayerBuffers;
		}
	}
}
