/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Persistence.Selectors;
using Sigma.Core.Persistence.Selectors.Network;
using Sigma.Core.Utils;

namespace Sigma.Core.Architecture
{
    /// <summary>
    /// A default implementation of the <see cref="INetwork"/> interface.
    /// Represents a neural network consisting of interconnected neural layers and a network architecture.
    /// </summary>
    [Serializable]
    public class Network : INetwork
    {
        /// <inheritdoc />
        public INetworkArchitecture Architecture { get; set; }

        /// <inheritdoc />
        public string Name { get; }

        /// <inheritdoc />
        public IRegistry Registry { get; }

        /// <summary>
        /// The computation handler associated with this network, which is used for initialisation and copy operations.
        /// Note: Set this 
        /// </summary>
        public IComputationHandler AssociatedHandler
        {
            get { return _associatedHandler; }
            set { _associatedHandler = value; }
        }

        /// <inheritdoc />
        public bool Initialised { get { return _initialised; } }

        [NonSerialized]
        private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
        private readonly List<InternalLayerBuffer> _orderedLayerBuffers;
        private readonly List<InternalLayerBuffer> _externalInputsLayerBuffers;
        private readonly List<InternalLayerBuffer> _externalOutputsLayerBuffers;
        private List<ILayer> _orderedLayers;

        [NonSerialized]
        private IComputationHandler _associatedHandler;
        private bool _initialised;
        
        /// <summary>
        /// Create a network with a certain unique name.
        /// </summary>
        /// <param name="name">The name.</param>
        public Network(string name = "unnamed")
        {
            if (name == null) throw new ArgumentNullException(nameof(name));

            Name = name;
            Registry = new Registry(tags: "network");
            _orderedLayerBuffers = new List<InternalLayerBuffer>();
            _externalInputsLayerBuffers = new List<InternalLayerBuffer>();
            _externalOutputsLayerBuffers = new List<InternalLayerBuffer>();
        }

        /// <inheritdoc />
        public virtual object DeepCopy()
        {
            Network copy = new Network(Name);
            copy.Architecture = (INetworkArchitecture) Architecture.DeepCopy();

            if (_initialised)
            {
                copy.Initialise(_associatedHandler);

                for (int i = 0; i < _orderedLayerBuffers.Count; i++)
                {
                    InternalLayerBuffer originalBuffer = _orderedLayerBuffers[i];
                    InternalLayerBuffer copyBuffer = copy._orderedLayerBuffers[i];

                    foreach (string parameterIdentifier in originalBuffer.Parameters.Keys.ToArray())
                    {
                        object value = originalBuffer.Parameters[parameterIdentifier];
                        IDeepCopyable deepCopyableValue = value as IDeepCopyable;
                        object copiedValue;

                        // copy and copy efficiently by any means possible
                        if (deepCopyableValue == null)
                        {
                            ICloneable cloneableValue = value as ICloneable;
                            copiedValue = cloneableValue?.Clone() ?? value;
                        }
                        else
                        {
                            INDArray asNDArray = value as INDArray;

                            if (asNDArray != null)
                            {
                                _associatedHandler.Fill(asNDArray, copyBuffer.Parameters.Get<INDArray>(parameterIdentifier));
                                copiedValue = copyBuffer.Parameters.Get<INDArray>(parameterIdentifier);
                            }
                            else
                            {
                                copiedValue = deepCopyableValue.DeepCopy();
                            }
                        }

                        copyBuffer.Parameters[parameterIdentifier] = copiedValue;
                    }
                }
            }

            copy.UpdateRegistry();

            return copy;
        }

        /// <inheritdoc />
        public void Validate()
        {
            if (Architecture == null)
            {
                throw new InvalidOperationException("Cannot validate network before assigning a network architecture.");
            }

            Architecture.Validate();
        }

        /// <inheritdoc />
        public void Initialise(IComputationHandler handler)
        {
            if (handler == null) throw new ArgumentNullException(nameof(handler));

            if (Architecture == null)
            {
                throw new InvalidOperationException("Cannot initialise network before assigning a network architecture.");
            }

            _logger.Debug($"Initialising network \"{Name}\" for handler {handler} containing {Architecture.LayerCount} layers...");

            _associatedHandler = handler;

            ITaskObserver prepareTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare);

            Architecture.ResolveAllNames();

            _orderedLayerBuffers.Clear();
            _externalInputsLayerBuffers.Clear();
            _externalOutputsLayerBuffers.Clear();

            Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry> mappedRegistriesByInOutputs = new Dictionary<Tuple<LayerConstruct, LayerConstruct>, IRegistry>();

            foreach (LayerConstruct layerConstruct in Architecture.YieldLayerConstructsOrdered())
            {
                ILayer layer = layerConstruct.InstantiateLayer(handler);

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
                    outputs[externalOutputAlias] = new Registry(tags: "external_output");
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

            UpdateRegistry();

            SigmaEnvironment.TaskManager.EndTask(prepareTask);

            _initialised = true;

            _logger.Debug($"Done initialising network \"{Name}\" for handler {handler} containing {Architecture.LayerCount} layers.");
        }

        protected virtual void UpdateRegistry()
        {
            Registry.Clear();

            Registry["initialised"] = _initialised;
            Registry["self"] = this;
            Registry["name"] = Name;
            Registry["architecture"] = Architecture?.Registry;

            IRegistry layersRegistry = new Registry(Registry);
            Registry["layers"] = layersRegistry;

            foreach (InternalLayerBuffer layerBuffer in _orderedLayerBuffers)
            {
                IRegistry exposedInputs = new Registry(parent: layerBuffer.Layer.Parameters);
                IRegistry exposedOutputs = new Registry(parent: layerBuffer.Layer.Parameters);

                foreach (string input in layerBuffer.Inputs.Keys)
                {
                    exposedInputs[input] = layerBuffer.Inputs[input];
                }

                foreach (string output in layerBuffer.Outputs.Keys)
                {
                    exposedOutputs[output] = layerBuffer.Outputs[output];
                }

                layerBuffer.Layer.Parameters["_inputs"] = exposedInputs;
                layerBuffer.Layer.Parameters["_outputs"] = exposedOutputs;

                if (layerBuffer.ExternalInputs.Length > 0)
                {
                    layerBuffer.Layer.Parameters.Tags.Add("external_input");
                }

                if (layerBuffer.ExternalOutputs.Length > 0)
                {
                    layerBuffer.Layer.Parameters.Tags.Add("external_output");
                }

                layersRegistry[layerBuffer.Layer.Name] = layerBuffer.Layer.Parameters;
            }
        }

        /// <inheritdoc />
        public void Run(IComputationHandler handler, bool trainingPass)
        {
            if (handler == null) throw new ArgumentNullException(nameof(handler));

            foreach (InternalLayerBuffer layerBuffer in _orderedLayerBuffers)
            {
                layerBuffer.Layer.Run(layerBuffer, handler, trainingPass);
            }
        }

        /// <inheritdoc />
        public void Reset()
        {
            _logger.Debug($"Resetting network \"{Name}\" to un-initialised state...");

            _orderedLayerBuffers.Clear();
            _orderedLayers.Clear();
            _externalInputsLayerBuffers.Clear();
            _externalOutputsLayerBuffers.Clear();

            _initialised = false;
            _associatedHandler = null;

            UpdateRegistry();

            _logger.Debug($"Done resetting network \"{Name}\". All layer buffer information was discarded.");
        }

	    /// <summary>
	    /// Transfer this networks' parameters to another network (may be uninitialised).
	    /// </summary>
	    /// <param name="other">The other network.</param>
	    public void TransferParametersTo(INetwork other)
	    {
			if (other == null) throw new ArgumentNullException(nameof(other));

		    if (!Equals(Architecture, other.Architecture))
		    {
			    throw new InvalidOperationException($"Cannot transfer parameters to network of different architecture (own architecture {Architecture} != {other.Architecture}).");
		    }

		    if (!other.Initialised)
		    {
				other.Initialise(_associatedHandler);
			}

		    ILayerBuffer[] otherBuffers = other.YieldLayerBuffersOrdered().ToArray();
		    for (var i = 0; i < _orderedLayerBuffers.Count; i++)
		    {
			    _orderedLayerBuffers[i].Parameters.CopyTo(otherBuffers[i].Parameters);

				// remove exposed data, not part of actual parameters
			    otherBuffers[i].Parameters.Remove("_outputs");
			    otherBuffers[i].Parameters.Remove("_inputs");
		    }

			// update registry if from the same type
			Network otherAsNetwork = other as Network;

			otherAsNetwork?.UpdateRegistry();
	    }

        /// <inheritdoc />
        public IEnumerable<ILayer> YieldLayersOrdered()
        {
            return _orderedLayers;
        }

        /// <inheritdoc />
        public IEnumerable<ILayerBuffer> YieldLayerBuffersOrdered()
        {
            return _orderedLayerBuffers;
        }

        /// <inheritdoc />
        public IEnumerable<ILayerBuffer> YieldExternalInputsLayerBuffers()
        {
            return _externalInputsLayerBuffers;
        }

        /// <inheritdoc />
        public IEnumerable<ILayerBuffer> YieldExternalOutputsLayerBuffers()
        {
            return _externalOutputsLayerBuffers;
        }

        /// <inheritdoc />
        public INetworkSelector<INetwork> Select()
        {
            return new DefaultNetworkSelector<INetwork>(this);
        }
    }
}
