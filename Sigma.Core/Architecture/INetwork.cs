﻿/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.Utils;

namespace Sigma.Core.Architecture
{ 
	/// <summary>
	/// A neural network consisting of interconnected neural layers and a network architecture.
	/// </summary>
	public interface INetwork : IDeepCopyable
	{
		/// <summary>
		/// The architecture of this network.
		/// </summary>
		INetworkArchitecture Architecture { get; set; }

		/// <summary>
		/// The name of this network.
		/// </summary>
		string Name { get; }

		/// <summary>
		/// The registry containing all relevant parameters and meaningful sub-registries (e.g. layers, architecture).
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Validate this network (e.g. ensure all connections are correctly assigned and compatible). 
		/// </summary>
		void Validate();

		/// <summary>
		/// Initialise this network and create all required layers as defined in the architecture constructs.
		/// Initialisers are NOT executed in this step as initialiser are associated with trainers, not networks.
		/// </summary>
		/// <param name="handler"></param>
		void Initialise(IComputationHandler handler);

		/// <summary>
		/// Run this network (forward pass). All external inputs and outputs must already be supplied and linked. 
		/// </summary>
		/// <param name="handler">The computation handler to use.</param>
		/// <param name="trainingPass">Indicate if this run is part of a training pass.</param>
		void Run(IComputationHandler handler, bool trainingPass);

		/// <summary>
		/// Get the layers of this network in the order they should be processed. 
		/// </summary>
		/// <returns></returns>
		IEnumerable<ILayer> YieldLayersOrdered();

		/// <summary>
		/// Get the layer buffers of this network in the order they should be processed. 
		/// </summary>
		/// <returns></returns>
		IEnumerable<ILayerBuffer> YieldLayerBuffersOrdered();

		/// <summary>
		/// Get the layer buffers of this network that are marked as having external inputs.
		/// </summary>
		/// <returns></returns>
		IEnumerable<ILayerBuffer> YieldExternalInputsLayerBuffers();

		/// <summary>
		/// Get the layer buffers of this network that are marked as having external outputs.
		/// </summary>
		/// <returns></returns>
		IEnumerable<ILayerBuffer> YieldExternalOutputsLayerBuffers();
	}
}