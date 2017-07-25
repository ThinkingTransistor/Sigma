/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Providers;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility methods for providing and managing network data.
	/// </summary>
	public static class DataUtils
	{
		public static void ProvideExternalInputData(INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			ProvideExternalInputData(new DefaultDataProvider(), localNetwork, currentBlock);
		}

		public static void ProvideExternalInputData(IDataProvider dataProvider, INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			foreach (ILayerBuffer layerBuffer in localNetwork.YieldExternalInputsLayerBuffers())
			{
				foreach (string externalInputAlias in layerBuffer.ExternalInputs)
				{
					dataProvider.ProvideExternalInput(externalInputAlias, layerBuffer.Inputs[externalInputAlias], layerBuffer.Layer, currentBlock);
				}
			}
		}

		public static void ProvideExternalOutputData(INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			ProvideExternalOutputData(new DefaultDataProvider(), localNetwork, currentBlock);
		}

		public static void ProvideExternalOutputData(IDataProvider dataProvider, INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			foreach (ILayerBuffer layerBuffer in localNetwork.YieldExternalOutputsLayerBuffers())
			{
				foreach (string externalOutputAlias in layerBuffer.ExternalOutputs)
				{
					dataProvider.ProvideExternalOutput(externalOutputAlias, layerBuffer.Outputs[externalOutputAlias], layerBuffer.Layer, currentBlock);
				}
			}
		}

		/// <summary>
		/// Make a data block out of a number of (name, ndarray) pairs. 
		/// </summary>
		/// <param name="blockData">The block data in the form of (name, ndarray) [(name, ndarray)] ...</param>
		/// <returns>The block resulting from the given block data.</returns>
		public static IDictionary<string, INDArray> MakeBlock(params object[] blockData)
		{
			if (blockData.Length % 2 != 0) throw new ArgumentException($"Block data must be in pairs of 2 (name, ndarray) but length was {blockData.Length}.");

			IDictionary<string, INDArray> block = new Dictionary<string, INDArray>();

			for (int i = 0; i < blockData.Length; i += 2)
			{
				string name = blockData[i] as string;
				INDArray array = blockData[i + 1] as INDArray;

				if (name == null) throw new ArgumentException($"Name must be of type string and non-null, but name at index {i} was {blockData[i]}");
				if (array == null) throw new ArgumentException($"Array must be of type INDArray and non-null, but array at index {i + 1} was {blockData[i + 1]}");
				if (block.ContainsKey(name)) throw new ArgumentException($"Duplicate name at index {i}: {name} already exists in this block.");

				block.Add(name, array);
			}

			return block;
		}
	}
}