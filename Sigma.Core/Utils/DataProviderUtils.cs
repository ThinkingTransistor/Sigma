using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Providers;

namespace Sigma.Core.Utils
{
	public static class DataProviderUtils
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
	}
}