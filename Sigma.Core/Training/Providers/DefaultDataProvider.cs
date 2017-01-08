/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using LinkAction = System.Action<Sigma.Core.Utils.IRegistry, Sigma.Core.Layers.ILayer, System.Collections.Generic.IDictionary<string, Sigma.Core.MathAbstract.INDArray>>;

namespace Sigma.Core.Training.Providers
{
	/// <summary>
	/// The default data provider, providing default links for default inputs and targets and optional additional input and output links.
	/// </summary>
	public class DefaultDataProvider : IDataProvider
	{
		/// <summary>
		/// The external input links attached to this provider.
		/// </summary>
		public IReadOnlyDictionary<string, LinkAction> ExternalInputLinks { get; }

		/// <summary>
		/// The external output links attached to this provider.
		/// </summary>
		public IReadOnlyDictionary<string, LinkAction> ExternalOutputLinks { get; }

		private readonly IDictionary<string, LinkAction> _externalInputLinks;
		private readonly IDictionary<string, LinkAction> _externalOutputLinks;

		public DefaultDataProvider()
		{
			_externalOutputLinks = new Dictionary<string, LinkAction>();
			_externalInputLinks = new Dictionary<string, LinkAction>();

			ExternalInputLinks = new ReadOnlyDictionary<string, LinkAction>(_externalInputLinks);
			ExternalOutputLinks = new ReadOnlyDictionary<string, LinkAction>(_externalOutputLinks);

			SetupDefaultLinks();
		}

		protected virtual void SetupDefaultLinks()
		{
			SetExternalInputLink("external_default", (inRegistry, layer, currentBlock) => { inRegistry["activations"] = currentBlock["inputs"]; });
			SetExternalInputLink("external_targets", (inRegistry, layer, currentBlock) => { inRegistry["activations"] = currentBlock["targets"]; });
		}

		/// <summary>
		/// Set an external input link for a certain external input alias to a link action.
		/// </summary>
		/// <param name="externalInputAlias">The external input alias to attach the link to.</param>
		/// <param name="linkAction">The link action to execute that supplies the registry with input data for that layer and input alias.</param>
		public void SetExternalInputLink(string externalInputAlias, LinkAction linkAction)
		{
			if (externalInputAlias == null) throw new ArgumentNullException(nameof(externalInputAlias));
			if (linkAction == null) throw new ArgumentNullException(nameof(linkAction));

			_externalInputLinks[externalInputAlias] = linkAction;
		}

		/// <summary>
		/// Remove an external input link with a certain external input alias.
		/// </summary>
		/// <param name="externalInputAlias">The external input alias to detach.</param>
		public void RemoveExternalInputLink(string externalInputAlias)
		{
			if (externalInputAlias == null) throw new ArgumentNullException(nameof(externalInputAlias));

			_externalInputLinks.Remove(externalInputAlias);
		}

		/// <summary>
		/// Set an external output link for a certain external output alias to a link action.
		/// </summary>
		/// <param name="externalOutputAlias">The external output alias to attach the link to.</param>
		/// <param name="linkAction">The link action to execute that supplies the registry with output data for that layer and output alias.</param>
		public void SetExternalOutputLink(string externalOutputAlias, LinkAction linkAction)
		{
			if (externalOutputAlias == null) throw new ArgumentNullException(nameof(externalOutputAlias));
			if (linkAction == null) throw new ArgumentNullException(nameof(linkAction));

			_externalInputLinks[externalOutputAlias] = linkAction;
		}

		/// <summary>
		/// Remove an external output link with a certain external output alias.
		/// </summary>
		/// <param name="externalOutputAlias">The external output alias to detach.</param>
		public void RemoveExternalOutputLink(string externalOutputAlias)
		{
			if (externalOutputAlias == null) throw new ArgumentNullException(nameof(externalOutputAlias));

			_externalInputLinks.Remove(externalOutputAlias);
		}

		/// <summary>
		/// Provide the external input for a certain layer's external input registry. 
		/// </summary>
		/// <param name="externalInputAlias">The alias of the external input. Typically indicates the type of input data to set.</param>
		/// <param name="inputRegistry">The input registry in which to set the external inputs.</param>
		/// <param name="layer">The layer the external input is attached to.</param>
		/// <param name="currentTrainingBlock">The current training block (as provided by the training data iterator).</param>
		public void ProvideExternalInput(string externalInputAlias, IRegistry inputRegistry, ILayer layer, IDictionary<string, INDArray> currentTrainingBlock)
		{
			if (!_externalInputLinks.ContainsKey(externalInputAlias))
			{
				throw new InvalidOperationException($"Cannot provide external input for external input alias {externalInputAlias} for layer {layer}, corresponding external input link is not attached.");
			}

			_externalInputLinks[externalInputAlias].Invoke(inputRegistry, layer, currentTrainingBlock);
		}

		/// <summary>
		/// Provide the external output from a certain layer's external output registry.
		/// </summary>
		/// <param name="externalOutputAlias">The alias of the external output. Typically indicates the type of output data that was set.</param>
		/// <param name="outputRegistry">The output registry from which to get the external outputs.</param>
		/// <param name="layer">The layer the external output is attached to.</param>
		/// <param name="currentTrainingBlock">The current training block (as provided by the training data iterator).</param>
		public void ProvideExternalOutput(string externalOutputAlias, IRegistry outputRegistry, ILayer layer, IDictionary<string, INDArray> currentTrainingBlock)
		{
			if (_externalOutputLinks.ContainsKey(externalOutputAlias))
			{
				_externalOutputLinks[externalOutputAlias].Invoke(outputRegistry, layer, currentTrainingBlock);
			}
		}
	}
}
