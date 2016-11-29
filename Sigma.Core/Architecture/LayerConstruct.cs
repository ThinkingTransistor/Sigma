/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A layer construct representing a certain named layer construct, where all parameters are stored in a parameter registry.
	/// Note: Layer constructs are like placeholders for actual layers during construction and architecture definition. 
	/// </summary>
	[Serializable]
	public class LayerConstruct
	{
		/// <summary>
		/// The unique name of the layer created in this layer construct.
		/// </summary>
		public string Name { get; set; }

		/// <summary>
		/// The parameters of the layer created in this layer construct.
		/// </summary>
		public IRegistry Parameters { get; protected set; }

		/// <summary>
		/// The inputs connected to this layer construct by their alias name.
		/// </summary>
		public IDictionary<string, LayerConstruct> Inputs { get; }

		/// <summary>
		/// The outputs connected to this layer construct by their alias name.
		/// </summary>
		public IDictionary<string, LayerConstruct> Outputs { get; }

		private readonly Type _layerInterfaceType = typeof(ILayer);
		private readonly Type _layerClassType;

		public LayerConstruct(string name, Type layerClassType)
		{
			ValidateLayerName(name);

			if (layerClassType == null)
			{
				throw new ArgumentNullException(nameof(layerClassType));
			}

			if (layerClassType.IsSubclassOf(_layerInterfaceType))
			{
				throw new ArgumentException($"Layer class type must be subclass of layer interface type IInterface, but was {layerClassType}.");
			}

			_layerClassType = layerClassType;

			Inputs = new Dictionary<string, LayerConstruct>();
			Outputs = new Dictionary<string, LayerConstruct>();
			Parameters = new Registry(tags: "layer");
		}

		private static void ValidateLayerName(string name)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			int autoNameCharacterCount = name.Count(c => c == '#');

			if (autoNameCharacterCount >= 2)
			{
				throw new ArgumentException($"There can be at most one auto name character '#' in a layer name, but given name {name} had more.");
			}
		}

		public virtual LayerConstruct Copy()
		{
			return new LayerConstruct(Name, _layerClassType);
		}

		public virtual ILayer InstantiateLayer(IComputationHandler handler)
		{
			return (ILayer) Activator.CreateInstance(_layerClassType, Parameters, handler);
		}

		public void AddOutput(LayerConstruct output, string alias = "default")
		{
			if (output == null)
			{
				throw new ArgumentNullException(nameof(output));
			}

			if (Outputs.ContainsKey(alias))
			{
				throw new ArgumentException($"Attempted to add duplicate input to construct {Name} with alias {alias}.");
			}

			Outputs.Add(alias, output);
		}

		public void AddInput(LayerConstruct input, string alias = "default")
		{
			if (input == null)
			{
				throw new ArgumentNullException(nameof(input));
			}

			if (Inputs.ContainsKey(alias))
			{
				throw new ArgumentException($"Attempted to add duplicate output to construct {Name} with alias {alias}.");
			}

			Inputs.Add(alias, input);
		}

		public static LinearNetworkArchitecture operator +(LayerConstruct self, LayerConstruct other)
		{
			if (self == null)
			{
				throw new ArgumentNullException(nameof(self));
			}

			if (other == null)
			{
				throw new ArgumentNullException(nameof(other));
			}

			self.AddOutput(other);
			other.AddInput(self);

			return new LinearNetworkArchitecture(self, other);
		}

		public static LinearNetworkArchitecture operator *(int multiplier, LayerConstruct self)
		{
			if (self == null)
			{
				throw new ArgumentNullException(nameof(self));
			}

			return multiplier * new LinearNetworkArchitecture(self);
		}
	}
}
