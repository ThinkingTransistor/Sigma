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
using log4net;
using Sigma.Core.Architecture.Linear;

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

		/// <summary>
		/// Indicate if the inputs of this layer are supplied externally.
		/// If set, this layer cannot have any internal inputs.
		/// </summary>
		public bool InputsExternal { get; internal set; }

		/// <summary>
		/// Indicate if the outputs of this layer are supplied externally.
		/// If set, this layer cannot have any internal outputs.
		/// </summary>
		public bool OutputsExternal { get; internal set; }

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly Type _layerInterfaceType = typeof(ILayer);
		private readonly Type _layerClassType;

		public LayerConstruct(string name, Type layerClassType)
		{
			ValidateLayerName(name);
			ValidateLayerClassType(layerClassType);

			Name = name;
			_layerClassType = layerClassType;

			Inputs = new Dictionary<string, LayerConstruct>();
			Outputs = new Dictionary<string, LayerConstruct>();
			Parameters = new Registry(tags: "layer");
		}

		private void ValidateLayerClassType(Type layerClassType)
		{
			if (layerClassType == null)
			{
				throw new ArgumentNullException(nameof(layerClassType));
			}

			if (layerClassType.IsSubclassOf(_layerInterfaceType))
			{
				throw new ArgumentException($"Layer class type must be subclass of layer interface type ILayer, but was {layerClassType}.");
			}
		}

		private static void ValidateLayerName(string name)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			int autoNameCharacterCount = name.Count(c => c == '#');

			if (autoNameCharacterCount > 1)
			{
				throw new ArgumentException($"There can be at most one auto name character '#' in a layer name, but given name {name} had more.");
			}
		}

		public virtual LayerConstruct Copy()
		{
			LayerConstruct copy = new LayerConstruct(Name, _layerClassType)
			{
				Parameters = Parameters,
				InputsExternal = InputsExternal,
				OutputsExternal = OutputsExternal
			};

			return copy;
		}

		public virtual ILayer InstantiateLayer(IComputationHandler handler)
		{
			try
			{
				return (ILayer) Activator.CreateInstance(_layerClassType, Name, Parameters, handler);
			}
			catch (MissingMethodException)
			{
				_logger.Error($"Unable to instantiate layer from construct. Referenced class type is missing required constructor with signature LayerClassName(string name, IRegistry parameters, IComputationHandler handler).");

				throw;
			}
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
