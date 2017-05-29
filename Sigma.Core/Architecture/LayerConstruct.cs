/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

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
		/// The unresolved name of the layer created in this construct (the unresolved name does not have to be unique).
		/// </summary>
		internal string UnresolvedName { get; }

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
		/// Indicate the alias names of the input layers to be supplied externally.
		/// </summary>
		public string[] ExternalInputs { get; internal set; }

		/// <summary>
		/// Indicate the alias names of the output layers to be supplied externally.
		/// </summary>
		public string[] ExternalOutputs { get; internal set; }

		/// <summary>
		/// Update parameters of this construct before instantiation. 
		/// Used for parameters influenced by surrounding constructs which cannot be known at initial creation of LayerConstruct.
		/// </summary>
		public event EventHandler<LayerConstructEventArgs> UpdateBeforeInstantiationEvent;

		/// <summary>
		/// Validate parameters of this construct.
		/// Used for parameters influenced by surrounding constructs which cannot be known at initial creation of LayerConstruct.
		/// </summary>
		public event EventHandler<LayerConstructEventArgs> ValidateEvent;

		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly Type _layerInterfaceType = typeof(ILayer);
		private readonly Type _layerClassType;

		public LayerConstruct(string name, Type layerClassType)
		{
			ValidateLayerName(name);
			ValidateLayerClassType(layerClassType);

			Name = name;
			UnresolvedName = name;
			_layerClassType = layerClassType;

			Inputs = new Dictionary<string, LayerConstruct>();
			Outputs = new Dictionary<string, LayerConstruct>();
			ExternalInputs = new string[0];
			ExternalOutputs = new string[0];
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
				Parameters = (IRegistry) Parameters.DeepCopy(),
				ExternalInputs = (string[]) ExternalInputs.Clone(),
				ExternalOutputs = (string[]) ExternalOutputs.Clone(),
				UpdateBeforeInstantiationEvent = UpdateBeforeInstantiationEvent,
				ValidateEvent = ValidateEvent
			};

			return copy;
		}

		public virtual void Validate()
		{
			ValidateEvent?.Invoke(this, new LayerConstructEventArgs(this));
		}

		public virtual ILayer InstantiateLayer(IComputationHandler handler)
		{
			UpdateBeforeInstantiationEvent?.Invoke(this, new LayerConstructEventArgs(this));

			try
			{
				return (ILayer) Activator.CreateInstance(_layerClassType, Name, Parameters, handler);
			}
			catch (MissingMethodException)
			{
				_logger.Error($"Unable to instantiate layer from construct {Name}. Referenced class type {_layerClassType} is missing required constructor with signature LayerClassName(string name, IRegistry parameters, IComputationHandler handler).");

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

		/// <summary>
		/// Check if the type of two layer constructs is equal.
		/// </summary>
		/// <param name="other">The other layer construct.</param>
		/// <returns>A boolean indicating whether or not the two layer construct types are equal.</returns>
		public bool TypeEquals(LayerConstruct other)
		{
			return other != null && other._layerClassType == _layerClassType;
		}
	}

	public class LayerConstructEventArgs : EventArgs
	{
		public readonly LayerConstruct Self;

		internal LayerConstructEventArgs(LayerConstruct self)
		{
			Self = self;
		}
	}
}
