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
		/// The inputs connected to this layer construct. 
		/// </summary>
		public ISet<LayerConstruct> Inputs { get; }

		/// <summary>
		/// The outputs connected to this layer construct. 
		/// </summary>
		public ISet<LayerConstruct> Outputs { get; }

		private readonly Type _layerInterfaceType = typeof(ILayer);
		private readonly Type _layerClassType;

		protected LayerConstruct(Type layerClassType)
		{
			if (layerClassType == null)
			{
				throw new ArgumentNullException(nameof(layerClassType));
			}

			if (layerClassType.IsSubclassOf(_layerInterfaceType))
			{
				throw new ArgumentException($"Layer class type must be subclass of layer interface type IInterface, but was {layerClassType}.");
			}

			_layerClassType = layerClassType;

			Inputs = new HashSet<LayerConstruct>();
			Outputs = new HashSet<LayerConstruct>();
		}

		public virtual LayerConstruct Copy()
		{
			return new LayerConstruct(_layerClassType);
		}

		public virtual ILayer InstantiateLayer(IComputationHandler handler)
		{
			return (ILayer) Activator.CreateInstance(_layerClassType, Parameters);
		}

		public void AddOutput(LayerConstruct output)
		{
			if (output == null)
			{
				throw new ArgumentNullException(nameof(output));
			}

			Outputs.Add(output);
		}

		public void AddInput(LayerConstruct input)
		{
			if (input == null)
			{
				throw new ArgumentNullException(nameof(input));
			}

			Inputs.Add(input);
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
