/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// A layer in a neural network model.
	/// </summary>
	public interface ILayer
	{
		/// <summary>
		/// The unique name of this layer. 
		/// </summary>
		string Name { get; }

		IRegistry Parameters { get; }

		void Run(IRegistry inputs, IRegistry parameters, IRegistry outputs);
	}

	/// <summary>
	/// A layer construct representing a certain named layer construct, where all parameters are stored in a parameter registry.
	/// </summary>
	public abstract class LayerConstruct
	{
		/// <summary>
		/// The unique name of the layer created in this layer construct.
		/// </summary>
		public string Name { get; set; }

		/// <summary>
		/// The parameters of the layer created in this layer construct.
		/// </summary>
		public IRegistry Parameters { get; protected set; }

		private readonly Type LayerInterfaceType = typeof(ILayer);
		private Type _layerClassType;

		protected LayerConstruct(Type layerClassType)
		{
			if (layerClassType == null)
			{
				throw new ArgumentNullException(nameof(layerClassType));
			}

			if (layerClassType.IsSubclassOf(LayerInterfaceType))
			{
				throw new ArgumentException($"Layer class type must be subclass of layer interface type IInterface, but was {layerClassType}.");
			}
		}

		public virtual ILayer InstantiateLayer(IComputationHandler handler)
		{
			return (ILayer) Activator.CreateInstance(_layerClassType, Parameters);
		}
	}
}
