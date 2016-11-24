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
	/// A basic base layer to simplify custom layer implementations of the ILayer interface. 
	/// </summary>
	public abstract class BaseLayer : ILayer
	{
		public string Name { get; }
		public string[] TrainableParameters { get; protected set; }
		public IRegistry Parameters { get; }

		protected BaseLayer(string name)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			Name = name;
			Parameters = new Registry(tags: "layer");
		}

		public abstract void Run(IRegistry inputs, IRegistry parameters, IRegistry outputs, IComputationHandler handler);
	}
}
