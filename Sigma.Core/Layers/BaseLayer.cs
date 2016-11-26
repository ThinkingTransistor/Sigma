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
		public string[] ExpectedInputs { get; protected set; } = { "default" };
		public string[] ExpectedOutputs { get; protected set; } = { "default" };
		public string[] TrainableParameters { get; protected set; } = { };
		public IRegistry Parameters { get; }

		/// <summary>
		/// Create a base layer with a certain unique name.
		/// </summary>
		/// <param name="name">The unique name of this layer.</param>
		/// <param name="parameters">The parameters to this layer.</param>
		/// <param name="handler">The handler to use for ndarray parameter creation.</param>
		protected BaseLayer(string name, IRegistry parameters, IComputationHandler handler)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			if (parameters == null)
			{
				throw new ArgumentNullException(nameof(parameters));
			}

			if (handler == null)
			{
				throw new ArgumentNullException(nameof(parameters));
			}

			Name = name;
			Parameters = parameters;
		}

		public abstract void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass = true);
	}
}
