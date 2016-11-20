using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// An initialiser that can be used to initialise any ndarray with a certain pattern (e.g. Gaussian). 
	/// </summary>
	public interface IInitialiser
	{
		/// <summary>
		/// Initialise a certain ndarray.
		/// </summary>
		/// <param name="array">The ndarray to initialise.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <param name="random">The randomiser to use (if required).</param>
		void Initialise(INDArray array, IComputationHandler handler, Random random);
	}
}
