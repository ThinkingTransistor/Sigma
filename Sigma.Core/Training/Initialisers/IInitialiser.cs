/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// An initialiser that can be used to initialise any ndarray with a certain pattern (e.g. Gaussian). 
	/// </summary>
	public interface IInitialiser
	{
		/// <summary>
		/// The registry containing relevant parameters and information about this initialiser.
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Initialise a certain ndarray.
		/// </summary>
		/// <param name="array">The ndarray to initialise.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <param name="random">The randomiser to use (if required).</param>
		void Initialise(INDArray array, IComputationHandler handler, Random random);

		/// <summary>
		/// Initialise a single number.
		/// </summary>
		/// <param name="number">The number to initialise.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <param name="random">The randomise to use (if required).</param>
		void Initialise(INumber number, IComputationHandler handler, Random random);
	}
}
