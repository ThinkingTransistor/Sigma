using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
		void Initialise(INDArray array);
	}
}
