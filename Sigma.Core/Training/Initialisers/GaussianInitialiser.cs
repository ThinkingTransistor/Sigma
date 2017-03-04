/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// A Gaussian initialiser for ndarrays with mean and standard deviation. 
	/// </summary>
	[Serializable]
	public class GaussianInitialiser : BaseInitialiser
	{
		/// <summary>
		/// Create a Gaussian initialiser with a certain standard deviation and optional mean.
		/// </summary>
		/// <param name="standardDeviation">The standard deviation.</param>
		/// <param name="mean">The mean.</param>
		public GaussianInitialiser(double standardDeviation, double mean = 0.0)
		{
			Registry.Set("mean", mean, typeof(double));
			Registry.Set("standard_deviation", standardDeviation, typeof(double));
		}

		public override object GetValue(long[] indices, long[] shape, Random random)
		{
			// box-muller transform for fast Gaussian values
			// see http://stackoverflow.com/questions/218060/random-gaussian-variables
			double u1 = random.NextDouble(); 
			double u2 = random.NextDouble();
			double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); 

			return Registry.Get<double>("mean") + Registry.Get<double>("standard_deviation") * randStdNormal;
		}
	}
}
