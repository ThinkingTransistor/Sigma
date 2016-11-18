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

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// A Gaussian initialiser for ndarrays with mean and standard deviation. 
	/// </summary>
	public class GaussianInitialiser : BaseInitialiser
	{
		/// <summary>
		/// The mean of this Gaussian initialiser.
		/// </summary>
		public double Mean { get; set; }

		/// <summary>
		/// The standard deviation of this Gaussian initialiser.
		/// </summary>
		public double StandardDeviation { get; set; }

		/// <summary>
		/// Create a Gaussian initialiser with a certain standard deviation and optional mean.
		/// </summary>
		/// <param name="standardDeviation">The standard deviation.</param>
		/// <param name="mean">The mean.</param>
		public GaussianInitialiser(double standardDeviation, double mean = 0.0)
		{
			this.Mean = mean;
			this.StandardDeviation = standardDeviation;
		}

		/// <summary>
		/// Create a Gaussian initialiser with a certain standard deviation and optional mean.
		/// </summary>
		/// <param name="standardDeviation">The standard deviation.</param>
		/// <param name="mean">The mean.</param>
		/// <returns>A Gaussian initialiser with the given standard deviation and mean.</returns>
		public GaussianInitialiser Gaussian(double standardDeviation, double mean = 0.0)
		{
			return new GaussianInitialiser(standardDeviation, mean);
		}

		public override object GetValue(long[] indices, long[] shape, Random random)
		{
			// box-muller transform for fast Gaussian values
			// see http://stackoverflow.com/questions/218060/random-gaussian-variables
			double u1 = random.NextDouble(); 
			double u2 = random.NextDouble();
			double randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Sin(2.0 * System.Math.PI * u2); 

			return Mean + StandardDeviation * randStdNormal;
		}
	}
}
