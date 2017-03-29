/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Initialisers
{
	[Serializable]
	public class XavierInitialiser : BaseInitialiser
	{
		/// <summary>
		/// Create a Xavier style initialiser with a certain mean and scale.
		/// The standard deviation is calculated as scale / ndarray.Length. 
		/// </summary>
		/// <param name="scale"></param>
		/// <param name="mean">The mean.</param>
		public XavierInitialiser(double scale = 1.0, double mean = 0.0)
		{
			Registry.Set("mean", mean, typeof(double));
			Registry.Set("scale", scale, typeof(double));
		}

		/// <summary>
		/// Get the value to set for certain indices shape and a helper randomiser. 
		/// </summary>
		/// <param name="indices">The indices.</param>
		/// <param name="shape">The shape.</param>
		/// <param name="random">The randomiser.</param>
		/// <returns>The value to set at the given indices.</returns>
		public override object GetValue(long[] indices, long[] shape, Random random)
		{
			// box-muller transform for fast Gaussian values
			// see http://stackoverflow.com/questions/218060/random-gaussian-variables
			double u1 = random.NextDouble();
			double u2 = random.NextDouble();
			double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
			double standardDeviation = Registry.Get<double>("scale") / ArrayUtils.Product(shape);

			return Registry.Get<double>("mean") +  standardDeviation * randStdNormal;
		}
	}
}
