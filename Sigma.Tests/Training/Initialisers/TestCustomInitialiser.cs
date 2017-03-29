/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using NUnit.Framework;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
using Sigma.Core.Training.Initialisers;

namespace Sigma.Tests.Training.Initialisers
{
	public class TestCustomInitialiser
	{
		[TestCase]
		public void TestCustomInitialiserCreate()
		{
			Assert.Throws<ArgumentNullException>(() => new CustomInitialiser(null));
		}

		[TestCase]
		public void TestCustomInitialiserInitialise()
		{
			CustomInitialiser initialiser = new CustomInitialiser((i, s, r) => 5);

			ADNDArray<float> array = new ADNDArray<float>(2, 3);

			initialiser.Initialise(array, new CpuFloat32Handler(), new Random());

			foreach (var value in array.GetDataAs<float>())
			{
				Assert.AreEqual(5, value);
			}
		}
	}
}
