/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Initialisers;
using System;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu;

namespace Sigma.Tests.Training.Initialisers
{
	public class TestConstantValueInitialiser
	{
		[TestCase]
		public void TestConstantValueInitialiserInitialise()
		{
			ConstantValueInitialiser initialiser = new ConstantValueInitialiser(2.0);

			IComputationHandler handler = new CpuFloat32Handler();
		    INDArray array = handler.NDArray(2L, 1L, 2L, 2L);

            Random random = new Random();

			Assert.Throws<ArgumentNullException>(() => initialiser.Initialise((INDArray) null, handler, random));
			Assert.Throws<ArgumentNullException>(() => initialiser.Initialise((INumber) null, handler, random));
			Assert.Throws<ArgumentNullException>(() => initialiser.Initialise(array, null, random));
			Assert.Throws<ArgumentNullException>(() => initialiser.Initialise(array, handler, null));

			initialiser.Initialise(array, handler, new Random());

			Assert.AreEqual(new float[] { 2, 2, 2, 2, 2, 2, 2, 2 }, array.GetDataAs<float>().GetValuesArrayAs<float>(0, 8));
		}
	}
}
