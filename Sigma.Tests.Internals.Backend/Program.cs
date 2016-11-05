using System;
using Sigma.Core;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using Sigma.Core.Data.Sources;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			log4net.Config.XmlConfigurator.Configure();

			URLSource source = new URLSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");

			source.Prepare();

			Console.ReadKey();
		}
	}
}
