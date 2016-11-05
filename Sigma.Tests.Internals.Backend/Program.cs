using System;
using Sigma.Core;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using Sigma.Core.Data.Sources;
using System.IO;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			log4net.Config.XmlConfigurator.Configure();

			URLSource source = new URLSource("http://thisisnotavalidwebsiteisitnoitisnt.noitisnt/notexistingfile.dat");

			source.Prepare();

			Stream stream = source.Retrieve();

			

			Console.ReadKey();
		}
	}
}
