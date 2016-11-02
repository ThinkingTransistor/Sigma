using Sigma.Core.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using System.Diagnostics;
using Sigma.Core;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			Registry trainer1 = new Registry(sigma.Registry, new string[] { "trainer" });
			Registry trainer2 = new Registry(sigma.Registry, new string[] { "trainer" });
			Registry weirdtrainer = new Registry(sigma.Registry, new string[] { "trainer" });

			sigma.Registry["trainer1"] = trainer1;
			sigma.Registry["trainer2"] = trainer2;
			sigma.Registry["weirdtrainer"] = weirdtrainer;

			trainer1["accuracy"] = 0.0f;
			trainer2["accuracy"] = 0.0f;

			sigma.RegistryResolver.ResolveSet<float>("*.accuracy", 1.0f, typeof(float));

			float[] result = null;

			Console.WriteLine(sigma.RegistryResolver.ResolveRetrieve<float>("*.accuracy", ref result));

			Console.WriteLine(sigma.Registry);
			Console.ReadKey();
		}
	}
}
