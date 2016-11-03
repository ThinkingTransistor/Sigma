/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data
{
	public class TestRegistryResolver
	{
		[TestCase]
		public void TestRegistryResolverModifyDirect()
		{
			Registry rootRegistry = new Registry(tags: "root");
			RegistryResolver resolver = new RegistryResolver(rootRegistry);

			Registry trainer1 = new Registry(rootRegistry, tags: "trainer");
			Registry trainer2 = new Registry(rootRegistry, tags: "trainer");

			rootRegistry["trainer1"] = trainer1;
			rootRegistry["trainer2"] = trainer2;

			//declare parameters in registry
			trainer1["accuracy"] = 0.0f;
			trainer2["accuracy"] = 0.0f;

			resolver.ResolveSet<float>("trainer1.accuracy", 1.0f, typeof(float));
			resolver.ResolveSet<float>("trainer2.accuracy", 2.0f, typeof(float));

			Assert.AreEqual(1.0f, resolver.ResolveGet<float>("trainer1.accuracy")[0]);
			Assert.AreEqual(2.0f, resolver.ResolveGet<float>("trainer2.accuracy")[0]);
		}

		[TestCase]
		public void TestRegistryResolverModifyComplex()
		{
			Registry rootRegistry = new Registry(tags: "root");
			RegistryResolver resolver = new RegistryResolver(rootRegistry);
			Registry trainer1 = new Registry(rootRegistry, tags: "trainer");
			Registry trainer2 = new Registry(rootRegistry, new string[] { "trainer" });
			Registry weirdtrainer = new Registry(rootRegistry, new string[] { "trainer" });
			Registry childRegistryToIgnore = new Registry(rootRegistry);

			Registry trainer1architecture = new Registry(trainer1, new string[] { "architecture" });
			Registry weirdarchitecture = new Registry(weirdtrainer, new string[] { "architecture" });

			rootRegistry["trainer1"] = trainer1;
			rootRegistry["trainer2"] = trainer2;
			rootRegistry["weirdtrainer"] = weirdtrainer;
			rootRegistry["childtoignore"] = childRegistryToIgnore;

			trainer1["architecture"] = trainer1architecture;
			weirdtrainer["architecture"] = weirdarchitecture;

			trainer1architecture["complexity"] = 2;
			weirdarchitecture["complexity"] = 3;

			trainer1["accuracy"] = 0.0f;
			trainer2["accuracy"] = 0.0f;

			resolver.ResolveSet<float>("trainer*.accuracy", 1.0f, typeof(float));

			resolver.ResolveSet<int>("*<trainer>.*<architecture>.complexity", 9, typeof(int));

			string[] resolved = null;

			Assert.AreEqual(new float[] { 1.0f, 1.0f }, resolver.ResolveGet<float>("trainer*.accuracy", ref resolved));
			Assert.AreEqual(new string[] { "trainer1.accuracy", "trainer2.accuracy" }, resolved);

			Assert.AreEqual(new int[] { 9, 9 }, resolver.ResolveGet<int>("*<trainer>.architecture.complexity"));

			Assert.AreEqual(new IRegistry[] { trainer1, trainer2, weirdtrainer }, resolver.ResolveGet<IRegistry>("*<trainer>", ref resolved));
			Assert.AreEqual(new string[] { "trainer1", "trainer2", "weirdtrainer" }, resolved);
		}

		[TestCase]
		public void TestRegistryResolverModifyHierarchy()
		{
			Registry rootRegistry = new Registry(tags: "root");
			RegistryResolver resolver = new RegistryResolver(rootRegistry);
			Registry trainer1 = new Registry(rootRegistry, tags: "trainer");
			Registry trainer2 = new Registry(rootRegistry, new string[] { "trainer" });

			Registry trainer1architecture = new Registry(trainer1, new string[] { "architecture" });
			Registry trainer2architecture = new Registry(trainer2, new string[] { "architecture" });

			rootRegistry["trainer1"] = trainer1;
			rootRegistry["trainer2"] = trainer2;

			trainer1["architecture"] = trainer1architecture;
			trainer2["architecture"] = trainer2architecture;

			trainer1architecture["complexity"] = 2;
			trainer2architecture["complexity"] = 3;

			Assert.AreEqual(new int[] { 2, 3 }, resolver.ResolveGet<int>("*.architecture.complexity"));

			Registry differentTrainer1Architecture = new Registry(trainer1, "architecture");

			trainer1["architecture"] = differentTrainer1Architecture;
			differentTrainer1Architecture["complexity"] = 5;

			resolver.ResolveSet<int>("*.architecture.complexity", 11, typeof(int));

			Assert.AreEqual(new int[] { 11, 11 }, resolver.ResolveGet<int>("*.architecture.complexity"));
		}
	}
}