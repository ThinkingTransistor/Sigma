/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;

namespace Sigma.Tests.Utils
{
	public class TestRegistry
	{
		[TestCase]
		public void TestRegistryReadWrite()
		{
			Registry registry = new Registry();

			registry["testkey1"] = "testvalue1";
			registry["testkey2"] = "testvalue2";

			Assert.AreEqual(registry["testkey1"], "testvalue1");
			Assert.AreEqual(registry["testkey2"], "testvalue2");

			Assert.Throws<System.ArgumentException>(() => registry.Add("testkey1", "anothertestvalue1"));
		}

		[TestCase]
		public void TestRegistryTypeChecking()
		{
			Registry registry = new Registry();

			registry.Set("checkedtestkey1", 0.0f, typeof(float));

			Assert.Throws<System.ArgumentException>(() => registry.Set("checkedtestkey1", "invalidtype"));

			registry.CheckTypes = false;

			registry.Set("checkedtestkey1", "invalidtype");
		}

		[TestCase]
		public void TestRegistryRemove()
		{
			Registry registry = new Registry();

			registry.Set("testkey1", 1);

			Assert.AreEqual(registry["testkey1"], 1);

			registry.Remove("testkey1");

			Assert.IsFalse(registry.ContainsKey("testkey1"));
		}

		[TestCase]
		public void TestRegistryGetAll()
		{
			Registry registry = new Registry();

			registry.Set("key1", "value1");
			registry.Set("key22", "value2");
			registry.Set("kay3", "value3");

			Assert.AreEqual(registry.GetAllValues<string>("key.*", typeof(string)), new string[]{ "value1", "value2" });
		}

		[TestCase]
		public void TestRegistryHierarchy()
		{
			Registry root = new Registry();
			Registry second = new Registry(root);
			Registry third = new Registry(second);
			Registry fourth = new Registry(third);

			Assert.AreSame(root, third.Root);
			Assert.AreSame(root, fourth.Root);

			Assert.AreSame(root, second.Parent);
			Assert.AreSame(second, third.Parent);
			Assert.AreSame(third, fourth.Parent);
		}
	}
}
