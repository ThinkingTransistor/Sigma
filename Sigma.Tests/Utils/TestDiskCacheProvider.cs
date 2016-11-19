/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;
using System;
using System.IO;

namespace Sigma.Tests.Utils
{
	public class TestDiskCacheProvider
	{
		[TestCase]
		public void TestDiskCacheProviderCreate()
		{
			Assert.Throws<ArgumentNullException>(() => new DiskCacheProvider(null));

			DiskCacheProvider provider = new DiskCacheProvider(Path.GetTempPath());

			Assert.AreEqual(Path.GetTempPath(), provider.RootDirectory);
		}

		[TestCase]
		public void TestDiskCacheProviderStore()
		{
			DiskCacheProvider provider = new DiskCacheProvider(Path.GetTempPath() + nameof(TestDiskCacheProviderStore));

			provider.Store("test", "hellofriend");
			provider.Store("tost", "hallofreund");

			Assert.IsTrue(provider.IsCached("test"));
			Assert.IsTrue(provider.IsCached("tost"));
		}

		[TestCase]
		public void TestDiskCacheProviderLoad()
		{
			DiskCacheProvider provider = new DiskCacheProvider(Path.GetTempPath() + nameof(TestDiskCacheProviderLoad));

			provider.Store("test", "hellofriend");
			provider.Store("tost", "hallofreund");

			Assert.AreEqual("hellofriend", provider.Load<string>("test"));
			Assert.AreEqual("hallofreund", provider.Load<string>("tost"));
		}
	}
}
