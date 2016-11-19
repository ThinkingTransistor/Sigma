/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Sources;
using System;
using System.IO;

namespace Sigma.Tests.Data.Sources
{
	public class TestMultiSource
	{
		private static void CreateTempFile(string name)
		{
			File.Create(Path.GetTempPath() + "/" + name).Dispose();
			File.WriteAllLines(Path.GetTempPath() + name, new[] { "5.1,3.5,1.4,0.2,Iris-setosa", "4.9,3.0,1.4,0.2,Iris-setosa", "4.7,3.2,1.3,0.2,Iris-setosa" });
		}

		private static void DeleteTempFile(string name)
		{
			File.Delete(Path.GetTempPath() + "/" + name);
		}

		[TestCase]
		public void TestMultiSourceCreate()
		{
			Assert.Throws<ArgumentNullException>(() => new MultiSource(null));
			Assert.Throws<ArgumentNullException>(() => new MultiSource(null, null));
			Assert.Throws<ArgumentNullException>(() => new MultiSource(new FileSource("totallynotexisting"), null));

			Assert.Throws<ArgumentException>(() => new MultiSource());

			new MultiSource(new FileSource("totallynotexisting"), new FileSource("otherfile"));
		}

		[TestCase]
		public void TestMultiSourceAssignActive()
		{
			string filename = ".unittest" + nameof(TestMultiSourceAssignActive);
			CreateTempFile(filename);

			FileSource shouldBeActiveSource = new FileSource(filename, Path.GetTempPath());
			MultiSource source = new MultiSource(new FileSource("totallynotexisting"), shouldBeActiveSource);

			Assert.AreSame(shouldBeActiveSource, source.ActiveSource);

			source = new MultiSource(shouldBeActiveSource, new FileSource("totallynotexisting"));

			Assert.AreSame(shouldBeActiveSource, source.ActiveSource);
		}

		[TestCase]
		public void TestMultiSourcePrepareRetrieveDispose()
		{
			string filename = ".unittest" + nameof(TestMultiSourceAssignActive);
			CreateTempFile(filename);

			MultiSource source = new MultiSource(new FileSource("totallynotexisting"), new FileSource(filename, Path.GetTempPath()));

			source.Prepare();

			Stream stream = source.Retrieve();
			StreamReader reader = new StreamReader(stream);

			Assert.AreEqual("5.1,3.5,1.4,0.2,Iris-setosa", reader.ReadLine());

			reader.Dispose();
			source.Dispose();

			DeleteTempFile(filename);
		}
	}
}
