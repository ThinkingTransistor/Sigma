/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Data.Sources;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data.Sources
{
	public class TestFileSource
	{
		[TestCase]
		public void TestFileSourceCheckNonExisting()
		{
			FileSource source = new FileSource("thisfiledoesntexistdoesit");

			Assert.IsFalse(source.Exists());
		}

		[TestCase]
		public void TestFileSourceAccessNonExisting()
		{
			FileSource source = new FileSource("thisfiledoesntexistdoesit");

			Assert.Throws<InvalidOperationException>(() => source.Prepare());

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());
		}

		[TestCase]
		public void TestFileSourceCheckExisting()
		{
			String fileFolder = Path.GetTempPath();
			String fileName = ".unittestfile1";

			File.Create(fileFolder + "/" + fileName).Dispose();

			FileSource source = new FileSource(".unittestfile1", fileFolder);

			Assert.IsTrue(source.Exists());

			File.Delete(fileFolder + "/" + fileName);
		}

		[TestCase]
		public void TestFileSourceAccessExisting()
		{
			String fileFolder = Path.GetTempPath();
			String fileName = ".unittestfile2";

			File.Create(fileFolder + "/" + fileName).Dispose();

			FileSource source = new FileSource(".unittestfile2", fileFolder);

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());

			source.Prepare();

			Stream stream = source.Retrieve();

			Assert.IsNotNull(stream);

			stream.Dispose();

			File.Delete(fileFolder + "/" + fileName);
		}
	}
}
