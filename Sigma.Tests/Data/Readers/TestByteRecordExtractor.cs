using NUnit.Framework;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data.Readers
{
	public class TestByteRecordExtractor
	{
		private static void CreateCSVTempFile(string name)
		{
			File.Create(Path.GetTempPath() + name).Dispose();

			//no idea how this part could be done better without making a mess
			File.WriteAllBytes(Path.GetTempPath() + name, new byte[] { 0, 0, 5, 3, 4 });
		}

		private static void DeleteTempFile(string name)
		{
			File.Delete(Path.GetTempPath() + name);
		}

		[TestCase]
		public void TestByteRecordReaderCreate()
		{
			string filename = ".unittestfile" + nameof(TestByteRecordReaderCreate);

			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			Assert.Throws<ArgumentNullException>(() => new ByteRecordReader(null, 1, 1));
			Assert.Throws<ArgumentException>(() => new ByteRecordReader(source, -1, 1));
			Assert.Throws<ArgumentException>(() => new ByteRecordReader(source, 1, 0));

			Assert.AreSame(source, new ByteRecordReader(source, 0, 1).Source);

			source.Dispose();

			DeleteTempFile(filename);
		}
	}
}
