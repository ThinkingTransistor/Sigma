using NUnit.Framework;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data.Extractors
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
		public void TestByteRecordExtractorCreate()
		{
			string filename = ".unittestfile" + nameof(TestByteRecordExtractorCreate);

			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			Assert.Throws<ArgumentNullException>(() => new ByteRecordExtractor(null));
			Assert.Throws<ArgumentException>(() => new ByteRecordExtractor(new Dictionary<string, long[][]>() { ["test"] = new long[1][] { new long[] { 1, 2, 3 } } }));

			ByteRecordExtractor extractor = new ByteRecordExtractor(ByteRecordExtractor.ParseExtractorParameters("inputs", new[] { 0L }, new[] { 1L }));

			Assert.AreEqual(new string[] { "inputs" }, extractor.SectionNames);

			source.Dispose();

			DeleteTempFile(filename);
		}
	}
}
