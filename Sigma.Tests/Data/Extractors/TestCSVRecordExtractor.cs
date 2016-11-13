/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends;
using System;
using System.Collections.Generic;
using System.IO;

namespace Sigma.Tests.Data.Preprocessors
{
	public class TestCSVRecordExtractor
	{
		private static void CreateCSVTempFile(string name)
		{
			File.Create(Path.GetTempPath() + name).Dispose();
			File.WriteAllLines(Path.GetTempPath() + name, new string[] { "5.1,3.5,1.4,0.2,Iris-setosa", "4.9,3.0,1.4,0.2,Iris-setosa", "4.7,3.2,1.3,0.2,Iris-setosa" });
		}

		private static void DeleteTempFile(string name)
		{
			File.Delete(Path.GetTempPath() + name);
		}

		[TestCase]
		public void TestCSVRecordExtractorCreate()
		{
			string filename = ".unittestfile" + nameof(TestCSVRecordExtractorCreate);

			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CSVRecordExtractor extractor = (CSVRecordExtractor) new CSVRecordReader(source).Extractor(new CSVRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new int[][] { new int[] { 0 } } }));

			Assert.AreSame(source, extractor.Reader.Source);
			Assert.AreEqual(new string[] { "inputs" }, extractor.SectionNames);

			source.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCSVRecordExtractorExtract()
		{
			string filename = ".unittestfile" + nameof(TestCSVRecordExtractorCreate);

			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CSVRecordExtractor extractor = (CSVRecordExtractor) new CSVRecordReader(source).Extractor(new CSVRecordExtractor(new Dictionary<string, IList<int>>() { ["inputs"] = new[] { 0, 1, 2 } }));

			Assert.Throws<InvalidOperationException>(() => extractor.ExtractDirect(3, new CPUFloat32Handler()));

			extractor.Prepare();

			Assert.Throws<ArgumentNullException>(() => extractor.ExtractDirect(3, null));
			Assert.Throws<ArgumentException>(() => extractor.ExtractDirect(-1, new CPUFloat32Handler()));

			Assert.AreEqual(5, extractor.ExtractDirect(3, new CPUFloat32Handler())["inputs"].GetValue<int>(0, 0, 0));

			source.Dispose();

			DeleteTempFile(filename);
		}
	}
}
