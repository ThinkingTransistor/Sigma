/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;

namespace Sigma.Tests.Data.Datasets
{
	public class TestDatasetBlockwiseSlice : BaseLocaleTest
	{
		private static void RedirectGlobalsToTempPath()
		{
			SigmaEnvironment.Globals["workspace_path"] = Path.GetTempPath();
			SigmaEnvironment.Globals["cache_path"] = Path.GetTempPath();
			SigmaEnvironment.Globals["datasets_path"] = Path.GetTempPath();
		}

		private static void CreateCsvTempFile(string name)
		{
			File.Create(Path.GetTempPath() + name).Dispose();
			File.WriteAllLines(Path.GetTempPath() + name, new[] { "5.1,3.5,1.4,0.2,Iris-setosa", "4.9,3.0,1.4,0.2,Iris-setosa", "4.7,3.2,1.3,0.2,Iris-setosa" });
		}

		private static void DeleteTempFile(string name)
		{
			File.Delete(Path.GetTempPath() + name);
		}

		[TestCase]
		public void TestDatasetBlockwiseSliceCreate()
		{
			RedirectGlobalsToTempPath();

			string filename = nameof(TestDatasetBlockwiseSliceCreate) + "test.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 1, 2, "targets", 3);
			ExtractedDataset dataset = new ExtractedDataset("name", ExtractedDataset.BlockSizeAuto, extractor);

			Assert.Throws<ArgumentNullException>(() => new DatasetBlockwiseSlice(null, 0, 0, 1));
			Assert.Throws<ArgumentException>(() => new DatasetBlockwiseSlice(dataset, 0, 0, 0));
			Assert.Throws<ArgumentException>(() => new DatasetBlockwiseSlice(dataset, -1, 0, 1));
			Assert.Throws<ArgumentException>(() => new DatasetBlockwiseSlice(dataset, 0, -1, 1));
			Assert.Throws<ArgumentException>(() => new DatasetBlockwiseSlice(dataset, 1, 0, 1));
			Assert.Throws<ArgumentException>(() => new DatasetBlockwiseSlice(dataset, 0, 2, 2));

			DatasetBlockwiseSlice slice = new DatasetBlockwiseSlice(dataset, 0, 1, 3);

			Assert.AreSame(dataset, slice.UnderlyingDataset);
			Assert.AreEqual(0, slice.SplitBeginIndex);
			Assert.AreEqual(1, slice.SplitEndIndex);
			Assert.AreEqual(2, slice.SplitSize);
			Assert.AreEqual(3, slice.SplitInterval);

			Assert.AreEqual(dataset.Name, slice.Name);
			Assert.AreEqual(dataset.TargetBlockSizeRecords, slice.TargetBlockSizeRecords);
			Assert.AreEqual(dataset.SectionNames, slice.SectionNames);

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestDatasetBlockwiseSliceFetch()
		{
			RedirectGlobalsToTempPath();

			string filename = nameof(TestDatasetBlockwiseSliceFetch) + "test.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 0, "targets", 3);
			ExtractedDataset dataset = new ExtractedDataset("name", 1, extractor);
			DatasetBlockwiseSlice slice = new DatasetBlockwiseSlice(dataset, 1, 2, 3);

			Assert.AreEqual(new float[] { 4.9f }, slice.FetchBlock(0, new CpuFloat32Handler())["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 1));

			extractor.Reader?.Dispose();
			dataset.Dispose();

			DeleteTempFile(filename);
		}
	}
}
