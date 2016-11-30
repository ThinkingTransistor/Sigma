/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends;
using System;
using System.IO;
using Sigma.Core.Handlers.Backends.NativeCpu;

namespace Sigma.Tests.Data.Datasets
{
	public class TestDatasetRecordwiseSlice
	{
		private static void RedirectGlobalsToTempPath()
		{
			SigmaEnvironment.Globals["workspacePath"] = Path.GetTempPath();
			SigmaEnvironment.Globals["cache"] = Path.GetTempPath();
			SigmaEnvironment.Globals["datasets"] = Path.GetTempPath();
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
		public void TestDatasetRecordwiseSliceCreate()
		{
			RedirectGlobalsToTempPath();

			string filename = nameof(TestDatasetRecordwiseSliceCreate) + "test.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 1, 2, "targets", 3);
			Dataset dataset = new Dataset("name", Dataset.BlockSizeAuto, extractor);

			Assert.Throws<ArgumentNullException>(() => new DatasetRecordwiseSlice(null, 0.0, 1.0));
			Assert.Throws<ArgumentException>(() => new DatasetRecordwiseSlice(dataset, -0.2, 1.0));
			Assert.Throws<ArgumentException>(() => new DatasetRecordwiseSlice(dataset, 0.7, -1.0));
			Assert.Throws<ArgumentException>(() => new DatasetRecordwiseSlice(dataset, 0.7, 0.6));

			DatasetRecordwiseSlice slice = new DatasetRecordwiseSlice(dataset, 0.1, 0.6);

			Assert.AreSame(dataset, slice.UnderlyingDataset);
			Assert.AreEqual(0.1, slice.ShareOffset);
			Assert.AreEqual(0.6, slice.Share);

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestDatsetRecordwiseSliceFetch()
		{
			RedirectGlobalsToTempPath();

			string filename = nameof(TestDatsetRecordwiseSliceFetch) + "test.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 0, "targets", 3);
			Dataset dataset = new Dataset("name", 3, extractor);
			DatasetRecordwiseSlice slice = new DatasetRecordwiseSlice(dataset, 0.1, 0.6);

			Assert.AreEqual(new float[] {5.1f, 4.9f}, slice.FetchBlock(0, new CpuFloat32Handler())["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2).TryGetValuesPackedArray());

			extractor.Reader?.Dispose();
			dataset.Dispose();

			DeleteTempFile(filename);
		}
	}
}
