﻿/* 
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
using Sigma.Core.MathAbstract;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Sigma.Core.Handlers.Backends.NativeCpu;

namespace Sigma.Tests.Data.Datasets
{
	public class TestDataset
	{
		private static void RedirectGlobalsToTempPath()
		{
			SigmaEnvironment.Globals["workspacePath"] = Path.GetTempPath();
			SigmaEnvironment.Globals["cache"] = Path.GetTempPath() + "sigmacache";
			SigmaEnvironment.Globals["datasets"] = Path.GetTempPath() + "sigmadatasets";
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
		public void TestDatasetCreate()
		{
			RedirectGlobalsToTempPath();

			string filename = "test.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 1, 2, "targets", 3);
			CsvRecordExtractor clashingExtractor = new CsvRecordReader(new FileSource(filename)).Extractor("inputs", 1, 2);

			Assert.Throws<ArgumentNullException>(() => new Dataset(null, null));
			Assert.Throws<ArgumentNullException>(() => new Dataset("name", null));
			Assert.Throws<ArgumentNullException>(() => new Dataset("name", 10, null));
			Assert.Throws<ArgumentNullException>(() => new Dataset("name", 10, null, extractor));

			Assert.Throws<ArgumentException>(() => new Dataset("name", 10));
			Assert.Throws<ArgumentException>(() => new Dataset("name", -3, extractor));
			Assert.Throws<ArgumentException>(() => new Dataset("name"));
			Assert.Throws<ArgumentException>(() => new Dataset("name", extractor, clashingExtractor));

			Assert.AreEqual("name", new Dataset("name", extractor).Name);

			Assert.Greater(new Dataset("name", extractor).TargetBlockSizeRecords, 0);
			Assert.Greater(new Dataset("name", Dataset.BlockSizeAuto, extractor).TargetBlockSizeRecords, 0);

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestDatasetFetchBlockSequential()
		{
			RedirectGlobalsToTempPath();

			string filename = $"test{nameof(TestDatasetFetchBlockSequential)}.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename, Path.GetTempPath())).Extractor("inputs", 1, 2, "targets", 3);
			Dataset dataset = new Dataset(name: "name", blockSizeRecords: 1, recordExtractors: extractor);
			CpuFloat32Handler handler = new CpuFloat32Handler();

			Dictionary<string, INDArray> namedArrays = dataset.FetchBlock(0, handler, false);

			Assert.AreEqual(new[] { 3.5f, 1.4f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			//fetch the same thing twice to check for same block
			namedArrays = dataset.FetchBlock(0, handler, false);

			Assert.AreEqual(new[] { 3.5f, 1.4f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			//skipping second block (index 1)

			namedArrays = dataset.FetchBlock(2, handler, false);

			Assert.AreEqual(new[] { 3.2f, 1.3f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			namedArrays = dataset.FetchBlock(1, handler, false);

			Assert.AreEqual(new[] { 3.0f, 1.4f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			namedArrays = dataset.FetchBlock(3, handler, false);

			Assert.IsNull(namedArrays);

			dataset.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public async Task TestDatasetFetchAsync()
		{
			RedirectGlobalsToTempPath();

			string filename = $"test{nameof(TestDatasetFetchAsync)}.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename, Path.GetTempPath())).Extractor("inputs", 1, 2, "targets", 3);
			Dataset dataset = new Dataset(name: "name", blockSizeRecords: 1, recordExtractors: extractor);
			CpuFloat32Handler handler = new CpuFloat32Handler();

			var block0 = dataset.FetchBlockAsync(0, handler);
			var block2 = dataset.FetchBlockAsync(2, handler);
			var block1 = dataset.FetchBlockAsync(1, handler);

			//mock a free block request to freak out the dataset controller
			dataset.FreeBlock(1, handler);

			Dictionary<string, INDArray> namedArrays0 = await block0;
			Dictionary<string, INDArray> namedArrays1 = await block1;
			Dictionary<string, INDArray> namedArrays2 = await block2;

			Assert.IsNotNull(namedArrays1);
			Assert.AreEqual(new[] { 3.0f, 1.4f }, namedArrays1["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			Assert.IsNotNull(namedArrays2);
			Assert.AreEqual(new[] { 3.2f, 1.3f }, namedArrays2["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			Assert.IsNotNull(namedArrays0);
			Assert.AreEqual(new[] { 3.5f, 1.4f }, namedArrays0["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			dataset.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestDatasetFreeBlockSequential()
		{
			RedirectGlobalsToTempPath();

			string filename = $"test{nameof(TestDatasetFetchBlockSequential)}.dat";

			CreateCsvTempFile(filename);

			CsvRecordExtractor extractor = new CsvRecordReader(new FileSource(filename, Path.GetTempPath())).Extractor("inputs", 1, 2, "targets", 3);
			Dataset dataset = new Dataset(name: "name", blockSizeRecords: 1, recordExtractors: extractor);
			CpuFloat32Handler handler = new CpuFloat32Handler();

			dataset.FetchBlock(0, handler, false);
			dataset.FetchBlock(1, handler, false);
			dataset.FetchBlock(2, handler, false);

			Assert.AreEqual(3, dataset.ActiveBlockRegionCount);

			dataset.FreeBlock(1, handler);
			dataset.FreeBlock(2, handler);

			Assert.AreEqual(1, dataset.ActiveBlockRegionCount);

			var namedArrays = dataset.FetchBlock(0, handler, false);
			Assert.AreEqual(new[] { 3.5f, 1.4f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			namedArrays = dataset.FetchBlock(1, handler, false);
			Assert.AreEqual(new[] { 3.0f, 1.4f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			namedArrays = dataset.FetchBlock(2, handler, false);
			Assert.AreEqual(new[] { 3.2f, 1.3f }, namedArrays["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));

			dataset.Dispose();

			DeleteTempFile(filename);
		}
	}
}
