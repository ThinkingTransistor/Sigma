/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using System;
using System.IO;

namespace Sigma.Tests.Data.Readers
{
	public class TestCsvRecordReader
	{
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
		public void TestCsvRecordReaderCreate()
		{
			string filename = ".unittestscsvrecorreader" + nameof(TestCsvRecordReaderCreate);
			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			CsvRecordReader reader = new CsvRecordReader(source);

			Assert.AreSame(source, reader.Source);
			Assert.Throws<ArgumentNullException>(() => new CsvRecordReader(null));

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCsvRecordReaderRead()
		{
			string filename = ".unittestscsvrecorreader" + nameof(TestCsvRecordReaderRead);
			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			CsvRecordReader reader = new CsvRecordReader(source);

			Assert.Throws<InvalidOperationException>(() => reader.Read(3));

			reader.Prepare();

			string[][] lineparts = (string[][]) reader.Read(2);

			Assert.AreEqual(2, lineparts.Length);
			Assert.AreEqual(5, lineparts[0].Length);
			Assert.AreEqual(new[] { "5.1", "3.5", "1.4", "0.2", "Iris-setosa" }, lineparts[0]);

			lineparts = (string[][]) reader.Read(3);

			Assert.AreEqual(1, lineparts.Length);
			Assert.AreEqual(5, lineparts[0].Length);

			reader.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCsvRecordReaderExtract()
		{
			CsvRecordReader reader = new CsvRecordReader(new FileSource("."));

			Assert.Throws<ArgumentException>(() => reader.Extractor());
			Assert.Throws<ArgumentException>(() => reader.Extractor("name", "name"));
			Assert.Throws<ArgumentException>(() => reader.Extractor(1));

			Assert.AreEqual(new[] { 0, 1, 2, 3, 6 }, reader.Extractor("inputs", new[] { 0, 3 }, 6).NamedColumnIndexMapping["inputs"]);
		}
	}
}