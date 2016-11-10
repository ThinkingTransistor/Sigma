/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using System;
using System.IO;

namespace Sigma.Tests.Data.Readers
{
	public class TestCSVRecordReader
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
		public void TestCSVRecordReaderCreate()
		{
			string filename = ".unittestscsvrecorreader" + nameof(TestCSVRecordReaderCreate);
			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			CSVRecordReader reader = new CSVRecordReader(source);

			Assert.AreSame(source, reader.Source);
			Assert.Throws<ArgumentNullException>(() => new CSVRecordReader(null));

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCSVRecordReaderRead()
		{
			string filename = ".unittestscsvrecorreader" + nameof(TestCSVRecordReaderRead);
			CreateCSVTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			CSVRecordReader reader = new CSVRecordReader(source);

			Assert.Throws<InvalidOperationException>(() => reader.Read(3));

			reader.Prepare();

			string[][] lineparts = (string[][]) reader.Read(2);

			Assert.AreEqual(2, lineparts.Length);
			Assert.AreEqual(5, lineparts[0].Length);
			Assert.AreEqual(new string[] { "5.1", "3.5", "1.4", "0.2", "Iris-setosa" }, lineparts[0]);

			lineparts = (string[][]) reader.Read(3);

			Assert.AreEqual(1, lineparts.Length);
			Assert.AreEqual(5, lineparts[0].Length);

			reader.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCSVRecordReaderExtract()
		{
			CSVRecordReader reader = new CSVRecordReader(new FileSource("."));

			Assert.Throws<ArgumentException>(() => reader.Extractor());
			Assert.Throws<ArgumentException>(() => reader.Extractor("name", "name"));
			Assert.Throws<ArgumentException>(() => reader.Extractor(1));

			Assert.AreEqual(new[] { 0, 1, 2, 3, 6 }, reader.Extractor("inputs", new int[] { 0, 3 }, 6).NamedColumnIndexMapping["inputs"]);
		}
	}
}