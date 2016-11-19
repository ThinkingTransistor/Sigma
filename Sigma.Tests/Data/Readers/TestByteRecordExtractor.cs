using NUnit.Framework;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using System;
using System.IO;

namespace Sigma.Tests.Data.Readers
{
	public class TestByteRecordExtractor
	{
		private static void CreateCsvTempFile(string name)
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

			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			Assert.Throws<ArgumentNullException>(() => new ByteRecordReader(null, 1, 1));
			Assert.Throws<ArgumentException>(() => new ByteRecordReader(source, -1, 1));
			Assert.Throws<ArgumentException>(() => new ByteRecordReader(source, 1, 0));

			Assert.AreSame(source, new ByteRecordReader(source, 0, 1).Source);

			source.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestByteRecordReaderRead()
		{
			string filename = ".unittestfile" + nameof(TestByteRecordReaderCreate);

			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());

			ByteRecordReader reader = new ByteRecordReader(source, 2, 1);

			Assert.Throws<InvalidOperationException>(() => reader.Read(1));

			reader.Prepare();

			Assert.AreEqual(new[] { new byte[] { 5 }, new byte[] { 3 }, new byte[] { 4 } }, reader.Read(4));

			source.Dispose();
			reader.Dispose();

			DeleteTempFile(filename);
		}
	}
}
