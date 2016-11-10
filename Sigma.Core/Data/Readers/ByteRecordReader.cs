/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;
using log4net;
using Sigma.Core.Math;
using System.IO;

namespace Sigma.Core.Data.Readers
{
	public class ByteRecordReader : IRecordReader
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public IDataSetSource Source { get; private set; }

		private int headerBytes;
		private int recordSizeBytes;
		private bool prepared;
		private bool skippedHeaderBytes;

		public ByteRecordReader(IDataSetSource source, int headerBytes, int recordSizeBytes)
		{
			if (source == null)
			{
				throw new ArgumentNullException("Source cannot be null.");
			}

			if (headerBytes < 0)
			{
				throw new ArgumentException($"Header bytes must be >= 0, but header bytes was {headerBytes}.");
			}

			if (recordSizeBytes < 0)
			{
				throw new ArgumentException($"Record size bytes must be >= 0, but record size bytes was {recordSizeBytes}.");
			}

			this.Source = source;
			this.headerBytes = headerBytes;
			this.recordSizeBytes = recordSizeBytes;
		}

		public void Prepare()
		{
			if (prepared)
			{
				return;
			}

			Source.Prepare();

			prepared = true;
		}

		public object Read(int numberOfRecords)
		{
			if (!prepared)
			{
				throw new InvalidOperationException("Cannot read from source before preparing this reader (missing Prepare() call?).");
			}

			Stream stream = Source.Retrieve();

			if (!skippedHeaderBytes)
			{
				int read = stream.Read(new byte[headerBytes], 0, headerBytes);

				skippedHeaderBytes = true;

				if (read != headerBytes)
				{
					return null;
				}
			}

			List<byte[]> records = new List<byte[]>();

			int numberOfRecordsRead = 0;
			while (numberOfRecordsRead < numberOfRecords)
			{
				byte[] buffer = new byte[recordSizeBytes];

				int readBytes = stream.Read(buffer, 0, recordSizeBytes);

				if (readBytes != recordSizeBytes)
				{
					break;
				}

				records.Add(buffer);

				numberOfRecordsRead++;
			}

			return records.ToArray();
		}

		public IRecordExtractor Extractor(IRecordExtractor extractor)
		{
			extractor.Reader = this;

			return extractor;
		}

		public void Dispose()
		{
		}
	}
}
