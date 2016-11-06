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
using System.IO;
using Sigma.Core.Utils;

namespace Sigma.Core.Data.Readers
{
	public class CSVRecordReader : IRecordReader
	{
		public IDataSetSource Source
		{
			get; private set;
		}

		private char separator;
		private bool skipFirstLine;
		private StreamReader reader;

		public CSVRecordReader(IDataSetSource source, char separator = ',', bool skipFirstLine = false)
		{
			if (source == null)
			{
				throw new ArgumentNullException("Source cannot be null.");
			}

			this.Source = source;
			this.separator = separator;
			this.skipFirstLine = skipFirstLine;
		}

		public IRecordExtractor Extractor(Dictionary<string, IList<int>> columnMappings)
		{
			return Extractor(new CSVRecordExtractor(columnMappings));
		}

		public IRecordExtractor Extractor(Dictionary<string, int[][]> columnMappings)
		{
			return Extractor(new CSVRecordExtractor(columnMappings));
		}

		public IRecordExtractor Extractor(IRecordExtractor extractor)
		{
			extractor.Reader = this;

			return extractor;
		}

		public void Prepare()
		{
			Source.Prepare();

			if (this.reader == null)
			{
				//we need to use the same reader for every read call because streamreader buffers when reading 
				// and we cannot assume that the underlying stream supports seeking
				this.reader = new StreamReader(Source.Retrieve());
			}
		}

		public T Read<T>(int numberOfRecords)
		{
			List<string[]> records = new List<string[]>();
			int numberRecordsRead = 0;
			int numberColumns = -1;

			if (skipFirstLine)
			{
				reader.ReadLine();
			}

			string line;
			while (numberRecordsRead < numberOfRecords)
			{
				line = this.reader.ReadLine();

				if (line == null)
				{
					break;
				}

				string[] lineParts = line.Split(separator);

				if (numberColumns == -1)
				{
					numberColumns = lineParts.Length;
				}

				//invalid line, lets look at the next one
				if (lineParts.Length != numberColumns)
				{
					continue;
				}

				records.Add(lineParts);

				numberRecordsRead++;
			}

			return (T) (object) (records.ToArray<string[]>());
		}
	}
}
