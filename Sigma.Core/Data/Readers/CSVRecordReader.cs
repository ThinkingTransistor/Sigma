/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A CSV record reader which reads comma separated values as string lines from a source.
	/// </summary>
	public class CSVRecordReader : IRecordReader
	{
		private const int NUMBER_COLUMNS_NOT_SET = -1;

		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private char separator;
		private bool skipFirstLine;
		private bool skippedFirstLine;
		private StreamReader reader;
		private int numberColumns = NUMBER_COLUMNS_NOT_SET;

		public IDataSetSource Source
		{
			get; private set;
		}

		/// <summary>
		/// Create a CSV record reader of a certain data set source and separator.
		/// </summary>
		/// <param name="source">The data set source.</param>
		/// <param name="separator">The separator to use in this CSV reader.</param>
		/// <param name="skipFirstLine">Indicate if the first line should be skipped.</param>
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

		public CSVRecordExtractor Extractor(params object[] parameters)
		{
			return Extractor(columnMappings: CSVRecordExtractor.ParseExtractorParameters(parameters));
		}

		public CSVRecordExtractor Extractor(Dictionary<string, IList<int>> columnMappings)
		{
			return (CSVRecordExtractor) Extractor(new CSVRecordExtractor(columnMappings));
		}

		public CSVRecordExtractor Extractor(Dictionary<string, int[][]> columnMappings)
		{
			return (CSVRecordExtractor) Extractor(new CSVRecordExtractor(columnMappings));
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

		public object Read(int numberOfRecords)
		{
			if (this.reader == null)
			{
				throw new InvalidOperationException("Cannot read from source before preparing this reader (missing Prepare() call?).");
			}

			if (numberOfRecords <= 0)
			{
				throw new ArgumentException($"Number of records to read must be > 0 but was {numberOfRecords}.");
			}

			logger.Info($"Reading requested {numberOfRecords} records from source {Source}...");

			List<string[]> records = new List<string[]>();
			int numberRecordsRead = 0;

			if (skipFirstLine && !skippedFirstLine)
			{
				reader.ReadLine();

				skippedFirstLine = true;
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

				//set number columns to the amount we find in the first column
				if (numberColumns == NUMBER_COLUMNS_NOT_SET)
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

			if (numberRecordsRead == 0)
			{
				logger.Info($"No more records could be read (requested: {numberOfRecords} records), end of stream most likely reached.");

				return null;
			}

			logger.Info($"Done reading records, read a total of {numberRecordsRead} records (requested: {numberOfRecords} records).");

			return records.ToArray<string[]>();
		}

		public void Dispose()
		{
			this.reader?.Dispose();
		}
	}
}
