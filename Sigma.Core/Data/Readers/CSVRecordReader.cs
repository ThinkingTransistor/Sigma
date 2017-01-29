/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A CSV record reader which reads comma separated values as string lines from a source.
	/// </summary>
	public class CsvRecordReader : IRecordReader
	{
		private const int NumberColumnsNotSet = -1;

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly char _separator;
		private readonly bool _skipFirstLine;
		private bool _skippedFirstLine;
		private StreamReader _reader;
		private int _numberColumns = NumberColumnsNotSet;

		public IDataSource Source
		{
			get; }

		/// <summary>
		/// Create a CSV record reader of a certain data set source and separator.
		/// </summary>
		/// <param name="source">The data set source.</param>
		/// <param name="separator">The separator to use in this CSV reader.</param>
		/// <param name="skipFirstLine">Indicate if the first line should be skipped.</param>
		public CsvRecordReader(IDataSource source, char separator = ',', bool skipFirstLine = false)
		{
			if (source == null)
			{
				throw new ArgumentNullException(nameof(source));
			}

			Source = source;
			_separator = separator;
			_skipFirstLine = skipFirstLine;
		}

		public CsvRecordExtractor Extractor(params object[] parameters)
		{
			return Extractor(columnMappings: CsvRecordExtractor.ParseExtractorParameters(parameters));
		}

		public CsvRecordExtractor Extractor(Dictionary<string, IList<int>> columnMappings)
		{
			return (CsvRecordExtractor) Extractor(new CsvRecordExtractor(columnMappings));
		}

		public CsvRecordExtractor Extractor(Dictionary<string, int[][]> columnMappings)
		{
			return (CsvRecordExtractor) Extractor(new CsvRecordExtractor(columnMappings));
		}

		public IRecordExtractor Extractor(IRecordExtractor extractor)
		{
			extractor.Reader = this;

			return extractor;
		}

		public void Prepare()
		{
			Source.Prepare();

			if (_reader == null)
			{
				//we need to use the same reader for every read call because streamreader buffers when reading 
				// and we cannot assume that the underlying stream supports seeking
				_reader = new StreamReader(Source.Retrieve());
			}
		}

		public object Read(int numberOfRecords)
		{
			if (_reader == null)
			{
				throw new InvalidOperationException("Cannot read from source before preparing this reader (missing Prepare() call?).");
			}

			if (numberOfRecords <= 0)
			{
				throw new ArgumentException($"Number of records to read must be > 0 but was {numberOfRecords}.");
			}

			_logger.Debug($"Reading requested {numberOfRecords} records from source {Source}...");

			List<string[]> records = new List<string[]>();
			int numberRecordsRead = 0;

			if (_skipFirstLine && !_skippedFirstLine)
			{
				_reader.ReadLine();

				_skippedFirstLine = true;
			}

			string line;
			while (numberRecordsRead < numberOfRecords)
			{
				line = _reader.ReadLine();

				if (line == null)
				{
					break;
				}

				string[] lineParts = line.Split(_separator);

				//set number columns to the amount we find in the first column
				if (_numberColumns == NumberColumnsNotSet)
				{
					_numberColumns = lineParts.Length;
				}

				//invalid line, lets look at the next one
				if (lineParts.Length != _numberColumns)
				{
					continue;
				}

				records.Add(lineParts);

				numberRecordsRead++;
			}

			if (numberRecordsRead == 0)
			{
				_logger.Info($"No more records could be read (requested: {numberOfRecords} records), end of stream most likely reached.");

				return null;
			}

			_logger.Debug($"Done reading records, read a total of {numberRecordsRead} records (requested: {numberOfRecords} records).");

			return records.ToArray<string[]>();
		}

		public void Dispose()
		{
			_reader?.Dispose();
		}
	}
}
