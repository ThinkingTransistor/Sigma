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
	public class CSVRecordReader : IRecordReader
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private char separator;
		private bool skipFirstLine;
		private StreamReader reader;

		public IDataSetSource Source
		{
			get; private set;
		}

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
			if (parameters.Length == 0)
			{
				throw new ArgumentException("Extractor parameters cannot be empty.");
			}

			Dictionary<string, IList<int>> columnMappings = new Dictionary<string, IList<int>>();

			string currentNamedSection = null;
			IList<int> currentColumnMapping = null;
			int paramIndex = 0;

			foreach (object param in parameters)
			{
				if (param is string)
				{
					currentNamedSection = (string) param;

					if (columnMappings.ContainsKey(currentNamedSection))
					{
						throw new ArgumentException($"Named sections can only be used once, but section {currentNamedSection} (argument {paramIndex}) was already used.");
					}

					currentColumnMapping = new List<int>();
					columnMappings.Add(currentNamedSection, currentColumnMapping);
				}
				else if (param is int || param is int[])
				{
					if (currentNamedSection == null)
					{
						throw new ArgumentException("Cannot assign parameters without naming a section.");
					}

					if (param is int)
					{
						currentColumnMapping.Add((int) param);
					}
					else
					{
						int[] range = (int[]) param;

						if (range.Length != 2)
						{
							throw new ArgumentException($"Column ranges can only be pairs (size 2), but array length of argument {paramIndex} was {range.Length}.");
						}

						int[] flatRange = ArrayUtils.Range(range[0], range[1]);

						for (int i = 0; i < flatRange.Length; i++)
						{
							currentColumnMapping.Add(flatRange[i]);
						}
					}
				}
				else
				{
					throw new ArgumentException("All parameters must be either of type string, int or int[]");
				}

				paramIndex++;
			}

			return Extractor(columnMappings: columnMappings);
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

		public T Read<T>(int numberOfRecords)
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

			logger.Info($"Done reading records, read a total of {numberRecordsRead} records (requested: {numberOfRecords} records).");

			return (T) (object) (records.ToArray<string[]>());
		}

		public void Dispose()
		{
			this.reader?.Dispose();
		}
	}
}
