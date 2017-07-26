/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A character-level reader, which reads records in characterwise.
	/// </summary>
	[Serializable]
	public class CharacterRecordReader : IRecordReader
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly Encoding _encoding;
		private readonly int _recordLengthInCharacters;
		private bool _prepared;

		/// <summary>
		/// Create a character record reader with a specific record length.
		/// </summary>
		/// <param name="source">The underlying data source to use.</param>
		/// <param name="recordLengthInCharacters">The record length in characters.</param>
		public CharacterRecordReader(IDataSource source, int recordLengthInCharacters) : this(source, recordLengthInCharacters, Encoding.ASCII)
		{
		}

		/// <summary>
		/// Create a character record reader with a specific record length and encoding.
		/// </summary>
		/// <param name="source">The underlying data source to use.</param>
		/// <param name="recordLengthInCharacters">The record length in characters.</param>
		/// <param name="encoding">The encoding to use.</param>
		public CharacterRecordReader(IDataSource source, int recordLengthInCharacters, Encoding encoding)
		{
			_recordLengthInCharacters = recordLengthInCharacters;
			_encoding = encoding;
			Source = source;
		}

		/// <summary>
		/// The underlying data set source.
		/// </summary>
		public IDataSource Source { get; }

		/// <summary>
		/// Prepare this record reader and its underlying resources to be read.
		/// Note: This function may be called more than once (and subsequent calls should probably be ignored, depending on the implementation). 
		/// </summary>
		public void Prepare()
		{
			Source.Prepare();
			_prepared = true;
		}

		/// <summary>
		/// Reads a number of records in any format (therefore attached extractors must be compatible). 
		/// This method is mostly used internally by extractors, which have to be compatible with the used record readers (so they know which data format to expect). 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to read.</param>
		/// <returns>An object of the given type representing a collection of a given number of records.</returns>
		public object Read(int numberOfRecords)
		{
			if (!_prepared)
			{
				throw new InvalidOperationException("Cannot read from source before preparing this reader (missing Prepare() call?).");
			}

			Stream stream = Source.Retrieve();

			StreamReader reader = new StreamReader(stream, Encoding.ASCII);
			List<short[]> records = new List<short[]>(numberOfRecords);
			int index = 0, charactersToRead = numberOfRecords * _recordLengthInCharacters;

			while (!reader.EndOfStream && index < charactersToRead)
			{
				int recordIndex = index / _recordLengthInCharacters;

				if (records.Count <= recordIndex || records[recordIndex] == null)
				{
					records.Insert(recordIndex, new short[_recordLengthInCharacters]);
				}

				records[recordIndex][index % _recordLengthInCharacters] = (short) reader.Read();

				index++;
			}

			if (index % _recordLengthInCharacters != 0)
			{
				records.RemoveAt(records.Count - 1);
			}

			return records.ToArray();
		}

		/// <summary>
		/// Attach a certain record extractor to this record reader.
		/// </summary>
		/// <param name="extractor">The extractor to attach this reader to.</param>
		/// <returns>The given extractor (for convenience).</returns>
		public IRecordExtractor Extractor(IRecordExtractor extractor)
		{
			extractor.Reader = this;

			return extractor;
		}

		/// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
		public void Dispose()
		{
		}
	}
}
