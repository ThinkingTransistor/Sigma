/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Data.Extractors
{
	/// <summary>
	/// The base class for extractors which implements some basic methods (which are probably be the same for most extractors).
	/// </summary>
	public abstract class BaseExtractor : IRecordExtractor
	{
		public IRecordReader Reader
		{
			get; set;
		}

		public IRecordExtractor ParentExtractor
		{
			get; set;
		}

		public string[] SectionNames { get; set; }

		public virtual Dictionary<string, INDArray> ExtractDirect(int numberOfRecords, IComputationHandler handler)
		{
			if (Reader == null)
			{
				throw new InvalidOperationException("Cannot extract from record extractor before attaching a reader (reader was null).");
			}

			if (handler == null)
			{
				throw new ArgumentNullException("Computation handler cannot be null.");
			}

			if (numberOfRecords <= 0)
			{
				throw new ArgumentException($"Number of records to read must be > 0 but was {numberOfRecords}.");
			}

			return ExtractDirectFrom(Reader.Read(numberOfRecords), numberOfRecords, handler);
		}

		public abstract Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler);

		public virtual Dictionary<string, INDArray> ExtractHierarchicalFrom(object readData, int numberOfRecords, IComputationHandler handler)
		{
			if (ParentExtractor != null)
			{
				readData = ParentExtractor.ExtractHierarchicalFrom(readData, numberOfRecords, handler);
			}

			return ExtractDirectFrom(readData, numberOfRecords, handler);
		}

		public virtual void Prepare()
		{
			if (Reader == null)
			{
				throw new InvalidOperationException("Cannot prepare record extractor before attaching a reader (reader was null).");
			}

			Reader.Prepare();
		}

		public IRecordPreprocessor Preprocess(params IRecordPreprocessor[] preprocessors)
		{
			if (preprocessors.Length == 0)
			{
				throw new ArgumentException("Cannot add an empty array of preprocessors to this extractor.");
			}

			IRecordPreprocessor firstPreprocessor = preprocessors[0];

			firstPreprocessor.Reader = this.Reader;
			firstPreprocessor.ParentExtractor = this;

			//add this extractors section names to the other extractor
			firstPreprocessor.SectionNames = MergeSectionNames(firstPreprocessor);

			if (preprocessors.Length > 1)
			{
				return firstPreprocessor.Preprocess(preprocessors.SubArray(1, preprocessors.Length - 1));
			}

			return firstPreprocessor;
		}

		public IRecordExtractor Extractor(params IRecordExtractor[] extractors)
		{
			IRecordExtractor firstExtractor = extractors[0];

			firstExtractor.Reader = this.Reader;
			firstExtractor.ParentExtractor = this;

			firstExtractor.SectionNames = MergeSectionNames(firstExtractor);

			if (extractors.Length > 1)
			{
				return firstExtractor.Extractor(extractors.SubArray(1, extractors.Length - 1));
			}

			return firstExtractor;
		}

		private string[] MergeSectionNames(IRecordExtractor otherExtractor)
		{
			ISet<string> allSectionNames = new HashSet<string>();

			foreach (string section in this.SectionNames)
			{
				allSectionNames.Add(section);
			}

			if (otherExtractor.SectionNames != null)
			{
				foreach (string section in otherExtractor.SectionNames)
				{
					allSectionNames.Add(section);
				}
			}

			return allSectionNames.ToArray();
		}

		public abstract void Dispose();
	}
}
