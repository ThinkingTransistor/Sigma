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

namespace Sigma.Core.Data.Extractors
{
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

		public abstract string[] SectionNames { get; protected set; }

		public abstract Dictionary<string, INDArray> ExtractDirect(int numberOfRecords, IComputationHandler handler);

		public abstract Dictionary<string, INDArray> ExtractFrom(object readData, int numberOfRecords, IComputationHandler handler);

		public abstract void Prepare();

		public IRecordPreprocessor Preprocess(params IRecordPreprocessor[] preprocessors)
		{
			if (preprocessors.Length == 0)
			{
				throw new ArgumentException("Cannot add an empty array of preprocessors to this extractor.");
			}

			IRecordPreprocessor firstPreprocessor = preprocessors[0];

			firstPreprocessor.Reader = this.Reader;
			firstPreprocessor.ParentExtractor = this;

			if (preprocessors.Length > 1)
			{
				return firstPreprocessor.Preprocess(preprocessors.SubArray(1, preprocessors.Length - 1));
			}

			return firstPreprocessor;
		}

		public IRecordExtractor Extract(params IRecordExtractor[] extractors)
		{
			IRecordExtractor firstExtractor = extractors[0];

			firstExtractor.Reader = this.Reader;
			firstExtractor.ParentExtractor = this;

			if (extractors.Length > 1)
			{
				return firstExtractor.Extract(extractors.SubArray(1, extractors.Length - 1));
			}

			return firstExtractor;
		}

		public abstract void Dispose();
	}
}
