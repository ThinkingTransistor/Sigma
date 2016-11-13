/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Extractors;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// The base class for all preprocessors. Takes care of selective per section processing and simplifies implementation of new preprocessors. 
	/// </summary>
	public abstract class BasePreprocessor : BaseExtractor, IRecordPreprocessor
	{
		/// <summary>
		/// Similar to <see cref="IRecordExtractor.SectionNames"/>, this specifies the specific sections processed in this extractor or null if all given sections are processed.
		/// </summary>
		public IReadOnlyCollection<string> ProcessedSectionNames { get; protected set; }

		public abstract bool AffectsDataShape { get; }

		/// <summary>
		/// Create a base processor with an optional array of sections to process.
		/// If an array of section names is specified, only the sections with those names are processed. 
		/// If no such array is specified (null or empty), all sections are processed.
		/// </summary>
		/// <param name="processedSectionNames"></param>
		protected BasePreprocessor(string[] processedSectionNames = null)
		{
			if (processedSectionNames != null && processedSectionNames.Length == 0)
			{
				processedSectionNames = null;
			}

			this.ProcessedSectionNames = processedSectionNames;
		}

		public override Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler)
		{
			Dictionary<string, INDArray> unprocessedNamedArrays = (Dictionary<string, INDArray>) readData;
			Dictionary<string, INDArray> processedNamedArrays = new Dictionary<string, INDArray>();

			foreach (string sectionName in unprocessedNamedArrays.Keys)
			{
				INDArray processedArray = unprocessedNamedArrays[sectionName];

				if (processedArray.Shape[0] != numberOfRecords)
				{
					long[] beginIndices = processedArray.Shape.ToArray();
					long[] endIndices = processedArray.Shape.ToArray();

					beginIndices[0] = 0;
					endIndices[0] = numberOfRecords;

					for (int i = 1; i < processedArray.Rank; i++)
					{
						beginIndices[i] = 0;
						endIndices[i] = processedArray.Shape[i]; 
					}

					processedArray = processedArray.Slice(beginIndices, endIndices);
				}

				if (ProcessedSectionNames == null || ProcessedSectionNames.Contains(sectionName))
				{
					processedArray = ProcessDirect(processedArray, handler);
				}

				processedNamedArrays.Add(sectionName, processedArray);
			}

			return processedNamedArrays;
		}

		/// <summary>
		/// Process a certain ndarray with a certain computation handler.
		/// </summary>
		/// <param name="array">The ndarray to process.</param>
		/// <param name="handler">The computation handler to do the processing with.</param>
		/// <returns>An ndarray with the processed contents of the given array (can be the same or a new one).</returns>
		protected abstract INDArray ProcessDirect(INDArray array, IComputationHandler handler);

		public override void Dispose()
		{
			// there shouldn't be anything to dispose in a preprocessor
		}
	}
}
