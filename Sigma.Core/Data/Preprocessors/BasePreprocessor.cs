using Sigma.Core.Data.Extractors;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Preprocessors
{
	public abstract class BasePreprocessor : BaseExtractor, IRecordPreprocessor
	{
		/// <summary>
		/// Similar to <see cref="IRecordExtractor.SectionNames"/>, this specifies the specific sections processed in this extractor or null if all given sections are processed.
		/// </summary>
		public IReadOnlyCollection<string> ProcessedSectionNames { get; protected set; }

		public virtual bool AffectsDataShape { get; }

		protected BasePreprocessor(params string[] processedSectionNames)
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
				if (ProcessedSectionNames == null || ProcessedSectionNames.Contains(sectionName))
				{
					processedNamedArrays.Add(sectionName, ProcessDirect(unprocessedNamedArrays[sectionName], handler));
				}
			}

			return processedNamedArrays;
		}

		protected abstract INDArray ProcessDirect(INDArray array, IComputationHandler handler);

		public override void Dispose()
		{
		}
	}
}
