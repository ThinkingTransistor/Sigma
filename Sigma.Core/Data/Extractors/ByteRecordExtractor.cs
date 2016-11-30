/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.MathAbstract.Backends.NativeCpu;

namespace Sigma.Core.Data.Extractors
{
	/// <summary>
	/// A byte record extractor, which extracts named ranges to ndarrays byte-wise from a byte record reader.
	/// </summary>
	public class ByteRecordExtractor : BaseExtractor
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly Dictionary<string, long[][]> _indexMappings;

		/// <summary>
		/// Create a byte record extractor with a certain named index mapping.
		/// </summary>
		/// <param name="indexMappings">The named index mappings.</param>
		public ByteRecordExtractor(Dictionary<string, long[][]> indexMappings)
		{
			if (indexMappings == null)
			{
				throw new ArgumentNullException(nameof(indexMappings));
			}

			foreach (string name in indexMappings.Keys)
			{
				if (indexMappings[name].Length % 2 != 0)
				{
					throw new ArgumentException($"All index mapping arrays have to be a multiple of 2 (start and end indices of each range), but index mapping for name {name} had {indexMappings[name].Length}.");
				}
			}

			_indexMappings = indexMappings;
			SectionNames = indexMappings.Keys.ToArray();
		}

		public override Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler)
		{
			// read data being null means no more data could be read so we will just pass that along
			if (readData == null)
			{
				return null;
			}

			byte[][] rawRecords = (byte[][]) readData;

			int numberOfRecordsToExtract = Math.Min(rawRecords.Length, numberOfRecords);

			_logger.Info($"Extracting {numberOfRecordsToExtract} records from reader {Reader} (requested: {numberOfRecords})...");

			Dictionary<string, INDArray> namedArrays = new Dictionary<string, INDArray>();

			foreach (string name in _indexMappings.Keys)
			{
				long[][] mappings = _indexMappings[name];
				long[][] perMappingShape = new long[mappings.Length / 2][];
				long[] perMappingLength = new long[mappings.Length / 2];
				long[] featureShape = new long[mappings[0].Length];

				for (int i = 0; i < mappings.Length; i += 2)
				{
					int halfIndex = i / 2;
					perMappingShape[halfIndex] = new long[mappings[0].Length];

					for (int y = 0; y < featureShape.Length; y++)
					{
						perMappingShape[halfIndex][y] = mappings[i + 1][y] - mappings[i][y];
						featureShape[y] += perMappingShape[halfIndex][y];
					}

					perMappingLength[i / 2] = ArrayUtils.Product(perMappingShape[halfIndex]);
				}

				long[] shape = new long[featureShape.Length + 2];

				shape[0] = numberOfRecordsToExtract;
				shape[1] = 1;

				Array.Copy(featureShape, 0, shape, 2, featureShape.Length);

				INDArray array = handler.NDArray(shape);

				long[] globalBufferIndices = new long[shape.Length];

				for (int r = 0; r < numberOfRecordsToExtract; r++)
				{
					byte[] record = rawRecords[r];

					globalBufferIndices[0] = r; //BatchTimeFeatures indexing
					globalBufferIndices[1] = 0;

					for (int i = 0; i < mappings.Length; i += 2)
					{
						long[] beginShape = mappings[i];
						long[] localShape = perMappingShape[i / 2];
						long[] localStrides = NDArray<byte>.GetStrides(localShape);
						long[] localBufferIndices = new long[mappings[i].Length];
						long length = perMappingLength[i / 2];
						long beginFlatIndex = ArrayUtils.Product(beginShape);

						for (int y = 0; y < length; y++)
						{
							localBufferIndices = NDArray<byte>.GetIndices(y, localShape, localStrides, localBufferIndices);
							localBufferIndices = ArrayUtils.Add(beginShape, localBufferIndices, localBufferIndices);

							Array.Copy(localBufferIndices, 0, globalBufferIndices, 2, localBufferIndices.Length);

							array.SetValue(record[beginFlatIndex + y], globalBufferIndices);
						}
					}
				}

				namedArrays.Add(name, array);
			}

			_logger.Info($"Done extracting {numberOfRecordsToExtract} records from reader {Reader} (requested: {numberOfRecords}).");

			return namedArrays;
		}

		public override void Dispose()
		{
		}

		public static Dictionary<string, long[][]> ParseExtractorParameters(params object[] parameters)
		{
			if (parameters.Length == 0)
			{
				throw new ArgumentException("Extractor parameters cannot be empty.");
			}

			Dictionary<string, long[][]> indexMappings = new Dictionary<string, long[][]>();

			string currentNamedSection = null;
			IList<long[]> currentParams = null;
			int paramIndex = 0;

			foreach (object param in parameters)
			{
				string sectionName = param as string;
				if (sectionName != null)
				{
					string previousSection = currentNamedSection;
					currentNamedSection = sectionName;

					if (indexMappings.ContainsKey(currentNamedSection))
					{
						throw new ArgumentException($"Named sections can only be used once, but section {currentNamedSection} (argument {paramIndex}) was already used.");
					}

					if (previousSection != null)
					{
						AddNamedIndexParameters(indexMappings, previousSection, currentParams, paramIndex);
					}

					currentParams = new List<long[]>();
				}
				else if (param is long[])
				{
					if (currentNamedSection == null)
					{
						throw new ArgumentException("Cannot assign parameters without naming a section.");
					}

					long[] indexRange = (long[]) param;

					for (int i = 0; i < indexRange.Length; i++)
					{
						if (indexRange[i] < 0)
						{
							throw new ArgumentException($"All parameters in index range have to be >= 0, but element at index {i} of parameter {paramIndex} was {indexRange[i]}.");
						}
					}

					currentParams.Add(indexRange);
				}
				else
				{
					throw new ArgumentException("All parameters must be either of type string or long[].");
				}

				paramIndex++;
			}

			if (currentNamedSection != null && !indexMappings.ContainsKey(currentNamedSection))
			{
				AddNamedIndexParameters(indexMappings, currentNamedSection, currentParams, paramIndex);
			}

			return indexMappings;
		}

		private static void AddNamedIndexParameters(Dictionary<string, long[][]> indexMappings, string currentNamedSection, IList<long[]> currentParams, int paramIndex)
		{
			if (currentParams.Count < 1)
			{
				throw new ArgumentException($"There must be >= 1 index regions, but in section {currentNamedSection} at parameter index {paramIndex} there were {currentParams.Count}.");
			}

			indexMappings.Add(currentNamedSection, currentParams.ToArray());
		}
	}
}
