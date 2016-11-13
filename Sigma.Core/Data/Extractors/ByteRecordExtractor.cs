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
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using System.ComponentModel;
using log4net;
using Sigma.Core.Utils;

namespace Sigma.Core.Data.Extractors
{
	/// <summary>
	/// A byte record extractor, which extracts named ranges to ndarrays byte-wise from a byte record reader.
	/// </summary>
	public class ByteRecordExtractor : BaseExtractor
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private Dictionary<string, long[][]> indexMappings;

		/// <summary>
		/// Create a byte record extractor with a certain named index mapping.
		/// </summary>
		/// <param name="indexMappings">The named index mappings.</param>
		public ByteRecordExtractor(Dictionary<string, long[][]> indexMappings)
		{
			if (indexMappings == null)
			{
				throw new ArgumentNullException("Index mappings cannot be null.");
			}

			foreach (string name in indexMappings.Keys)
			{
				if (indexMappings[name].Length % 2 != 0)
				{
					throw new ArgumentException($"All index mapping arrays have to be a multiple of 2 (start and end indices of each range), but index mapping for name {name} had {indexMappings[name].Length}.");
				}
			}

			this.indexMappings = indexMappings;
			this.SectionNames = indexMappings.Keys.ToArray();
		}

		public override Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler)
		{
			// read data being null means no more data could be read so we will just pass that along
			if (readData == null)
			{
				return null;
			}

			byte[][] rawRecords = (byte[][]) readData;

			int numberOfRecordsToExtract = System.Math.Min(rawRecords.Length, numberOfRecords);

			logger.Info($"Extracting {numberOfRecordsToExtract} records from reader {Reader} (requested: {numberOfRecords})...");

			Dictionary<string, INDArray> namedArrays = new Dictionary<string, INDArray>();

			foreach (string name in indexMappings.Keys)
			{
				long[][] mappings = indexMappings[name];
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

				INDArray array = handler.Create(shape);
				TypeConverter converter = TypeDescriptor.GetConverter(typeof(double));

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

							array.SetValue<byte>(record[beginFlatIndex + y], globalBufferIndices);
						}
					}
				}

				namedArrays.Add(name, array);
			}

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
				if (param is string)
				{
					string previousSection = currentNamedSection;
					currentNamedSection = (string) param;

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
