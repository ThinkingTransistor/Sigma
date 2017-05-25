/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Preprocessors.Adaptive;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of default-anythings for quick testing and prototyping.
	/// </summary>
	public static class Defaults
	{
		/// <summary>
		/// A collection of default datasets.
		/// </summary>
		public static class Datasets
		{
			#region Raw datasets

			/// <summary>
			/// Create a raw logical AND dataset.
			/// The samples are (0, 0 => 0), (0, 1 => 0), (1, 0 => 0), (1, 1 => 1).
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The AND dataset.</returns>
			public static IDataset And(string name = "and")
			{
				RawDataset dataset = new RawDataset(name);
				dataset.AddRecords("inputs", new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 1 });
				dataset.AddRecords("targets", new[] { 0 }, new[] { 0 }, new[] { 0 }, new[] { 1 });

				return dataset;
			}

			/// <summary>
			/// Create a raw logical XOR dataset.
			/// The samples are (0, 0 => 0), (0, 1 => 1), (1, 0 => 1), (1, 1 => 0).
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The XOR dataset.</returns>
			public static IDataset Xor(string name = "xor")
			{
				RawDataset dataset = new RawDataset(name);
				dataset.AddRecords("inputs", new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 1 });
				dataset.AddRecords("targets", new[] { 0 }, new[] { 1 }, new[] { 1 }, new[] { 0 });

				return dataset;
			}

			#endregion

			#region Extracted datasets

			/// <summary>
			/// Create an extracted IRIS dataset, automatically download any required resources.
			/// This dataset is normalised, one-hot-target-preprocessed and shuffled.
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The IRIS dataset.</returns>
			public static IDataset Iris(string name = "iris")
			{
				IRecordExtractor irisExtractor = new CsvRecordReader(
					new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")))
					.Extractor("inputs", new[] { 0, 3 }, "targets", 4)
					.AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica")
					.Preprocess(new OneHotPreprocessor("targets", minValue: 0, maxValue: 2))
					.Preprocess(new AdaptivePerIndexNormalisingPreprocessor(minOutputValue: 0.0, maxOutputValue: 1.0, sectionNames: "inputs"))
					.Preprocess(new ShufflePreprocessor());

				return new ExtractedDataset(name, ExtractedDataset.BlockSizeAuto, false, irisExtractor);
			}

			/// <summary>
			/// Create an extracted MNIST dataset, automatically download any required resources (may take a while).
			/// This dataset is normalised and one-hot-target-preprocessed.
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The MNIST dataset.</returns>
			public static IDataset Mnist(string name = "mnist")
			{
				IRecordExtractor mnistImageExtractor = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28,
					source: new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))))
					.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

				IRecordExtractor mnistTargetExtractor = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1,
					source: new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))))
					.Extractor("targets", new[] { 0L }, new[] { 1L }).Preprocess(new OneHotPreprocessor(minValue: 0, maxValue: 9));

				return new ExtractedDataset(name, ExtractedDataset.BlockSizeAuto, false, mnistImageExtractor, mnistTargetExtractor);
			}


			/// <summary>
			/// Create an extracted WDBC (Wisconsin Diagnostic Breast Cancer) dataset, automatically download any required resources.
			/// This dataset is normalised and shuffled.
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>THe WDBC dataset.</returns>
			public static IDataset Wdbc(string name = "wdbc")
			{
				IRecordExtractor extractor = new CsvRecordReader(
					new MultiSource(new FileSource("wdbc.data"), new UrlSource("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")))
					.Extractor("inputs", new[] { 2, 31 }, "targets", 1)
					.AddValueMapping(1, "M", "B")
					.Preprocess(new AdaptivePerIndexNormalisingPreprocessor(0.0, 1.0, "inputs"))
					.Preprocess(new ShufflePreprocessor());

				return new ExtractedDataset(name, extractor);
			}

			/// <summary>
			/// Create an extracted heart disease dataset, automatically download any required resources.
			/// This dataset is normalised, one-hot-target-preprocessed and shuffled.
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The heart disease dataset.</returns>
			public static IDataset HeartDisease(string name = "heart_disease")
			{
				IRecordExtractor extractor = new CsvRecordReader(separator: ' ',
						source: new MultiSource(new FileSource("reprocessed.hungarian.data"), new UrlSource("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data")))
					.Extractor("inputs", new[] { 0, 12 }, "targets", 13)
					.AddValueMapping(3, new Dictionary<object, object> { ["-9"] = 132.133 }) // substitute averages for missing values
					.AddValueMapping(4, new Dictionary<object, object> { ["-9"] = 231.224 })
					.AddValueMapping(7, new Dictionary<object, object> { ["-9"] = 138.656 })
					.AddSpanValueMapping(8, 12, new Dictionary<object, object> { ["-9"] = 0 })
					.AddSpanValueMapping(5, 6, new Dictionary<object, object> { ["-9"] = 0 })
					.Preprocess(new OneHotPreprocessor("targets", 0, 4))
					.Preprocess(new AdaptivePerIndexNormalisingPreprocessor(0.0, 1.0, "inputs"))
					.Preprocess(new ShufflePreprocessor());

				return new ExtractedDataset(name, extractor);
			}

			/// <summary>
			/// Create an extracted parkinsons dataset, automatically download any required resources.
			/// This dataset is normalised and shuffled.
			/// </summary>
			/// <param name="name"></param>
			/// <returns></returns>
			public static IDataset Parkinsons(string name = "parkinsons")
			{
				IRecordExtractor extractor = new CsvRecordReader(skipFirstLine: true,
					source: new MultiSource(new FileSource("parkinsons.data"), new UrlSource("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")))
					.Extractor("inputs", new[] { 1, 16 }, new[] { 18, 23 }, "targets", 17)
					.Preprocess(new PerIndexNormalisingPreprocessor(0.0, 1.0, "inputs"))
					.Preprocess(new ShufflePreprocessor());

				return new ExtractedDataset(name, extractor);
			}

			/// <summary>
			/// Create an extracted Connect4 dataset (6 by 7 field).
			/// The dataset is shuffled and one-hot-target-preprocessed.
			/// </summary>
			/// <param name="name">The optional name.</param>
			/// <returns>The Connect4 dataset.</returns>
			public static IDataset Connect4(string name = "connect4")
			{
				CsvRecordExtractor csvExtractor = new CsvRecordReader(
					new MultiSource(new FileSource("connect-4.data"), new UrlSource("https://raw.githubusercontent.com/moroshko/connect4/master/connect-4.data")))
					.Extractor("inputs", new[] { 0, 41 }, "targets", 42)
					.AddValueMapping(42, "loss", "draw", "win");

				Dictionary<object, object> mappings = new Dictionary<object, object>()
				{
					["x"] = 1,
					["b"] = 0,
					["o"] = -1
				};

				for (int i = 0; i < 42; i++)
				{
					csvExtractor.AddValueMapping(i, mapping: mappings);
				}

				var extractor = csvExtractor
					.Preprocess(new OneHotPreprocessor("targets", 0, 2))
					.Preprocess(new ShufflePreprocessor());

				return new ExtractedDataset(name, 67557, extractor);
			}

			public static IDataset TicTacToe(string name = "tictactoe")
			{
				int[] board = new int[3 * 3];
				int[] states = new int[] { -1, 0, 1 }; //player o, empty, player x

				IDictionary<int[], int[]> scoredBoards = new Dictionary<int[], int[]>();

				_InternalScoreBoardsRec(0, board, states, scoredBoards);
				
				Random rng = new Random();

				var scoredBoardsAsArray = scoredBoards.ToArray().OrderBy(x => rng.Next());

				RawDataset dataset = new RawDataset(name);

				dataset.AddRecords("inputs", scoredBoardsAsArray.Select(x => x.Key).ToArray());
				dataset.AddRecords("targets", scoredBoards.Select(x => x.Value).ToArray());

				return dataset;
			}

			private static void _InternalScoreBoardsRec(int currentPosition, int[] board, int[] states, IDictionary<int[], int[]> scoredBoards)
			{
				for (int i = 0; i < states.Length; i++)
				{
					board[currentPosition] = states[i];
					int[] boardClone = (int[])board.Clone();
					int score = _InternalScoreBoard(boardClone);

					if (score < 2 && currentPosition < board.Length - 1)
					{
						_InternalScoreBoardsRec(currentPosition + 1, boardClone, states, scoredBoards);
					}
					else
					{
						int[] scoreOneHot = new int[3];
						scoreOneHot[score] = 1;

						//Console.WriteLine($"{boardClone[0]} {boardClone[1]} {boardClone[2]}\n" +
						//					$"{boardClone[3]} {boardClone[4]} {boardClone[5]} \t => {score}\n" +
						//					$"{boardClone[6]} {boardClone[7]} {boardClone[8]}\n");

						scoredBoards.Add(boardClone, scoreOneHot);
					}
				}
			}

			private static int _InternalScoreBoard(int[] board)
			{
				int score = 0;

				for (int i = 0; i < board.Length; i++)
				{
					if (board[i] != 1) continue;

					int row = i / 3, col = i % 3;
					int horizontalStreak = 0, verticalStreak = 0;

					for (int y = i + 1; y < board.Length; y++)
					{
						int otherRow = y / 3;
						if (otherRow != row || board[i] != board[y]) break;

						horizontalStreak++;
					}

					if (horizontalStreak == 1) score = 1;
					if (horizontalStreak == 2) return 2;

					for (int y = i + 1; y < board.Length; y++)
					{
						int otherCol = y % 3;
						if (otherCol != col || board[i] != board[y]) break;

						verticalStreak++;
					}

					if (verticalStreak == 1) score = 1;
					if (verticalStreak == 2) return 2;
				}

				//if (board[4] == 1)
				//{
				//	if (board[0] == board[4])
				//	{
				//		score = 1;
				//		if (board[4] == board[8]) return 2;
				//	}
				//	else if (board[4] == board[8]) score = 1;

				//	if (board[2] == board[4])
				//	{
				//		score = 1;
				//		if (board[4] == board[6]) return 2;
				//	}
				//	else if (board[4] == board[6]) score = 1;
				//}

				return score;
			}

			#endregion
		}
	}
}
