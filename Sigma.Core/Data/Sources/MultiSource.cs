/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Utils;
using System;
using System.IO;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A collection of data set sources which automatically chooses the first existing source and then acts like that data source.
	/// </summary>
	[Serializable]
	public class MultiSource : IDataSource
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// The data source active in this multi source which this data source will imitate.
		/// Can be null if not yet set or none of the given source exists.
		/// </summary>
		public IDataSource ActiveSource { get; private set; }

		public string ResourceName
		{
			get
			{
				if (ActiveSource == null)
				{
					throw new InvalidOperationException($"Cannot get resource name of multi source {this} because no underlying source was active.");
				}

				return ActiveSource?.ResourceName;
			}
		}

		public bool Seekable => ActiveSource?.Seekable ?? false;

		private bool _fetchedActiveSource;

		private readonly IDataSource[] _sources;

		/// <summary>
		/// Create a multi data set source with a certain array of underlying source, of which the first existing source will be exposed.
		/// </summary>
		/// <param name="sources">The array of underlying sources to consider.</param>
		public MultiSource(params IDataSource[] sources)
		{
			if (sources == null)
			{
				throw new ArgumentNullException(nameof(sources));
			}

			if (sources.Length == 0)
			{
				throw new ArgumentException("There must be > 0 sources, but sources array length was 0.");
			}

			for (int i = 0; i < sources.Length; i++)
			{
				if (sources[i] == null)
				{
					throw new ArgumentNullException($"No part of the sources array can be null but element at index {i} in the source array was null.");
				}
			}

			_sources = sources;

			FetchActiveSource();
		}

		private void FetchActiveSource()
		{
			if (_fetchedActiveSource)
			{
				return;
			}

			foreach (IDataSource source in _sources)
			{
				if (source.Exists())
				{
					ActiveSource = source;

					_logger.Debug($"Found existing underlying source {source}, set as active source and forwarding its output.");

					break;
				}
			}

			_fetchedActiveSource = true;
		}

		public bool Exists()
		{
			return ActiveSource?.Exists() ?? false;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare multi source, none of the underlying sources {ArrayUtils.ToString(_sources)} exists.");
			}

			ActiveSource.Prepare();
		}

		public Stream Retrieve()
		{
			if (ActiveSource == null)
			{
				throw new InvalidOperationException("Cannot retrieve multi source, the multi source was not properly prepared (none of the sources exists or missing Prepare() call).");
			}

			return ActiveSource.Retrieve();
		}

		public void Dispose()
		{
			ActiveSource?.Dispose();
		}
	}
}
