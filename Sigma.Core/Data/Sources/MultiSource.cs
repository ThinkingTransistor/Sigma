/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A collection of data set sources which automatically chooses the first existing source and then acts like that data source.
	/// </summary>
	public class MultiSource : IDataSetSource
	{
		/// <summary>
		/// The data source active in this multi source which this data source will imitate.
		/// Can be null if not yet set or none of the given source exists.
		/// </summary>
		public IDataSetSource ActiveSource { get; private set; }

		public string ResourceName { get { return ActiveSource?.ResourceName; } }

		public bool Seekable { get { return ActiveSource?.Seekable ?? false; } }

		private bool fetchedActiveSource;

		private IDataSetSource[] sources;

		/// <summary>
		/// Create a multi data set source with a certain array of underlying source, of which the first existing source will be exposed.
		/// </summary>
		/// <param name="sources">The array of underlying sources to consider.</param>
		public MultiSource(params IDataSetSource[] sources)
		{
			if (sources == null)
			{
				throw new ArgumentNullException("Sources cannot be null.");
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

			this.sources = sources;

			FetchActiveSource();
		}

		private void FetchActiveSource()
		{
			if (fetchedActiveSource)
			{
				return;
			}

			foreach (IDataSetSource source in sources)
			{
				if (source.Exists())
				{
					ActiveSource = source;

					break;
				}
			}

			fetchedActiveSource = true;
		}

		public bool Exists()
		{
			return ActiveSource?.Exists() ?? false;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare multi source, none of the underlying sources {ArrayUtils.ToString(sources)} exists.");
			}

			ActiveSource.Prepare();
		}

		public Stream Retrieve()
		{
			if (ActiveSource == null)
			{
				throw new InvalidOperationException($"Cannot retrieve multi source, the multi source was not properly prepared (none of the sources exists or missing Prepare() call).");
			}

			return ActiveSource.Retrieve();
		}

		public void Dispose()
		{
			ActiveSource?.Dispose();
		}
	}
}
