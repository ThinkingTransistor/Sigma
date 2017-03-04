/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Utils;
using System;
using System.IO;
using System.Net;
using Sigma.Core.Persistence;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A URL resource used for datasets. Entire resource is downloaded and stored locally for processing.
	/// </summary>
	[Serializable]
	public class UrlSource : IDataSource, ISerialisationNotifier
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string ResourceName { get; }

		public int NumberRetriesOnError { get; }

		public bool Seekable => true;

		private bool _exists;
		private bool _checkedExists;
		private bool _prepared;

		private readonly string _localDownloadPath;
		private readonly string _localTempDownloadPath;

		[NonSerialized]
		private IWebProxy _proxy;

		private FileSource _underlyingFileSource;

		/// <summary>
		/// Create a URL source with a certain URL and store the downloaded file in the datasets directory with an inferred name. 
		/// </summary>
		/// <param name="url">The URL.</param>
		public UrlSource(string url) : this(url, SigmaEnvironment.Globals["datasets_path"] + GetFileNameFromUrl(url))
		{
		}

		public UrlSource(string url, string localDownloadPath) : this(url, localDownloadPath, localDownloadPath + ".sgdownload")
		{
		}

		/// <summary>
		/// Create a URL source with a certain URL and store the downloaded file in the datasets directory with an inferred name. 
		/// </summary>
		/// <param name="url">The URL.</param>
		/// <param name="localDownloadPath">The local download path, where this file will available after completing download.</param>
		/// <param name="localTempDownloadPath">The local temp download path, where the file is downloaded to until download completion.</param>
		/// <param name="proxy">The optional web proxy to use for file downloads.</param>
		/// <param name="numberRetriesOnError">The number of times to retry the download if it failed.</param>
		public UrlSource(string url, string localDownloadPath, string localTempDownloadPath, IWebProxy proxy = null, int numberRetriesOnError = 2)
		{
			if (url == null)
			{
				throw new ArgumentNullException(nameof(url));
			}

			if (localDownloadPath == null)
			{
				throw new ArgumentNullException(nameof(localDownloadPath));
			}

			if (numberRetriesOnError < 0)
			{
				throw new ArgumentException($"Number retries on error must be >= 0, but was {numberRetriesOnError}.");
			}

			ResourceName = url;
			_localDownloadPath = localDownloadPath;
			_localTempDownloadPath = localTempDownloadPath;

			_proxy = proxy ?? SigmaEnvironment.Globals.Get<IWebProxy>("web_proxy");
			NumberRetriesOnError = numberRetriesOnError;
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			_proxy = SigmaEnvironment.Globals.Get<IWebProxy>("web_proxy");
		}

		private static string GetFileNameFromUrl(string url)
		{
			return Path.GetFileName(new Uri(url).LocalPath);
		}

		private void CheckExists()
		{
			_logger.Debug($"Establishing web connection to check if URL {ResourceName} exists and is accessible...");

			HttpWebRequest request = WebRequest.Create(ResourceName) as HttpWebRequest;

			if (request == null)
			{
				throw new InvalidOperationException($"Unable to create web request for {ResourceName}.");
			}

			request.Proxy = _proxy;
			request.Method = "HEAD";

			try
			{
				HttpWebResponse response = request.GetResponse() as HttpWebResponse;

				if (response != null)
				{
					_exists = response.StatusCode == HttpStatusCode.OK;

					response.Dispose();
				}
			}
			catch
			{
				_exists = false;
			}

			_logger.Debug(_exists
				? $"Web connection ended, URL \"{ResourceName}\" exists and is accessible."
				: $"Web connection ended, URL \"{ResourceName}\" does not exist or is not accessible.");
		}

		public bool Exists()
		{
			if (!_checkedExists)
			{
				CheckExists();

				_checkedExists = true;
			}

			return _exists;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare URL source, underlying URL resource \"{ResourceName}\" does not exist or is not accessible.");
			}

			if (_prepared)
			{
				return;
			}

			DirectoryInfo directoryInfo = new FileInfo(_localDownloadPath).Directory;
			if (directoryInfo != null)
			{
				Directory.CreateDirectory(directoryInfo.FullName);
			}

			int numberRetriesLeft = NumberRetriesOnError;
			bool downloadSuccess;

			do
			{
				ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Download, ResourceName);

				if (File.Exists(_localDownloadPath))
				{
					File.Delete(_localDownloadPath);
				}

				_logger.Info($"Downloading URL resource \"{ResourceName}\" to local temp path \"{_localTempDownloadPath}\"...");

				using (BlockingWebClient client = new BlockingWebClient(timeoutMilliseconds: 16000))
				{
					downloadSuccess = client.DownloadFile(ResourceName, _localTempDownloadPath, task);

					if (downloadSuccess)
					{
						_logger.Info($"Completed download of URL resource \"{ResourceName}\" to local temp path \"{_localDownloadPath}\" ({client.PreviousBytesReceived/1024}kB).");

						_logger.Debug($"Moving temp download file \"{_localTempDownloadPath}\" to local download target location \"{_localDownloadPath}\"...");

						try
						{
							File.Move(_localTempDownloadPath, _localDownloadPath);
						}
						catch (Exception e)
						{
							_logger.Error($"Unable to move temporary download file \"{_localTempDownloadPath}\" to local download target location \"{_localDownloadPath}\": " + e);

							SigmaEnvironment.TaskManager.CancelTask(task);
							
							throw;
						}
						_logger.Debug($"Done moving temp download file \"{_localTempDownloadPath}\" to local download target location \"{_localDownloadPath}\".");

						SigmaEnvironment.TaskManager.EndTask(task);
					}
					else
					{
						_logger.Warn($"Failed to download URL source \"{ResourceName}\", could not prepare this URL source correctly.");

						File.Delete(_localTempDownloadPath);

						SigmaEnvironment.TaskManager.CancelTask(task);
					}
				}

				if (!downloadSuccess && numberRetriesLeft > 0)
				{
					_logger.Info($"Retrying download, retry attempt {NumberRetriesOnError - numberRetriesLeft + 1} of {NumberRetriesOnError}...");
				}
			} while (!downloadSuccess && numberRetriesLeft-- > 0); 

			_logger.Debug($"Opened file \"{_localDownloadPath}\".");

			FileInfo localDownloadFileInfo = new FileInfo(_localDownloadPath);

			// cannot move for unit test because of permissions, use temp folder I think?
			_underlyingFileSource = new FileSource(localDownloadFileInfo.Name, localDownloadFileInfo.DirectoryName);
			_underlyingFileSource.Prepare();

			_prepared = true;
		}

		public Stream Retrieve()
		{
			if (_underlyingFileSource == null)
			{
				throw new InvalidOperationException("Cannot retrieve URL source, URL source was not prepared correctly (missing or failed Prepare() call?).");
			}

			return _underlyingFileSource.Retrieve();
		}

		public void Dispose()
		{
			_underlyingFileSource?.Dispose();
		}
	}
}
