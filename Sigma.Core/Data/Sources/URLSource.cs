/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Sources
{
	public class URLSource : IDataSetSource
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private bool exists;
		private bool checkedExists;

		private string url;
		private string localDownloadPath;
		private FileStream localDownloadedFileStream;

		public bool Chunkable
		{
			get { return true; }
		}

		public URLSource(string url) : this(url, SigmaEnvironment.Globals["datasets"] + GetFileNameFromURL(url))
		{
		}

		public URLSource(string url, string localDownloadPath)
		{
			if (url == null)
			{
				throw new ArgumentNullException($"URL cannot be null.");
			}

			if (localDownloadPath == null)
			{
				throw new ArgumentNullException($"Local download path cannot be null.");
			}

			this.url = url;
			this.localDownloadPath = localDownloadPath;
		}

		private static string GetFileNameFromURL(string url)
		{
			return System.IO.Path.GetFileName(new Uri(url).LocalPath);
		}

		private bool CheckExists()
		{
			logger.Info($"Establishing web connection to check if URL {url} exists and is accessible...");

			HttpWebRequest request = WebRequest.Create(url) as HttpWebRequest;

			request.Method = "HEAD";

			try
			{
				HttpWebResponse response = request.GetResponse() as HttpWebResponse;

				this.exists = response.StatusCode == HttpStatusCode.OK;
			}
			catch
			{
				this.exists = false;
			}

			if (exists)
			{
				logger.Info($"Web connection ended, URL \"{url}\" exists and is accessible.");
			}
			else
			{
				logger.Info($"Web connection ended, URL \"{url}\" does not exist or is not accessible.");
			}

			return this.exists;
		}

		public bool Exists()
		{
			if (!checkedExists)
			{
				CheckExists();

				checkedExists = true;
			}

			return exists;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare URL source, underlying URL resource \"{url}\" does not exist.");
			}

			Directory.CreateDirectory(new FileInfo(localDownloadPath).Directory.FullName);

			logger.Info($"Starting download of URL resource \"{url}\" to local path \"{localDownloadPath}\"...");

			using (BlockingWebClient client = new BlockingWebClient(timeoutMilliseconds: 8000))
			{
				//TODO when tasks are done - this should be a task
				//client.progressChangedEvent 
				client.DownloadFile(url, localDownloadPath);
			}

			logger.Info($"Download of URL resource \"{url}\" to local path \"{localDownloadPath}\" completed.");

			localDownloadedFileStream = new FileStream(localDownloadPath, FileMode.Open);
		}

		public Stream Retrieve()
		{
			if (localDownloadedFileStream != null)
			{
				throw new InvalidOperationException("Cannot retrieve URL source, URL source was not fetched correctly (missing or failed Prepare() call?).");
			}

			return localDownloadedFileStream;
		}
	}

	internal class BlockingWebClient : WebClient
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private int timeoutMilliseconds;

		public long previousBytesReceived;

		private bool downloadSuccess;

		private EventWaitHandle asyncWait = new ManualResetEvent(false);
		private Timer timeoutTimer = null;

		public delegate void ProgressChanged(long newBytesReceived, long totalBytesReceived, long totalBytes, int progressPercentage);

		public event ProgressChanged progressChangedEvent;

		public BlockingWebClient(int timeoutMilliseconds = 8000)
		{
			if (timeoutMilliseconds <= 0)
			{
				throw new ArgumentException($"Timeout must be > 0, but timeout was {timeoutMilliseconds}.");
			}

			this.timeoutMilliseconds = timeoutMilliseconds;

			this.DownloadFileCompleted += new System.ComponentModel.AsyncCompletedEventHandler(DownloadFileCompletedHandle);
			this.DownloadProgressChanged += new DownloadProgressChangedEventHandler(DownloadProgressChangedHandle);

			this.timeoutTimer = new Timer(this.OnTimeout, null, this.timeoutMilliseconds, System.Threading.Timeout.Infinite);
		}

		private void OnProgressChanged(long newBytesReceived, long totalBytesReceived, long totalBytes, int progressPercentage)
		{
			if (this.progressChangedEvent != null)
			{
				this.progressChangedEvent(newBytesReceived, totalBytesReceived, totalBytes, progressPercentage);
			}
		}

		private void OnTimeout(object ignored)
		{
			if (this.downloadSuccess)
			{
				return;
			}

			this.CancelAsync();
			this.downloadSuccess = false;

			this.logger.Warn($"Aborted download, connection timed out (more than {timeoutMilliseconds}ms passed since client last received anything).");

			this.asyncWait.Set();
		}

		public new bool DownloadFile(string url, string outputPath)
		{
			this.downloadSuccess = false;

			this.asyncWait.Reset();

			Uri uri = new Uri(url);

			base.DownloadFileAsync(uri, outputPath);

			this.asyncWait.WaitOne();

			return downloadSuccess;
		}

		private void DownloadFileCompletedHandle(object sender, System.ComponentModel.AsyncCompletedEventArgs ev)
		{
			this.asyncWait.Set();

			this.downloadSuccess = true;
		}

		private void DownloadProgressChangedHandle(object sender, DownloadProgressChangedEventArgs evvent)
		{
			long newBytesReceived = evvent.BytesReceived - previousBytesReceived;
			previousBytesReceived = evvent.BytesReceived;

			OnProgressChanged(newBytesReceived, previousBytesReceived, evvent.TotalBytesToReceive, evvent.ProgressPercentage);

			this.timeoutTimer.Change(this.timeoutMilliseconds, System.Threading.Timeout.Infinite);
		}

		protected override WebRequest GetWebRequest(Uri address)
		{
			WebRequest request = base.GetWebRequest(address);

			request.Timeout = this.timeoutMilliseconds;

			return request;
		}
	}
}
