/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.IO;
using System.Net;
using System.Threading;

namespace Sigma.Core.Utils
{
	public class WebUtils
	{
		private static ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Reads a proxy configuration from a file and returns a custom proxy if file exists and is valid, get a default proxy otherwise.
		/// </summary>
		/// <param name="filepath">The proxy configuration file path.</param>
		/// <param name="defaultProxy">The default proxy to return if the file was not found or could not be parsed.</param>
		/// <returns>The custom proxy as specified in the file if found and valid, the default proxy otherwise.</returns>
		public static IWebProxy GetProxyFromFileOrDefault(string filepath, IWebProxy defaultProxy = null)
		{
			if (defaultProxy == null)
			{
				defaultProxy = System.Net.WebRequest.DefaultWebProxy;
			}

			if (!File.Exists(filepath))
			{
				return defaultProxy;
			}

			WebProxy tempProxy = new WebProxy();
			string address = null;
			int port = 80;

			string username = null;
			string password = "";

			using (StreamReader file = System.IO.File.OpenText(filepath))
			{
				string line;
				while ((line = file.ReadLine()) != null)
				{
					string[] parts = line.Split('=');
					string key = parts[0];
					string value = parts[1];

					try
					{
						if (key == "address" || key == "proxyaddress")
						{
							address = value.Trim();
						}
						else if (key == "port" || key == "proxyport")
						{
							port = int.Parse(value.Trim());
						}
						else if (key == "user" || key == "username")
						{
							username = value.Trim();
						}
						else if (key == "pass" || key == "password")
						{
							password = value.Trim();
						}
					}
					catch (Exception ex)
					{
						logger.Warn($"Invalid entry at line {line} in file {filepath}.", ex);	
					}
				}
			}

			if (address == null)
			{
				return defaultProxy;
			}

			WebProxy customProxy = new WebProxy(address, port);

			if (username != null)
			{
				customProxy.Credentials = new NetworkCredential(username, password);
			}

			return customProxy;
		}
	}

	/// <summary>
	/// A custom WebClient implementation which allows downloading a file with progress reporting within the calling thread via events. 
	/// </summary>
	public class BlockingWebClient : WebClient
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private int timeoutMilliseconds;

		public long previousBytesReceived;

		private bool downloadSuccess;

		private EventWaitHandle asyncWait = new ManualResetEvent(false);
		private Timer timeoutTimer = null;

		public delegate void ProgressChanged(long newBytesReceived, long totalBytesReceived, long totalBytes, int progressPercentage);

		public event ProgressChanged progressChangedEvent;

		public BlockingWebClient(int timeoutMilliseconds = 16000, WebProxy proxy = null)
		{
			if (timeoutMilliseconds <= 0)
			{
				throw new ArgumentException($"Timeout must be > 0, but timeout was {timeoutMilliseconds}.");
			}

			this.timeoutMilliseconds = timeoutMilliseconds;

			this.DownloadFileCompleted += new System.ComponentModel.AsyncCompletedEventHandler(DownloadFileCompletedHandle);
			this.DownloadProgressChanged += new DownloadProgressChangedEventHandler(DownloadProgressChangedHandle);

			this.timeoutTimer = new Timer(this.OnTimeout, null, this.timeoutMilliseconds, System.Threading.Timeout.Infinite);

			this.Proxy = proxy;

			if (this.Proxy == null)
			{
				this.Proxy = SigmaEnvironment.Globals.Get<IWebProxy>("webProxy");
			}
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

		/// <summary>
		/// A custom download file method which enables progress reporting within the same thread. 
		/// </summary>
		/// <param name="url">The url to download from.</param>
		/// <param name="outputPath">The output path (where the downloaded file will be stored).</param>
		/// <returns>A boolean indicating whether the download was successful.</returns>
		public new bool DownloadFile(string url, string outputPath)
		{
			this.downloadSuccess = false;

			this.asyncWait.Reset();

			Uri uri = new Uri(url);

			base.DownloadFileAsync(uri, outputPath);

			this.asyncWait.WaitOne();

			if (previousBytesReceived <= 0)
			{
				downloadSuccess = false;
			}

			return downloadSuccess;
		}

		private void DownloadFileCompletedHandle(object sender, System.ComponentModel.AsyncCompletedEventArgs ev)
		{
			this.asyncWait.Set();

			this.downloadSuccess = true;
		}

		private void DownloadProgressChangedHandle(object sender, DownloadProgressChangedEventArgs ev)
		{
			long newBytesReceived = ev.BytesReceived - previousBytesReceived;
			previousBytesReceived = ev.BytesReceived;

			OnProgressChanged(newBytesReceived, previousBytesReceived, ev.TotalBytesToReceive, ev.ProgressPercentage);

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
