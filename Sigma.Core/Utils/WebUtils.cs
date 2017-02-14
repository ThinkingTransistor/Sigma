/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using log4net;

namespace Sigma.Core.Utils
{
	public class WebUtils
	{
		private static readonly ILog Logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Get the proxy from the parameters (without credentials) - this may be easier than <see cref="GetProxyFromFileOrDefault"/>.
		/// </summary>
		/// <param name="address">The address of the proxy server.</param>
		/// <param name="port">The port of the proxy server.</param>
		/// <returns>A <see cref="IWebProxy"/> based on the given parameters.</returns>
		public static IWebProxy GetProxy(string address, int port = 8080)
		{
			return GetProxy(address, port, null, null);
		}

		/// <summary>
		/// Get the proxy from the parameters - this may be easier than <see cref="GetProxyFromFileOrDefault"/>.
		/// </summary>
		/// <param name="address">The address of the proxy server.</param>
		/// <param name="port">The port of the proxy server.</param>
		/// <param name="username">The username - this can be null (password is also ignored).</param>
		/// <param name="password">The password - this can be null (username is also ignored).</param>
		/// <returns>A <see cref="IWebProxy"/> based on the given parameters.</returns>
		public static IWebProxy GetProxy(string address, int port, string username, string password)
		{
			if (string.IsNullOrEmpty(address))
			{
				throw new ArgumentNullException(nameof(address));
			}

			if (port <= 0 || port >= 65536)
			{
				throw new ArgumentOutOfRangeException(nameof(port));
			}

			WebProxy proxy = new WebProxy(address, port);
			if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(password))
			{
				proxy.Credentials = new NetworkCredential(username, password);
			}

			return proxy;
		}

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
				defaultProxy = WebRequest.DefaultWebProxy;
			}

			if (!File.Exists(filepath))
			{
				Logger.Info($"Using default web proxy ({defaultProxy}), given proxy file \"{filepath}\" does not exist.");

				return defaultProxy;
			}

			string address = null;
			int port = 80;

			string username = null;
			string password = "";

			using (StreamReader file = File.OpenText(filepath))
			{
				string line;
				while ((line = file.ReadLine()) != null)
				{
					string key = line.Substring(0, line.IndexOf('='));
					string value = line.Substring(line.IndexOf('=') + 1);

					try
					{
						if (Match(key, "address", "proxyaddress"))
						{
							address = value.Trim();
						}
						else if (Match(key, "port", "proxyport"))
						{
							port = int.Parse(value.Trim());
						}
						else if (Match(key, "user", "username"))
						{
							username = value.Trim();
						}
						else if (Match(key, "pass", "password"))
						{
							password = value.Trim();
						}
					}
					catch (Exception ex)
					{
						Logger.Warn($"Invalid entry at line {line} in file \"{filepath}\".", ex);
					}
				}
			}

			if (address == null)
			{
				Logger.Info($"Using default web proxy ({defaultProxy}), given proxy file \"{filepath}\" did not contain a proxy address (key=\"address\").");

				return defaultProxy;
			}

			WebProxy customProxy = new WebProxy(address, port);

			bool credentialsProvided = username != null;
			if (credentialsProvided)
			{
				customProxy.Credentials = new NetworkCredential(username, password);
			}

			Logger.Info($"Using custom proxy ({defaultProxy}) " + (credentialsProvided ? "with" : "without") + "credentials");

			return customProxy;
		}

		private static bool Match(string actual, params string[] toMatch)
		{
			return toMatch.Any(match => actual.Equals(match, StringComparison.OrdinalIgnoreCase));
		}
	}

	/// <summary>
	/// A custom WebClient implementation which allows downloading a file with progress reporting within the calling thread via events. 
	/// </summary>
	public class BlockingWebClient : WebClient
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly int _timeoutMilliseconds;

		public long PreviousBytesReceived;

		private bool _downloadSuccess;
		private IProgress<float> _downloadProgress;

		private readonly EventWaitHandle _asyncWait = new ManualResetEvent(false);
		private readonly Timer _timeoutTimer;

		public delegate void ProgressChanged(long newBytesReceived, long totalBytesReceived, long totalBytes, int progressPercentage);

		public event ProgressChanged ProgressChangedEvent;

		public BlockingWebClient(int timeoutMilliseconds = 16000, IWebProxy proxy = null)
		{
			if (timeoutMilliseconds <= 0)
			{
				throw new ArgumentException($"Timeout must be > 0, but timeout was {timeoutMilliseconds}.");
			}

			_timeoutMilliseconds = timeoutMilliseconds;

			DownloadFileCompleted += DownloadFileCompletedHandle;
			DownloadProgressChanged += DownloadProgressChangedHandle;

			_timeoutTimer = new Timer(OnTimeout, null, _timeoutMilliseconds, Timeout.Infinite);

			Proxy = proxy ?? SigmaEnvironment.Globals.Get<IWebProxy>("web_proxy");
		}

		private void OnProgressChanged(long newBytesReceived, long totalBytesReceived, long totalBytes, int progressPercentage)
		{
			ProgressChangedEvent?.Invoke(newBytesReceived, totalBytesReceived, totalBytes, progressPercentage);
		}

		private void OnTimeout(object ignored)
		{
			if (_downloadSuccess)
			{
				return;
			}

			CancelAsync();
			_downloadSuccess = false;

			_logger.Warn($"Aborted download, connection timed out (more than {_timeoutMilliseconds}ms passed since client last received anything).");

			_asyncWait.Set();
		}

		/// <summary>
		/// A custom download file method which enables progress reporting within the same thread. 
		/// </summary>
		/// <param name="url">The url to download from.</param>
		/// <param name="outputPath">The output path (where the downloaded file will be stored).</param>
		/// <param name="progress">The optional progress reporter to report progress to.</param>
		/// <returns>A boolean indicating whether the download was successful.</returns>
		public bool DownloadFile(string url, string outputPath, IProgress<float> progress = null)
		{
			_downloadSuccess = false;
			_downloadProgress = progress;

			_asyncWait.Reset();

			Uri uri = new Uri(url);

			DownloadFileAsync(uri, outputPath);

			_asyncWait.WaitOne();

			if (PreviousBytesReceived <= 0)
			{
				_downloadSuccess = false;
			}

			return _downloadSuccess;
		}

		private void DownloadFileCompletedHandle(object sender, System.ComponentModel.AsyncCompletedEventArgs ev)
		{
			_asyncWait.Set();

			_downloadSuccess = true;
		}

		private void DownloadProgressChangedHandle(object sender, DownloadProgressChangedEventArgs ev)
		{
			long newBytesReceived = ev.BytesReceived - PreviousBytesReceived;
			PreviousBytesReceived = ev.BytesReceived;

			OnProgressChanged(newBytesReceived, PreviousBytesReceived, ev.TotalBytesToReceive, ev.ProgressPercentage);

			_downloadProgress?.Report(ev.ProgressPercentage);

			_timeoutTimer.Change(_timeoutMilliseconds, Timeout.Infinite);
		}

		protected override WebRequest GetWebRequest(Uri address)
		{
			WebRequest request = base.GetWebRequest(address);

			if (request != null)
			{
				request.Timeout = _timeoutMilliseconds;
			}

			return request;
		}
	}
}
