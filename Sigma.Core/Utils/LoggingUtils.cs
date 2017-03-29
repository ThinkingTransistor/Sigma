/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using log4net.Core;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility methods for logging with <see cref="log4net"/>.
	/// </summary>
	public static class LoggingUtils
	{
		/// <summary>
		/// Dynamically log a message to a certain level using a certain logger.
		/// </summary>
		/// <param name="level">The log level (fatal, error, warn, info, debug).</param>
		/// <param name="message">The message to log.</param>
		/// <param name="logger">The logger to use.</param>
		public static void Log(Level level, string message, ILog logger)
		{
			if (level == Level.Fatal)
			{
				logger.Fatal(message);
			}
			else if (level == Level.Error)
			{
				logger.Error(message);
			}
			else if (level == Level.Warn)
			{
				logger.Warn(message);
			}
			else if (level == Level.Info)
			{
				logger.Info(message);
			}
			else if (level == Level.Debug)
			{
				logger.Debug(message);
			}
		}
	}
}
