/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
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
			if (level.Value == Level.Fatal.Value)
			{
				logger.Fatal(message);
			}
			else if (level.Value == Level.Error.Value)
			{
				logger.Error(message);
			}
			else if (level.Value == Level.Warn.Value)
			{
				logger.Warn(message);
			}
			else if (level.Value == Level.Info.Value)
			{
				logger.Info(message);
			}
			else if (level.Value == Level.Debug.Value)
			{
				logger.Debug(message);
			}
			else
			{
			    throw new ArgumentException($"Level {level} is not a supported logging level (supported levels are fatal, error, warn, info, debug).");
            }
        }
	}
}
