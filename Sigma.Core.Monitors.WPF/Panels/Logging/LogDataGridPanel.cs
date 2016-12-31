using System;
using System.Threading;
using System.Windows.Threading;
using log4net;
using log4net.Appender;
using log4net.Core;
using log4net.Repository.Hierarchy;
using Sigma.Core.Monitors.WPF.Panels.DataGrids;

namespace Sigma.Core.Monitors.WPF.Panels.Logging
{
	public class LogEntry
	{
		public DateTime TimeStamp { get; set; }
		public Level Level { get; set; }
		public string Thread { get; set; }
		public string Logger { get; set; }
		public string Message { get; set; }

		public LogEntry(LoggingEvent loggingEvent)
		{
			TimeStamp = loggingEvent.TimeStamp;
			Level = loggingEvent.Level;
			Thread = loggingEvent.ThreadName;
			Logger = loggingEvent.LoggerName;
			int lastIndex = Logger.LastIndexOf(".", StringComparison.Ordinal) + 1;
			Logger = Logger.Substring(lastIndex, Logger.Length - lastIndex);

			Message = loggingEvent.MessageObject.ToString();
		}
	}

	public class LogDataGridPanel : SimpleDataGridPanel<LogEntry>, IAppender
	{
		public LogDataGridPanel(string title) : base(title)
		{
			// ReSharper disable once VirtualMemberCallInConstructor
			AssignToLog();
		}

		protected virtual void AssignToLog()
		{
			((Hierarchy) LogManager.GetRepository()).Root.AddAppender(this);
		}

		public void DoAppend(LoggingEvent loggingEvent)
		{
			Dispatcher.Invoke(DispatcherPriority.Background, new ParameterizedThreadStart(AddItem), new LogEntry(loggingEvent));
		}

		private void AddItem(object item)
		{
			Items.Add((LogEntry) item);
			//Content.ScrollIntoView(item);
		}

		public void Close()
		{
		}
	}
}