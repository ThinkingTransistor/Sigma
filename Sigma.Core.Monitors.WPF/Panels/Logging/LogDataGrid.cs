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
			Message = loggingEvent.MessageObject.ToString();
		}
	}

	public class LogDataGrid : SimpleDataGridPanel<LogEntry>, IAppender
	{
		public LogDataGrid(string title) : base(title)
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
			//Content.LoadingRow += LoadingRow;
		}

		//private void LoadingRow(object sender, DataGridRowEventArgs e)
		//{
		//	DataGridRow row = e.Row;
		//	DataRowView rView = row.Item as DataRowView;

		//	if (rView != null && rView.Row.ItemArray[2].ToString().Contains("DEBUG"))
		//	{
		//		row.Background = Brushes.Green;
		//	}
		//	else
		//	{
		//		row.Background = Brushes.Red;
		//	}

		//}

		public void Close()
		{
		}
	}
}