/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.ComponentModel;
using System.Windows.Data;
using log4net;
using log4net.Appender;
using log4net.Core;
using log4net.Filter;
using log4net.Repository.Hierarchy;
using Sigma.Core.Monitors.WPF.Panels.DataGrids;

namespace Sigma.Core.Monitors.WPF.Panels.Logging
{
	/// <summary>
	/// A container class for a processed <see cref="LoggingEvent"/>.
	/// </summary>
	public class LogEntry
	{
		/// <summary>
		/// The date and time when the log occurred.
		/// </summary>
		public DateTime TimeStamp { get; set; }

		/// <summary>
		/// The severity of the log (i.e log level).
		/// </summary>
		public Level Level { get; set; }

		/// <summary>
		/// The thread in which the event occurred.
		/// </summary>
		public string Thread { get; set; }

		/// <summary>
		/// The name of the logger (i.e. class name without namespace).
		/// </summary>
		public string Logger { get; set; }

		/// <summary>
		/// The message that was passed to the entry. 
		/// </summary>
		public string Message { get; set; }

		/// <summary>
		/// Create a <see cref="LogEntry"/> that automatically populates its data from a
		/// <see cref="LoggingEvent"/>.
		/// </summary>
		/// <param name="loggingEvent">The event from which the data is processed.</param>
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

	/// <summary>
	/// A sophisticated log viewer that visualises the data inside a DataGrid. 
	/// </summary>
	public class LogDataGridPanel : SimpleDataGridPanel<LogEntry>, IAppender
	{
		private readonly IFilter _logfilter;

		/// <summary>
		/// Create a <see cref="LogDataGridPanel"/>.
		/// </summary>
		/// <param name="title">The title of the <see cref="SigmaPanel"/>.</param>
		/// <param name="logfilter">The filter that decides whether a log is displayed or not.</param>
		/// <param name="content">The content that will be displayed instead of the title if not <c>null</c>.</param>
		public LogDataGridPanel(string title, IFilter logfilter = null, object content = null) : base(title, content)
		{
			_logfilter = logfilter;

			// Sort after 
			ICollectionView dataView = CollectionViewSource.GetDefaultView(Content.ItemsSource);
			//clear the existing sort order
			dataView.SortDescriptions.Clear();
			//create a new sort order for the sorting that is done lastly
			dataView.SortDescriptions.Add(new SortDescription(nameof(LogEntry.TimeStamp), ListSortDirection.Descending));
			//refresh the view which in turn refresh the grid
			dataView.Refresh();

			// ReSharper disable once VirtualMemberCallInConstructor
			AssignToLog();
		}

		/// <summary>
		/// Assign to the log manager.
		/// </summary>
		protected virtual void AssignToLog()
		{
			((Hierarchy) LogManager.GetRepository()).Root.AddAppender(this);
		}

		void IAppender.DoAppend(LoggingEvent loggingEvent)
		{
			if (_logfilter == null || _logfilter.Decide(loggingEvent) == FilterDecision.Accept)
			{
				Dispatcher.InvokeAsync(() => AddItem(new LogEntry(loggingEvent)));
				//Dispatcher.Invoke(DispatcherPriority.Background, new ParameterizedThreadStart(AddItem), new LogEntry(loggingEvent));
			}
		}

		private void AddItem(object item)
		{
			Items.Add((LogEntry) item);
			//Content.ScrollIntoView(item);
		}

		void IAppender.Close() { }
	}
}
