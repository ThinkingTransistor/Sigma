using System.Windows;
using System.Windows.Controls;
using log4net.Appender;
using log4net.Core;

namespace Sigma.Core.Monitors.WPF.Panels.Logging
{
	public class LogPanel : SigmaPanel, IAppender
	{
		public new TextBox Content;

		public LogPanel(string title) : base(title)
		{
			Content = new TextBox { IsReadOnly = true, TextWrapping = TextWrapping.Wrap, AcceptsReturn = true, Margin = new Thickness(10) };
			base.Content = Content;

			((log4net.Repository.Hierarchy.Hierarchy) log4net.LogManager.GetRepository()).Root.AddAppender(this);
		}

		public void Close()
		{

		}

		public void DoAppend(LoggingEvent loggingEvent)
		{
			Dispatcher.Invoke(() => Content.AppendText($"{loggingEvent.Level.Name} {loggingEvent.MessageObject}\n"));
		}
	}
}