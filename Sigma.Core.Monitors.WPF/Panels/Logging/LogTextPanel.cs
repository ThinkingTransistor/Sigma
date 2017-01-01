/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using log4net;
using log4net.Appender;
using log4net.Core;
using log4net.Repository.Hierarchy;

namespace Sigma.Core.Monitors.WPF.Panels.Logging
{
	public class LogTextPanel : SigmaPanel, IAppender
	{
		public new TextBox Content;

		public LogTextPanel(string title) : base(title)
		{
			Content = new TextBox
			{
				IsReadOnly = true,
				TextWrapping = TextWrapping.Wrap,
				AcceptsReturn = true,
				Margin = new Thickness(10)
			};
			base.Content = Content;

			((Hierarchy) LogManager.GetRepository()).Root.AddAppender(this);
		}

		public void Close()
		{
		}

		public void DoAppend(LoggingEvent loggingEvent)
		{
			Dispatcher.Invoke(() => Content.AppendText($"{loggingEvent.Level.Name}\t{loggingEvent.MessageObject}\n"));
		}
	}
}