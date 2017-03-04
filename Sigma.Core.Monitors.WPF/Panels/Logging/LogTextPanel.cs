/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using log4net;
using log4net.Appender;
using log4net.Core;
using log4net.Repository.Hierarchy;

namespace Sigma.Core.Monitors.WPF.Panels.Logging
{
	/// <summary>
	/// This <see cref="LogTextPanel"/> pastes all new log entries in a <see cref="TextBox"/>.
	/// 
	/// For a more sophisticated logger see <see cref="LogDataGridPanel"/>.
	/// </summary>
	public class LogTextPanel : GenericPanel<TextBox>, IAppender
	{
		/// <summary>
		/// A panel that displays log entries in a <see cref="TextBox"/>. It will automatically assign
		/// to the appender list (i.e. receive logs).
		/// </summary>
		/// <param name="title"></param>
		/// <param name="content"></param>
		public LogTextPanel(string title, object content = null) : base(title, content)
		{
			Content = new TextBox
			{
				IsReadOnly = true,
				TextWrapping = TextWrapping.Wrap,
				AcceptsReturn = true,
				Margin = new Thickness(10)
			};

			((Hierarchy) LogManager.GetRepository()).Root.AddAppender(this);
		}

		void IAppender.Close()
		{
		}

		void IAppender.DoAppend(LoggingEvent loggingEvent)
		{
			try
			{

				Dispatcher.Invoke(() => Content.AppendText($"{loggingEvent.Level.Name}\t{loggingEvent.MessageObject}\n"));
			}
			catch (Exception)
			{
				((IAppender) this).Close();
			}
		}
	}
}