//
// ImpObservableCollection.cs
//
// Copyright (c) 2010, Ashley Davis, @@email@@, @@website@@
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted 
// provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this list of conditions 
//   and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//   and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

using System.Windows;

namespace Sigma.Core.Monitors.WPF.NetView.NetworkUIs
{
	/// <summary>
	/// Base class for connection dragging event args.
	/// </summary>
	public class ConnectionDragEventArgs : RoutedEventArgs
	{
		#region Private Data Members

		/// <summary>
		/// The connector that will be dragged out.
		/// </summary>
		protected object connection;

		#endregion Private Data Members

		/// <summary>
		/// The NodeItem or it's DataContext (when non-NULL).
		/// </summary>
		public object Node { get; }

		/// <summary>
		/// The ConnectorItem or it's DataContext (when non-NULL).
		/// </summary>
		public object ConnectorDraggedOut { get; }

		#region Private Methods

		protected ConnectionDragEventArgs(RoutedEvent routedEvent, object source, object node, object connection, object connector) :
			base(routedEvent, source)
		{
			this.Node = node;
			ConnectorDraggedOut = connector;
			this.connection = connection;
		}

		#endregion Private Methods
	}

	/// <summary>
	/// Arguments for event raised when the user starts to drag a connection out from a node.
	/// </summary>
	public class ConnectionDragStartedEventArgs : ConnectionDragEventArgs
	{
		/// <summary>
		/// The connection that will be dragged out.
		/// </summary>
		public object Connection
		{
			get
			{
				return connection;
			}
			set
			{
				connection = value;
			}
		}

		#region Private Methods

		internal ConnectionDragStartedEventArgs(RoutedEvent routedEvent, object source, object node, object connector) :
			base(routedEvent, source, node, null, connector)
		{
		}

		#endregion Private Methods
	}

	/// <summary>
	/// Defines the event handler for the ConnectionDragStarted event.
	/// </summary>
	public delegate void ConnectionDragStartedEventHandler(object sender, ConnectionDragStartedEventArgs e);

	/// <summary>
	/// Arguments for event raised while user is dragging a node in the network.
	/// </summary>
	public class QueryConnectionFeedbackEventArgs : ConnectionDragEventArgs
	{
		#region Private Data Members

		/// <summary>
		/// The ConnectorItem or it's DataContext (when non-NULL).
		/// </summary>
		private object draggedOverConnector = null;

		/// <summary>
		/// Set to 'true' / 'false' to indicate that the connection from the dragged out connection to the dragged over connector is valid.
		/// </summary>
		private bool connectionOk = true;

		/// <summary>
		/// The indicator to display.
		/// </summary>
		private object feedbackIndicator = null;

		#endregion Private Data Members

		/// <summary>
		/// The ConnectorItem or it's DataContext (when non-NULL).
		/// </summary>
		public object DraggedOverConnector
		{
			get
			{
				return draggedOverConnector;
			}
		}

		/// <summary>
		/// The connection that will be dragged out.
		/// </summary>
		public object Connection
		{
			get
			{
				return connection;
			}
		}

		/// <summary>
		/// Set to 'true' / 'false' to indicate that the connection from the dragged out connection to the dragged over connector is valid.
		/// </summary>
		public bool ConnectionOk
		{
			get
			{
				return connectionOk;
			}
			set
			{
				connectionOk = value;
			}
		}

		/// <summary>
		/// The indicator to display.
		/// </summary>
		public object FeedbackIndicator
		{
			get
			{
				return feedbackIndicator;
			}
			set
			{
				feedbackIndicator = value;
			}
		}

		#region Private Methods

		internal QueryConnectionFeedbackEventArgs(RoutedEvent routedEvent, object source, 
			object node, object connection, object connector, object draggedOverConnector) :
			base(routedEvent, source, node, connection, connector)
		{
			this.draggedOverConnector = draggedOverConnector;
		}

		#endregion Private Methods
	}

	/// <summary>
	/// Defines the event handler for the QueryConnectionFeedback event.
	/// </summary>
	public delegate void QueryConnectionFeedbackEventHandler(object sender, QueryConnectionFeedbackEventArgs e);

	/// <summary>
	/// Arguments for event raised while user is dragging a node in the network.
	/// </summary>
	public class ConnectionDraggingEventArgs : ConnectionDragEventArgs
	{
		/// <summary>
		/// The connection being dragged out.
		/// </summary>
		public object Connection
		{
			get
			{
				return connection;
			}
		}

		#region Private Methods

		internal ConnectionDraggingEventArgs(RoutedEvent routedEvent, object source,
				object node, object connection, object connector) :
			base(routedEvent, source, node, connection, connector)
		{
		}

		#endregion Private Methods
	}

	/// <summary>
	/// Defines the event handler for the ConnectionDragging event.
	/// </summary>
	public delegate void ConnectionDraggingEventHandler(object sender, ConnectionDraggingEventArgs e);

	/// <summary>
	/// Arguments for event raised when the user has completed dragging a connector.
	/// </summary>
	public class ConnectionDragCompletedEventArgs : ConnectionDragEventArgs
	{
		#region Private Data Members

		/// <summary>
		/// The ConnectorItem or it's DataContext (when non-NULL).
		/// </summary>
		private object connectorDraggedOver = null;

		#endregion Private Data Members

		/// <summary>
		/// The ConnectorItem or it's DataContext (when non-NULL).
		/// </summary>
		public object ConnectorDraggedOver
		{
			get
			{
				return connectorDraggedOver;
			}
		}

		/// <summary>
		/// The connection that will be dragged out.
		/// </summary>
		public object Connection
		{
			get
			{
				return connection;
			}
		}

		#region Private Methods

		internal ConnectionDragCompletedEventArgs(RoutedEvent routedEvent, object source, object node, object connection, object connector, object connectorDraggedOver) :
			base(routedEvent, source, node, connection, connector)
		{
			this.connectorDraggedOver = connectorDraggedOver;
		}

		#endregion Private Methods
	}

	/// <summary>
	/// Defines the event handler for the ConnectionDragCompleted event.
	/// </summary>
	public delegate void ConnectionDragCompletedEventHandler(object sender, ConnectionDragCompletedEventArgs e);
}
