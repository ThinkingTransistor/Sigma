using Sigma.Core.Monitors.WPF.NetView.NetworkModel;
using Sigma.Core.Monitors.WPF.NetView.NetworkUIs;
using System;
using System.Windows;
using System.Windows.Input;

namespace Sigma.Core.Monitors.WPF.NetView
{

	public partial class NetLayout 
	{
		public NetLayout()
		{
			InitializeComponent();
		}

		/// <summary>
		/// Convenient accessor for the view-model.
		/// </summary>
		public NetLayoutViewModel ViewModel => (NetLayoutViewModel)DataContext;

		/// <summary>
		/// Event raised when the user has started to drag out a connection.
		/// </summary>
		private void networkControl_ConnectionDragStarted(object sender, ConnectionDragStartedEventArgs e)
		{
			ConnectorViewModel draggedOutConnector = (ConnectorViewModel)e.ConnectorDraggedOut;
			Point curDragPoint = Mouse.GetPosition(networkControl);

			//
			// Delegate the real work to the view model.
			//
			ConnectionViewModel connection = ViewModel.ConnectionDragStarted(draggedOutConnector, curDragPoint);

			//
			// Must return the view-model object that represents the connection via the event args.
			// This is so that NetworkView can keep track of the object while it is being dragged.
			//
			e.Connection = connection;
		}

		/// <summary>
		/// Event raised, to query for feedback, while the user is dragging a connection.
		/// </summary>
		private void networkControl_QueryConnectionFeedback(object sender, QueryConnectionFeedbackEventArgs e)
		{
			ConnectorViewModel draggedOutConnector = (ConnectorViewModel)e.ConnectorDraggedOut;
			ConnectorViewModel draggedOverConnector = (ConnectorViewModel)e.DraggedOverConnector;

			object feedbackIndicator;
			bool connectionOk;
			ViewModel.QueryConnnectionFeedback(draggedOutConnector, draggedOverConnector, out feedbackIndicator, out connectionOk);

			//
			// Return the feedback object to NetworkView.
			// The object combined with the data-template for it will be used to create a 'feedback icon' to
			// display (in an adorner) to the user.
			//
			e.FeedbackIndicator = feedbackIndicator;

			//
			// Let NetworkView know if the connection is ok or not ok.
			//
			e.ConnectionOk = connectionOk;
		}

		/// <summary>
		/// Event raised while the user is dragging a connection.
		/// </summary>
		private void networkControl_ConnectionDragging(object sender, ConnectionDraggingEventArgs e)
		{
			Point curDragPoint = Mouse.GetPosition(networkControl);
			ConnectionViewModel connection = (ConnectionViewModel)e.Connection;
			ViewModel.ConnectionDragging(curDragPoint, connection);
		}

		/// <summary>
		/// Event raised when the user has finished dragging out a connection.
		/// </summary>
		private void networkControl_ConnectionDragCompleted(object sender, ConnectionDragCompletedEventArgs e)
		{
			ConnectorViewModel connectorDraggedOut = (ConnectorViewModel)e.ConnectorDraggedOut;
			ConnectorViewModel connectorDraggedOver = (ConnectorViewModel)e.ConnectorDraggedOver;
			ConnectionViewModel newConnection = (ConnectionViewModel)e.Connection;
			ViewModel.ConnectionDragCompleted(newConnection, connectorDraggedOut, connectorDraggedOver);
		}

		/// <summary>
		/// Event raised to delete the selected node.
		/// </summary>
		private void DeleteSelectedNodes_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			ViewModel.DeleteSelectedNodes();
		}

		/// <summary>
		/// Event raised to create a new node.
		/// </summary>
		private void CreateNode_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			CreateNode();
		}

		/// <summary>
		/// Event raised to delete a node.
		/// </summary>
		private void DeleteNode_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			NodeViewModel node = (NodeViewModel)e.Parameter;
			ViewModel.DeleteNode(node);
		}

		/// <summary>
		/// Event raised to delete a connection.
		/// </summary>
		private void DeleteConnection_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			ConnectionViewModel connection = (ConnectionViewModel)e.Parameter;
			ViewModel.DeleteConnection(connection);
		}

		/// <summary>
		/// Creates a new node in the network at the current mouse location.
		/// </summary>
		private void CreateNode()
		{
			throw new NotImplementedException();
			//Point newNodePosition = Mouse.GetPosition(networkControl);
			//ViewModel.CreateNode("New Node!", newNodePosition, true);
		}

		/// <summary>
		/// Event raised when the size of a node has changed.
		/// </summary>
		private void Node_SizeChanged(object sender, SizeChangedEventArgs e)
		{
			//
			// The size of a node, as determined in the UI by the node's data-template,
			// has changed.  Push the size of the node through to the view-model.
			//
			FrameworkElement element = (FrameworkElement)sender;
			NodeViewModel node = (NodeViewModel)element.DataContext;
			node.Size = new Size(element.ActualWidth, element.ActualHeight);
		}

		private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{

		}
	}
}