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

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace Sigma.Core.Monitors.WPF.NetView.NetworkUIs
{
	/// <summary>
	/// This is a UI element that represents a network/flow-chart node.
	/// </summary>
	public class NodeItem : ListBoxItem
	{
		#region Dependency Property/Event Definitions

		public static readonly DependencyProperty XProperty =
			DependencyProperty.Register("X", typeof(double), typeof(NodeItem),
				new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));
		public static readonly DependencyProperty YProperty =
			DependencyProperty.Register("Y", typeof(double), typeof(NodeItem),
				new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

		public static readonly DependencyProperty ZIndexProperty =
			DependencyProperty.Register("ZIndex", typeof(int), typeof(NodeItem),
				new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

		internal static readonly DependencyProperty ParentNetworkViewProperty =
			DependencyProperty.Register("ParentNetworkView", typeof(Sigma.Core.Monitors.WPF.NetView.NetworkUIs.NetworkView), typeof(NodeItem), 
				new FrameworkPropertyMetadata(ParentNetworkView_PropertyChanged));

		internal static readonly RoutedEvent NodeDragStartedEvent =
			EventManager.RegisterRoutedEvent("NodeDragStarted", RoutingStrategy.Bubble, typeof(NodeDragStartedEventHandler), typeof(NodeItem));

		internal static readonly RoutedEvent NodeDraggingEvent =
			EventManager.RegisterRoutedEvent("NodeDragging", RoutingStrategy.Bubble, typeof(NodeDraggingEventHandler), typeof(NodeItem));

		internal static readonly RoutedEvent NodeDragCompletedEvent =
			EventManager.RegisterRoutedEvent("NodeDragCompleted", RoutingStrategy.Bubble, typeof(NodeDragCompletedEventHandler), typeof(NodeItem));

		#endregion Dependency Property/Event Definitions

		public NodeItem()
		{
			//
			// By default, we don't want this UI element to be focusable.
			//
			Focusable = false;
		}

		/// <summary>
		/// The X coordinate of the node.
		/// </summary>
		public double X
		{
			get
			{
				return (double)GetValue(XProperty);
			}
			set
			{
				SetValue(XProperty, value);
			}
		}

		/// <summary>
		/// The Y coordinate of the node.
		/// </summary>
		public double Y
		{
			get
			{
				return (double)GetValue(YProperty);
			}
			set
			{
				SetValue(YProperty, value);
			}
		}

		/// <summary>
		/// The Z index of the node.
		/// </summary>
		public int ZIndex
		{
			get
			{
				return (int)GetValue(ZIndexProperty);
			}
			set
			{
				SetValue(ZIndexProperty, value);
			}
	   }            

		#region Private Data Members\Properties

		/// <summary>
		/// Reference to the data-bound parent NetworkView.
		/// </summary>
		internal Sigma.Core.Monitors.WPF.NetView.NetworkUIs.NetworkView ParentNetworkView
		{
			get
			{
				return (Sigma.Core.Monitors.WPF.NetView.NetworkUIs.NetworkView)GetValue(ParentNetworkViewProperty);
			}
			set
			{
				SetValue(ParentNetworkViewProperty, value);
			}
		}

		/// <summary>
		/// The point the mouse was last at when dragging.
		/// </summary>
		private Point lastMousePoint;

		/// <summary>
		/// Set to 'true' when left mouse button is held down.
		/// </summary>
		private bool isLeftMouseDown = false;

		/// <summary>
		/// Set to 'true' when left mouse button and the control key are held down.
		/// </summary>
		private bool isLeftMouseAndControlDown = false;

		/// <summary>
		/// Set to 'true' when dragging has started.
		/// </summary>
		private bool isDragging = false;

		/// <summary>
		/// The threshold distance the mouse-cursor must move before dragging begins.
		/// </summary>
		private static readonly double DragThreshold = 5;

		#endregion Private Data Members\Properties

		#region Private Methods

		/// <summary>
		/// Static constructor.
		/// </summary>
		static NodeItem()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(NodeItem), new FrameworkPropertyMetadata(typeof(NodeItem)));
		}

		/// <summary>
		/// Bring the node to the front of other elements.
		/// </summary>
		internal void BringToFront()
		{
			if (ParentNetworkView == null)
			{
				return;
			}

			int maxZ = ParentNetworkView.FindMaxZIndex();
			ZIndex = maxZ + 1;
		}

		/// <summary>
		/// Called when a mouse button is held down.
		/// </summary>
		protected override void OnMouseDown(MouseButtonEventArgs e)
		{
			base.OnMouseDown(e);

			BringToFront();

			if (ParentNetworkView != null)
			{
				ParentNetworkView.Focus();
			}

			if (e.ChangedButton == MouseButton.Left && ParentNetworkView != null)
			{
				lastMousePoint = e.GetPosition(ParentNetworkView);
				isLeftMouseDown = true;

				LeftMouseDownSelectionLogic();

				e.Handled = true;
			}
			else if (e.ChangedButton == MouseButton.Right && ParentNetworkView != null)
			{
				RightMouseDownSelectionLogic();
		   }
		}

		/// <summary>
		/// This method contains selection logic that is invoked when the left mouse button is pressed down.
		/// The reason this exists in its own method rather than being included in OnMouseDown is 
		/// so that ConnectorItem can reuse this logic from its OnMouseDown.
		/// </summary>
		internal void LeftMouseDownSelectionLogic()
		{
			if ((Keyboard.Modifiers & ModifierKeys.Control) != 0)
			{
				//
				// Control key was held down.
				// This means that the rectangle is being added to or removed from the existing selection.
				// Don't do anything yet, we will act on this later in the MouseUp event handler.
				//
				isLeftMouseAndControlDown = true;
			}
			else
			{
				//
				// Control key is not held down.
				//
				isLeftMouseAndControlDown = false;

				if (ParentNetworkView.SelectedNodes.Count == 0)
				{
					//
					// Nothing already selected, select the item.
					//
					IsSelected = true;
				}
				else if (ParentNetworkView.SelectedNodes.Contains(this) ||
						 ParentNetworkView.SelectedNodes.Contains(DataContext))
				{
					// 
					// Item is already selected, do nothing.
					// We will act on this in the MouseUp if there was no drag operation.
					//
				}
				else
				{
					//
					// Item is not selected.
					// Deselect all, and select the item.
					//
					ParentNetworkView.SelectedNodes.Clear();
					IsSelected = true;
				}
			}
		}

		/// <summary>
		/// This method contains selection logic that is invoked when the right mouse button is pressed down.
		/// The reason this exists in its own method rather than being included in OnMouseDown is 
		/// so that ConnectorItem can reuse this logic from its OnMouseDown.
		/// </summary>
		internal void RightMouseDownSelectionLogic()
		{
			if (ParentNetworkView.SelectedNodes.Count == 0)
			{
				//
				// Nothing already selected, select the item.
				//
				IsSelected = true;
			}
			else if (ParentNetworkView.SelectedNodes.Contains(this) ||
					 ParentNetworkView.SelectedNodes.Contains(DataContext))
			{
				// 
				// Item is already selected, do nothing.
				//
			}
			else
			{
				//
				// Item is not selected.
				// Deselect all, and select the item.
				//
				ParentNetworkView.SelectedNodes.Clear();
				IsSelected = true;
			}
		}

		/// <summary>
		/// Called when the mouse cursor is moved.
		/// </summary>
		protected override void OnMouseMove(MouseEventArgs e)
		{
			base.OnMouseMove(e);

			if (isDragging)
			{
				//
				// Raise the event to notify that dragging is in progress.
				//

				Point curMousePoint = e.GetPosition(ParentNetworkView);

				object item = this;
				if (DataContext != null)
				{
					item = DataContext;
				}

				Vector offset = curMousePoint - lastMousePoint;
				if (offset.X != 0.0 ||
					offset.Y != 0.0)
				{
					lastMousePoint = curMousePoint;

					RaiseEvent(new NodeDraggingEventArgs(NodeDraggingEvent, this, new object[] { item }, offset.X, offset.Y));
				}
			}
			else if (isLeftMouseDown && ParentNetworkView.EnableNodeDragging)
			{
				//
				// The user is left-dragging the node,
				// but don't initiate the drag operation until 
				// the mouse cursor has moved more than the threshold distance.
				//
				Point curMousePoint = e.GetPosition(ParentNetworkView);
				var dragDelta = curMousePoint - lastMousePoint;
				double dragDistance = Math.Abs(dragDelta.Length);
				if (dragDistance > DragThreshold)
				{
					//
					// When the mouse has been dragged more than the threshold value commence dragging the node.
					//

					//
					// Raise an event to notify that that dragging has commenced.
					//
					NodeDragStartedEventArgs eventArgs = new NodeDragStartedEventArgs(NodeDragStartedEvent, this, new NodeItem[] { this });
					RaiseEvent(eventArgs);

					if (eventArgs.Cancel)
					{
						//
						// Handler of the event disallowed dragging of the node.
						//
						isLeftMouseDown = false;
						isLeftMouseAndControlDown = false;
						return;
					}

					isDragging = true;
					CaptureMouse();
					e.Handled = true;
				}
			}
		}

		/// <summary>
		/// Called when a mouse button is released.
		/// </summary>
		protected override void OnMouseUp(MouseButtonEventArgs e)
		{
			base.OnMouseUp(e);

			if (isLeftMouseDown)
			{
				if (isDragging)
				{
					//
					// Raise an event to notify that node dragging has finished.
					//

					RaiseEvent(new NodeDragCompletedEventArgs(NodeDragCompletedEvent, this, new NodeItem[] { this }));

					ReleaseMouseCapture();

					isDragging = false;
				}
				else
				{
					//
					// Execute mouse up selection logic only if there was no drag operation.
					//

					LeftMouseUpSelectionLogic();
				}

				isLeftMouseDown = false;
				isLeftMouseAndControlDown = false;

				e.Handled = true;
			}
		}

		/// <summary>
		/// This method contains selection logic that is invoked when the left mouse button is released.
		/// The reason this exists in its own method rather than being included in OnMouseUp is 
		/// so that ConnectorItem can reuse this logic from its OnMouseUp.
		/// </summary>
		internal void LeftMouseUpSelectionLogic()
		{
			if (isLeftMouseAndControlDown)
			{
				//
				// Control key was held down.
				// Toggle the selection.
				//
				IsSelected = !IsSelected;
			}
			else
			{
				//
				// Control key was not held down.
				//
				if (ParentNetworkView.SelectedNodes.Count == 1 &&
					(ParentNetworkView.SelectedNode == this ||
					 ParentNetworkView.SelectedNode == DataContext))
				{
					//
					// The item that was clicked is already the only selected item.
					// Don't need to do anything.
					//
				}
				else
				{
					//
					// Clear the selection and select the clicked item as the only selected item.
					//
					ParentNetworkView.SelectedNodes.Clear();
					IsSelected = true;
				}
			}

			isLeftMouseAndControlDown = false;
		}

		/// <summary>
		/// Event raised when the ParentNetworkView property has changed.
		/// </summary>
		private static void ParentNetworkView_PropertyChanged(DependencyObject o, DependencyPropertyChangedEventArgs e)
		{
			//
			// Bring new nodes to the front of the z-order.
			//
			var nodeItem = (NodeItem) o;
			nodeItem.BringToFront();
		}

		#endregion Private Methods
	}
}
