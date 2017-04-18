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
    /// This is the UI element for a connector.
    /// Each nodes has multiple connectors that are used to connect it to other nodes.
    /// </summary>
    public class ConnectorItem : ContentControl
    {
        #region Dependency Property/Event Definitions

        public static readonly DependencyProperty HotspotProperty =
            DependencyProperty.Register("Hotspot", typeof(Point), typeof(ConnectorItem));

        internal static readonly DependencyProperty ParentNetworkViewProperty =
            DependencyProperty.Register("ParentNetworkView", typeof(NetworkView), typeof(ConnectorItem),
                new FrameworkPropertyMetadata(ParentNetworkView_PropertyChanged));

        internal static readonly DependencyProperty ParentNodeItemProperty =
            DependencyProperty.Register("ParentNodeItem", typeof(NodeItem), typeof(ConnectorItem));

        internal static readonly RoutedEvent ConnectorDragStartedEvent =
            EventManager.RegisterRoutedEvent("ConnectorDragStarted", RoutingStrategy.Bubble, typeof(ConnectorItemDragStartedEventHandler), typeof(ConnectorItem));

        internal static readonly RoutedEvent ConnectorDraggingEvent =
            EventManager.RegisterRoutedEvent("ConnectorDragging", RoutingStrategy.Bubble, typeof(ConnectorItemDraggingEventHandler), typeof(ConnectorItem));

        internal static readonly RoutedEvent ConnectorDragCompletedEvent =
            EventManager.RegisterRoutedEvent("ConnectorDragCompleted", RoutingStrategy.Bubble, typeof(ConnectorItemDragCompletedEventHandler), typeof(ConnectorItem));

        #endregion Dependency Property/Event Definitions

        #region Private Data Members

        /// <summary>
        /// The point the mouse was last at when dragging.
        /// </summary>
        private Point lastMousePoint;

        /// <summary>
        /// Set to 'true' when left mouse button is held down.
        /// </summary>
        private bool isLeftMouseDown = false;

        /// <summary>
        /// Set to 'true' when the user is dragging the connector.
        /// </summary>
        private bool isDragging = false;

        /// <summary>
        /// The threshold distance the mouse-cursor must move before dragging begins.
        /// </summary>
        private static readonly double DragThreshold = 2;

        #endregion Private Data Members

        public ConnectorItem()
        {
            //
            // By default, we don't want a connector to be focusable.
            //
            Focusable = false;

            //
            // Hook layout update to recompute 'Hotspot' when the layout changes.
            //
            LayoutUpdated += new EventHandler(ConnectorItem_LayoutUpdated);
        }

        /// <summary>
        /// Automatically updated dependency property that specifies the hotspot (or center point) of the connector.
        /// Specified in content coordinate.
        /// </summary>
        public Point Hotspot
        {
            get
            {
                return (Point)GetValue(HotspotProperty);
            }
            set
            {
                SetValue(HotspotProperty, value);
            }
        }

        #region Private Data Members\Properties

        /// <summary>
        /// Reference to the data-bound parent NetworkView.
        /// </summary>
        internal NetworkView ParentNetworkView
        {
            get
            {
                return (NetworkView)GetValue(ParentNetworkViewProperty);
            }
            set
            {
                SetValue(ParentNetworkViewProperty, value);
            }
        }

       
        /// <summary>
        /// Reference to the data-bound parent NodeItem.
        /// </summary>
        internal NodeItem ParentNodeItem
        {
            get
            {
                return (NodeItem)GetValue(ParentNodeItemProperty);
            }
            set
            {
                SetValue(ParentNodeItemProperty, value);
            }
        }

        #endregion Private Data Members\Properties

        #region Private Methods

        /// <summary>
        /// Static constructor.
        /// </summary>
        static ConnectorItem()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(ConnectorItem), new FrameworkPropertyMetadata(typeof(ConnectorItem)));
        }

        /// <summary>
        /// A mouse button has been held down.
        /// </summary>
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);

            if (ParentNodeItem != null)
            {
                ParentNodeItem.BringToFront();
            }

            if (ParentNetworkView != null)
            {
                ParentNetworkView.Focus();
            }

            if (e.ChangedButton == MouseButton.Left)
            {
                if (ParentNodeItem != null)
                {
                    //
                    // Delegate to parent node to execute selection logic.
                    //
                    ParentNodeItem.LeftMouseDownSelectionLogic();
                }

                lastMousePoint = e.GetPosition(ParentNetworkView);
                isLeftMouseDown = true;
                e.Handled = true;
            }
            else if (e.ChangedButton == MouseButton.Right)
            {
                if (ParentNodeItem != null)
                {
                    //
                    // Delegate to parent node to execute selection logic.
                    //
                    ParentNodeItem.RightMouseDownSelectionLogic();
                }
            }
        }

        /// <summary>
        /// The mouse cursor has been moved.
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
                Vector offset = curMousePoint - lastMousePoint;
                if (offset.X != 0.0 &&
                    offset.Y != 0.0)
                {
                    lastMousePoint = curMousePoint;

                    RaiseEvent(new ConnectorItemDraggingEventArgs(ConnectorDraggingEvent, this, offset.X, offset.Y));
                }

                e.Handled = true;
            }
            else if (isLeftMouseDown)
            {
                if (ParentNetworkView != null &&
                    ParentNetworkView.EnableConnectionDragging)
                {
                    //
                    // The user is left-dragging the connector and connection dragging is enabled,
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
                        var eventArgs = new ConnectorItemDragStartedEventArgs(ConnectorDragStartedEvent, this);
                        RaiseEvent(eventArgs);

                        if (eventArgs.Cancel)
                        {
                            //
                            // Handler of the event disallowed dragging of the node.
                            //
                            isLeftMouseDown = false;
                            return;
                        }

                        isDragging = true;
                        CaptureMouse();
                        e.Handled = true;
                    }
                }
            }
        }

        /// <summary>
        /// A mouse button has been released.
        /// </summary>
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            base.OnMouseUp(e);

            if (e.ChangedButton == MouseButton.Left)
            {
                if (isLeftMouseDown)
                {
                    if (isDragging)
                    {
                        RaiseEvent(new ConnectorItemDragCompletedEventArgs(ConnectorDragCompletedEvent, this));
                        
                        ReleaseMouseCapture();

                        isDragging = false;
                    }
                    else
                    {
                        //
                        // Execute mouse up selection logic only if there was no drag operation.
                        //
                        if (ParentNodeItem != null)
                        {
                            //
                            // Delegate to parent node to execute selection logic.
                            //
                            ParentNodeItem.LeftMouseUpSelectionLogic();
                        }
                    }

                    isLeftMouseDown = false;

                    e.Handled = true;
                }
            }
        }

        /// <summary>
        /// Cancel connection dragging for the connector that was dragged out.
        /// </summary>
        internal void CancelConnectionDragging()
        {
            if (isLeftMouseDown)
            {
                //
                // Raise ConnectorDragCompleted, with a null connector.
                //
                RaiseEvent(new ConnectorItemDragCompletedEventArgs(ConnectorDragCompletedEvent, null));

                isLeftMouseDown = false;
                ReleaseMouseCapture();
            }
        }

        /// <summary>
        /// Event raised when 'ParentNetworkView' property has changed.
        /// </summary>
        private static void ParentNetworkView_PropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            ConnectorItem c = (ConnectorItem)d;
            c.UpdateHotspot();
        }

        /// <summary>
        /// Event raised when the layout of the connector has been updated.
        /// </summary>
        private void ConnectorItem_LayoutUpdated(object sender, EventArgs e)
        {
            UpdateHotspot();
        }

        /// <summary>
        /// Update the connector hotspot.
        /// </summary>
        private void UpdateHotspot()
        {
            if (ParentNetworkView == null)
            {
                // No parent NetworkView is set.
                return;
            }

            if (!ParentNetworkView.IsAncestorOf(this))
            {
                //
                // The parent NetworkView is no longer an ancestor of the connector.
                // This happens when the connector (and its parent node) has been removed from the network.
                // Reset the property null so we don't attempt to check again.
                //
                ParentNetworkView = null;
                return;
            }

            //
            // The parent NetworkView is still valid.
            // Compute the center point of the connector.
            //
            var centerPoint = new Point(ActualWidth / 2, ActualHeight / 2);

            //
            // Transform the center point so that it is relative to the parent NetworkView.
            // Then assign it to Hotspot.  Usually Hotspot will be data-bound to the application
            // view-model using OneWayToSource so that the value of the hotspot is then pushed through
            // to the view-model.
            //
            Hotspot = TransformToAncestor(ParentNetworkView).Transform(centerPoint);
       }

        #endregion Private Methods
    }
}
