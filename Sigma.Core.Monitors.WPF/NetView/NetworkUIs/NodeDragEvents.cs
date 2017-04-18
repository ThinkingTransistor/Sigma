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

using System.Collections;
using System.Windows;

namespace Sigma.Core.Monitors.WPF.NetView.NetworkUIs
{
    /// <summary>
    /// Base class for node dragging event args.
    /// </summary>
    public class NodeDragEventArgs : RoutedEventArgs
    {
        /// <summary>
        /// The NodeItem's or their DataContext (when non-NULL).
        /// </summary>
        public ICollection nodes = null;

        protected NodeDragEventArgs(RoutedEvent routedEvent, object source, ICollection nodes) :
            base(routedEvent, source)
        {
            this.nodes = nodes;
        }

        /// <summary>
        /// The NodeItem's or their DataContext (when non-NULL).
        /// </summary>
        public ICollection Nodes
        {
            get
            {
                return nodes;
            }
        }
    }

    /// <summary>
    /// Defines the event handler for NodeDragStarted events.
    /// </summary>
    public delegate void NodeDragEventHandler(object sender, NodeDragEventArgs e);

    /// <summary>
    /// Arguments for event raised when the user starts to drag a node in the network.
    /// </summary>
    public class NodeDragStartedEventArgs : NodeDragEventArgs
    {
        /// <summary>
        /// Set to 'false' to disallow dragging.
        /// </summary>
        private bool cancel = false;

        internal NodeDragStartedEventArgs(RoutedEvent routedEvent, object source, ICollection nodes) :
            base(routedEvent, source, nodes)
        {
        }

        /// <summary>
        /// Set to 'false' to disallow dragging.
        /// </summary>
        public bool Cancel
        {
            get
            {
                return cancel;
            }
            set
            {
                cancel = value;
            }
        }
    }

    /// <summary>
    /// Defines the event handler for NodeDragStarted events.
    /// </summary>
    public delegate void NodeDragStartedEventHandler(object sender, NodeDragStartedEventArgs e);

    /// <summary>
    /// Arguments for event raised while user is dragging a node in the network.
    /// </summary>
    public class NodeDraggingEventArgs : NodeDragEventArgs
    {
        /// <summary>
        /// The amount the node has been dragged horizontally.
        /// </summary>
        public double horizontalChange = 0;

        /// <summary>
        /// The amount the node has been dragged vertically.
        /// </summary>
        public double verticalChange = 0;

        internal NodeDraggingEventArgs(RoutedEvent routedEvent, object source, ICollection nodes, double horizontalChange, double verticalChange) :
            base(routedEvent, source, nodes)
        {
            this.horizontalChange = horizontalChange;
            this.verticalChange = verticalChange;
        }

        /// <summary>
        /// The amount the node has been dragged horizontally.
        /// </summary>
        public double HorizontalChange
        {
            get
            {
                return horizontalChange;
            }
        }

        /// <summary>
        /// The amount the node has been dragged vertically.
        /// </summary>
        public double VerticalChange
        {
            get
            {
                return verticalChange;
            }
        }
    }

    /// <summary>
    /// Defines the event handler for NodeDragStarted events.
    /// </summary>
    public delegate void NodeDraggingEventHandler(object sender, NodeDraggingEventArgs e);

    /// <summary>
    /// Arguments for event raised when the user has completed dragging a node in the network.
    /// </summary>
    public class NodeDragCompletedEventArgs : NodeDragEventArgs
    {
        public NodeDragCompletedEventArgs(RoutedEvent routedEvent, object source, ICollection nodes) :
            base(routedEvent, source, nodes)
        {
        }
    }

    /// <summary>
    /// Defines the event handler for NodeDragCompleted events.
    /// </summary>
    public delegate void NodeDragCompletedEventHandler(object sender, NodeDragCompletedEventArgs e);
}
