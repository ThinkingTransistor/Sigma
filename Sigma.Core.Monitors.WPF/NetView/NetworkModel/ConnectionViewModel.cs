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
using System.Windows.Media;
using Sigma.Core.Monitors.WPF.NetView.Utils;

namespace Sigma.Core.Monitors.WPF.NetView.NetworkModel
{
	/// <summary>
	/// Defines a connection between two connectors (aka connection points) of two nodes.
	/// </summary>
	public sealed class ConnectionViewModel : AbstractModelBase
	{
		#region Internal Data Members

		/// <summary>
		/// The source connector the connection is attached to.
		/// </summary>
		private ConnectorViewModel _sourceConnector;

		/// <summary>
		/// The destination connector the connection is attached to.
		/// </summary>
		private ConnectorViewModel _destConnector;

		/// <summary>
		/// The source and dest hotspots used for generating connection points.
		/// </summary>
		private Point _sourceConnectorHotspot;
		private Point _destConnectorHotspot;

		/// <summary>
		/// Points that make up the connection.
		/// </summary>
		private PointCollection _points;

		#endregion Internal Data Members

		/// <summary>
		/// The source connector the connection is attached to.
		/// </summary>
		public ConnectorViewModel SourceConnector
		{
			get
			{
				return _sourceConnector;
			}
			set
			{
				if (_sourceConnector == value)
				{
					return;
				}

				if (_sourceConnector != null)
				{
					_sourceConnector.AttachedConnections.Remove(this);
					_sourceConnector.HotspotUpdated -= sourceConnector_HotspotUpdated;
				}

				_sourceConnector = value;

				if (_sourceConnector != null)
				{
					_sourceConnector.AttachedConnections.Add(this);
					_sourceConnector.HotspotUpdated += sourceConnector_HotspotUpdated;
					SourceConnectorHotspot = _sourceConnector.Hotspot;
				}

				OnPropertyChanged("SourceConnector");
				OnConnectionChanged();
			}
		}

		/// <summary>
		/// The destination connector the connection is attached to.
		/// </summary>
		public ConnectorViewModel DestConnector
		{
			get
			{
				return _destConnector;
			}
			set
			{
				if (_destConnector == value)
				{
					return;
				}

				if (_destConnector != null)
				{
					_destConnector.AttachedConnections.Remove(this);
					_destConnector.HotspotUpdated -= destConnector_HotspotUpdated;
				}

				_destConnector = value;

				if (_destConnector != null)
				{
					_destConnector.AttachedConnections.Add(this);
					_destConnector.HotspotUpdated += destConnector_HotspotUpdated;
					DestConnectorHotspot = _destConnector.Hotspot;
				}

				OnPropertyChanged("DestConnector");
				OnConnectionChanged();
			}
		}

		/// <summary>
		/// The source and dest hotspots used for generating connection points.
		/// </summary>
		public Point SourceConnectorHotspot
		{
			get
			{
				return _sourceConnectorHotspot;
			}
			set
			{
				_sourceConnectorHotspot = value;

				ComputeConnectionPoints();

				OnPropertyChanged("SourceConnectorHotspot");
			}
		}

		public Point DestConnectorHotspot
		{
			get
			{
				return _destConnectorHotspot;
			}
			set
			{
				_destConnectorHotspot = value;

				ComputeConnectionPoints();

				OnPropertyChanged("DestConnectorHotspot");
			}
		}

		/// <summary>
		/// Points that make up the connection.
		/// </summary>
		public PointCollection Points
		{
			get
			{
				return _points;
			}
			set
			{
				_points = value;

				OnPropertyChanged("Points");
			}
		}

		/// <summary>
		/// Event fired when the connection has changed.
		/// </summary>
		public event EventHandler<EventArgs> ConnectionChanged;

		#region Private Methods

		/// <summary>
		/// Raises the 'ConnectionChanged' event.
		/// </summary>
		private void OnConnectionChanged()
		{
			if (ConnectionChanged != null)
			{
				ConnectionChanged(this, EventArgs.Empty);
			}
		}

		/// <summary>
		/// Event raised when the hotspot of the source connector has been updated.
		/// </summary>
		private void sourceConnector_HotspotUpdated(object sender, EventArgs e)
		{
			SourceConnectorHotspot = SourceConnector.Hotspot;
		}

		/// <summary>
		/// Event raised when the hotspot of the dest connector has been updated.
		/// </summary>
		private void destConnector_HotspotUpdated(object sender, EventArgs e)
		{
			DestConnectorHotspot = DestConnector.Hotspot;
		}

		/// <summary>
		/// Rebuild connection points.
		/// </summary>
		private void ComputeConnectionPoints()
		{
			PointCollection computedPoints = new PointCollection {SourceConnectorHotspot};

			double deltaX = Math.Abs(DestConnectorHotspot.X - SourceConnectorHotspot.X);
			double deltaY = Math.Abs(DestConnectorHotspot.Y - SourceConnectorHotspot.Y);
			if (deltaX > deltaY)
			{
				double midPointX = SourceConnectorHotspot.X + ((DestConnectorHotspot.X - SourceConnectorHotspot.X) / 2);
				computedPoints.Add(new Point(midPointX, SourceConnectorHotspot.Y));
				computedPoints.Add(new Point(midPointX, DestConnectorHotspot.Y));
			}
			else
			{
				double midPointY = SourceConnectorHotspot.Y + ((DestConnectorHotspot.Y - SourceConnectorHotspot.Y) / 2);
				computedPoints.Add(new Point(SourceConnectorHotspot.X, midPointY));
				computedPoints.Add(new Point(DestConnectorHotspot.X, midPointY));
			}

			computedPoints.Add(DestConnectorHotspot);
			computedPoints.Freeze();

			Points = computedPoints;
		}

		#endregion Private Methods
	}
}
