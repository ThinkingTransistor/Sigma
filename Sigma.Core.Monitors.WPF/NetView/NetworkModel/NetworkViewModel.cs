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
using Sigma.Core.Monitors.WPF.NetView.Utils;

namespace Sigma.Core.Monitors.WPF.NetView.NetworkModel
{
	/// <summary>
	/// Defines a network of nodes and connections between the nodes.
	/// </summary>
	public sealed class NetworkViewModel
	{
		#region Internal Data Members

		/// <summary>
		/// The collection of nodes in the network.
		/// </summary>
		private ImpObservableCollection<NodeViewModel> nodes = null;

		/// <summary>
		/// The collection of connections in the network.
		/// </summary>
		private ImpObservableCollection<ConnectionViewModel> connections = null;

		#endregion Internal Data Members

		/// <summary>
		/// The collection of nodes in the network.
		/// </summary>
		public ImpObservableCollection<NodeViewModel> Nodes
		{
			get
			{
				if (nodes == null)
				{
					nodes = new ImpObservableCollection<NodeViewModel>();
				}

				return nodes;
			}
		}

		/// <summary>
		/// The collection of connections in the network.
		/// </summary>
		public ImpObservableCollection<ConnectionViewModel> Connections
		{
			get
			{
				if (connections == null)
				{
					connections = new ImpObservableCollection<ConnectionViewModel>();
					connections.ItemsRemoved += new EventHandler<CollectionItemsChangedEventArgs>(connections_ItemsRemoved);
				}

				return connections;
			}
		}

		/// <summary>
		/// Connect two nodes together.
		/// </summary>
		/// <param name="a">The first node.</param>
		/// <param name="b">The second node.</param>
		/// <param name="aOut">The output of the first node.</param>
		/// <param name="bIn">The input of the second node. </param>
		public void Connect(NodeViewModel a, NodeViewModel b, ConnectorViewModel aOut, ConnectorViewModel bIn)
		{
			Connections.Add(new ConnectionViewModel
			{
				SourceConnector = aOut,
				DestConnector = bIn
			});
		}

		/// <summary>
		/// Connect two nodes together.
		/// </summary>
		/// <param name="a">The first node.</param>
		/// <param name="b">The second node.</param>
		/// <param name="aOutIndex">The index of the output of the first node.</param>
		/// <param name="bInIndex">The index of the input of the second node.</param>
		public void Connect(NodeViewModel a, NodeViewModel b, int aOutIndex, int bInIndex)
		{
			Connect(a, b, a.OutputConnectors[aOutIndex], b.InputConnectors[bInIndex]);
		}

		#region Private Methods

		/// <summary>
		/// Event raised then Connections have been removed.
		/// </summary>
		private void connections_ItemsRemoved(object sender, CollectionItemsChangedEventArgs e)
		{
			foreach (ConnectionViewModel connection in e.Items)
			{
				connection.SourceConnector = null;
				connection.DestConnector = null;
			}
		}

		#endregion Private Methods
	}
}
