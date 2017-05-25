using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Navigation;

namespace Sigma.Core.Monitors.WPF.NetView.Graphing
{
	/// <summary>
	/// A node in a graph. 
	/// </summary>
	public struct GraphNode : IEquatable<GraphNode>
	{
		/// <summary>
		/// The name of this node. 
		/// </summary>
		public readonly string Name;

		/// <summary>
		/// A list of connections this node has.
		/// </summary>
		private readonly HashSet<GraphConnection> _connections;

		/// <summary>
		/// A list of connections this node has.
		/// </summary>
		public IReadOnlyCollection<GraphConnection> Connections => _connections.ToList();

		/// <summary>
		/// The amount of incoming connection.
		/// </summary>
		public int IncomingConnections { get; private set; }

		/// <summary>
		/// The amount of outgoing connections.
		/// </summary>
		public int OutgoingConnections { get; private set; }

		/// <summary>
		/// Create a new graph node from a name. 
		/// </summary>
		/// <param name="name">The name of the node.</param>
		public GraphNode(string name)
		{
			Name = name;
			_connections = new HashSet<GraphConnection>();
			IncomingConnections = OutgoingConnections = 0;
		}

		/// <summary>
		/// Add a graph connection (does not modify the other part of the connection). 
		/// May not be intended to use directly, but wrapped by a class (e.g. <see cref="IGraphStructure"/>)
		/// </summary>
		/// <param name="connection">The connection between two nodes.</param>
		public bool AddConnection(GraphConnection connection)
		{
			if (_connections.Add(connection))
			{
				// compare both to support circular dependencies.

				if (connection.SourceNode == this) OutgoingConnections++;
				if (connection.DestinationNode == this) IncomingConnections++;

				return true;
			}

			return false;
		}

		public bool Equals(GraphNode other)
		{
			return string.Equals(Name, other.Name) && IncomingConnections == other.IncomingConnections && OutgoingConnections == other.OutgoingConnections;
		}

		public override bool Equals(object obj)
		{
			if (ReferenceEquals(null, obj)) return false;
			return obj is GraphNode && Equals((GraphNode)obj);
		}

		public override int GetHashCode()
		{
			unchecked
			{
				int hashCode = Name.GetHashCode();
				hashCode = (hashCode * 397) ^ _connections.GetHashCode();
				hashCode = (hashCode * 397) ^ IncomingConnections;
				hashCode = (hashCode * 397) ^ OutgoingConnections;
				return hashCode;
			}
		}

		/// <summary>Returns a value that indicates whether the values of two <see cref="T:Sigma.Core.Monitors.WPF.NetView.Graphing.GraphNode" /> objects are equal.</summary>
		/// <param name="left">The first value to compare.</param>
		/// <param name="right">The second value to compare.</param>
		/// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
		public static bool operator ==(GraphNode left, GraphNode right)
		{
			return left.Equals(right);
		}

		/// <summary>Returns a value that indicates whether two <see cref="T:Sigma.Core.Monitors.WPF.NetView.Graphing.GraphNode" /> objects have different values.</summary>
		/// <param name="left">The first value to compare.</param>
		/// <param name="right">The second value to compare.</param>
		/// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
		public static bool operator !=(GraphNode left, GraphNode right)
		{
			return !left.Equals(right);
		}
	}
}