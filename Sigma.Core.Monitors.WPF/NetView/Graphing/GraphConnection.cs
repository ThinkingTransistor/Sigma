using System;

namespace Sigma.Core.Monitors.WPF.NetView.Graphing
{
	/// <summary>
	/// A connection between two graph nodes. 
	/// </summary>
	public struct GraphConnection : IEquatable<GraphConnection>
	{
		/// <summary>
		/// The name of this connection. 
		/// </summary>
		public readonly string SourceName, DestinationName;

		/// <summary>
		/// The source node of this connection.
		/// </summary>
		public readonly GraphNode SourceNode;

		/// <summary>
		/// The destination node of this connection.
		/// </summary>
		public readonly GraphNode DestinationNode;

		/// <summary>
		/// Create a new graph connection between two nodes. It will not modify the connection list of source and destination.
		/// </summary>
		/// <param name="source">The source node.</param>
		/// <param name="sourceName">The name of the source socket.</param>
		/// <param name="destination">The destination node. </param>
		/// <param name="destinationName">The name of the destination socket.</param>
		public GraphConnection(GraphNode source, string sourceName, GraphNode destination, string destinationName)
		{
			SourceNode = source;
			SourceName = sourceName;
			DestinationNode = destination;
			DestinationName = destinationName;
		}

		/// <inheritdoc />
		public bool Equals(GraphConnection other)
		{
			return string.Equals(SourceName, other.SourceName) && string.Equals(DestinationName, other.DestinationName) && SourceNode.Equals(other.SourceNode) && DestinationNode.Equals(other.DestinationNode);
		}

		/// <inheritdoc />
		public override bool Equals(object obj)
		{
			if (ReferenceEquals(null, obj)) return false;
			return obj is GraphConnection && Equals((GraphConnection) obj);
		}

		/// <inheritdoc />
		public override int GetHashCode()
		{
			unchecked
			{
				int hashCode = (SourceName != null ? SourceName.GetHashCode() : 0);
				hashCode = (hashCode * 397) ^ (DestinationName != null ? DestinationName.GetHashCode() : 0);
				hashCode = (hashCode * 397) ^ SourceNode.GetHashCode();
				hashCode = (hashCode * 397) ^ DestinationNode.GetHashCode();
				return hashCode;
			}
		}

		/// <summary>Returns a value that indicates whether the values of two <see cref="T:Sigma.Core.Monitors.WPF.NetView.Graphing.GraphConnection" /> objects are equal.</summary>
		/// <param name="left">The first value to compare.</param>
		/// <param name="right">The second value to compare.</param>
		/// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
		public static bool operator ==(GraphConnection left, GraphConnection right)
		{
			return left.Equals(right);
		}

		/// <summary>Returns a value that indicates whether two <see cref="T:Sigma.Core.Monitors.WPF.NetView.Graphing.GraphConnection" /> objects have different values.</summary>
		/// <param name="left">The first value to compare.</param>
		/// <param name="right">The second value to compare.</param>
		/// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
		public static bool operator !=(GraphConnection left, GraphConnection right)
		{
			return !left.Equals(right);
		}
	}
}