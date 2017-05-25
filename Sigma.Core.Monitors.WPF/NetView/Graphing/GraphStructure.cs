namespace Sigma.Core.Monitors.WPF.NetView.Graphing
{
	/// <summary>
	/// A structure that represents a graph with a single entry point.
	/// </summary>
	public interface IGraphStructure
	{
		/// <summary>
		/// The root node of this graph.
		/// </summary>
		GraphNode Root { get; }

		/// <summary>
		/// Add a single node (destination) to the graph with the correct connection. 
		/// </summary>
		/// <param name="source">The source node - it will be used to correctly set the connection.</param>
		/// <param name="sourceName">The name of the source socket.</param>
		/// <param name="destination">This node will be added with a correct connection.</param>
		/// <param name="destinationName">The name of the destination socket.</param>
		bool AddNode(GraphNode source, string sourceName, GraphNode destination, string destinationName);
	}

	/// <summary>
	/// The default implementation of the graph structure.
	/// </summary>
	public class GraphStructure : IGraphStructure
	{
		/// <inheritdoc/>
		public GraphNode Root { get; }

		/// <summary>
		/// Create a new graph structure with a given source node. 
		/// </summary>
		/// <param name="root">The root node of the graph.</param>
		public GraphStructure(GraphNode root)
		{
			Root = root;
		}

		/// <inheritdoc />
		public bool AddNode(GraphNode source, string sourceName, GraphNode destination, string destinationName)
		{
			GraphConnection connection = new GraphConnection(source, sourceName, destination, destinationName);

			return source.AddConnection(connection) && destination.AddConnection(connection);
		}
	}
}