using System;
using System.Collections.Generic;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Monitors.WPF.NetView;
using Sigma.Core.Monitors.WPF.NetView.Graphing;
using Sigma.Core.Monitors.WPF.NetView.NetworkModel;

namespace Sigma.Core.Monitors.WPF.Panels.Graphing
{
	/// <summary>
	/// This panel is capable of illustring a graph of nodes. 
	/// </summary>
	public class GraphPanel : GenericPanel<NetLayout>, IGraphVisualiser
	{
		private ILog _logger = LogManager.GetLogger(typeof(GraphPanel));

		/// <summary>
		/// Get the graph structure that is being represented.
		/// </summary>
		public IGraphStructure GraphStructure { get; private set; }

		private Dictionary<GraphNode, NodeViewModel> GraphViewMapping;

		/// <summary>
		///     Create a GraphPanel with a given title and a set of nodes.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="graphStructure">The graph structure that is visualised.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public GraphPanel(string title, IGraphStructure graphStructure, object headerContent = null) : base(title, headerContent)
		{
			Init(graphStructure);
		}

		public GraphPanel(string title, INetworkArchitecture networkArchitecture, object headerContent = null) : base(title, headerContent)
		{
			GraphStructure structure = null;
			GraphNode prevNode = new GraphNode();
			LayerConstruct prevLayerConstruct = null;
			networkArchitecture.ResolveAllNames();
			foreach (LayerConstruct layerConstruct in networkArchitecture.YieldLayerConstructsOrdered())
			{
				GraphNode node = new GraphNode(layerConstruct.Name);
				if (structure == null)
				{
					structure = new GraphStructure(node);
				}
				else
				{
					//string outStr = "out";
					//string inStr = "in";
					int input = -1;
					int output = -1;
					if (layerConstruct.Parameters.ContainsKey("size"))
					{
						output = (int)layerConstruct.Parameters["size"];
					}
					if (prevLayerConstruct.Parameters.ContainsKey("size"))
					{
						input = (int)prevLayerConstruct.Parameters["size"];
					}
					if (output == -1) output = input;
					else if (input == -1) input = output;

					structure.AddNode(prevNode, $"out ({output})", node, $"in ({input})");
				}

				prevNode = node;
				prevLayerConstruct = layerConstruct;
			}

			Init(structure);
		}

		private void Init(IGraphStructure graphStructure)
		{
			GraphStructure = graphStructure;
			Content = new NetLayout();
			PopulateNetLayout(Content, graphStructure);
		}

		/// <summary>
		/// Fill a given layout with a given graph structure.
		/// </summary>
		/// <param name="layout">The layout that will contain the definition for the nodes.</param>
		/// <param name="structure">The structure of the nodes. </param>
		protected virtual void PopulateNetLayout(NetLayout layout, IGraphStructure structure)
		{
			_logger.Debug($"Populating netlayout with a graph (root node: {structure.Root})");

			GraphViewMapping = new Dictionary<GraphNode, NodeViewModel>();

			PopulateForward(structure.Root);

			GraphViewMapping = null;

			//NodeViewModel node1 = viewModel.CreateNode("Test node", 100, 100, false);
			//NodeViewModel node2 = viewModel.CreateNode("Test node", 400, 100, false);

			//viewModel.Connect(node1, node2, 0, 0);

			_logger.Debug($"Finished populating netlayout with a graph (root node: {structure.Root})");
		}

		private int tmp = 30;

		protected virtual NodeViewModel PopulateForward(GraphNode node)
		{
			NodeViewModel root = new NodeViewModel(node.Name);
			root.X = tmp;
			root.Y = 450;
			tmp += 200;

			GraphViewMapping.Add(node, root);

			foreach (GraphConnection connection in node.Connections)
			{
				// if its an outgoing connection
				if (connection.SourceNode == node)
				{
					// if it is not circular
					if (connection.DestinationNode != node)
					{
						NodeViewModel next;
						if (!GraphViewMapping.TryGetValue(connection.DestinationNode, out next))
						{
							next = PopulateForward(connection.DestinationNode);
						}

						ConnectorViewModel rootOut = new ConnectorViewModel(connection.SourceName);
						root.OutputConnectors.Add(rootOut);
						ConnectorViewModel nextIn = new ConnectorViewModel(connection.DestinationName);
						next.InputConnectors.Add(nextIn);
						Content.ViewModel.Connect(root, next, rootOut, nextIn);
					}
					else
					{
						throw new NotImplementedException();
					}
				}
			}

			Content.ViewModel.Network.Nodes.Add(root);

			return root;
		}
	}
}