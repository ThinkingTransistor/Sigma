using System;
using System.Collections.Generic;
using System.Linq;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
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

		public GraphPanel(string title, IDataset dataset, object headerContent = null) : base(title, headerContent)
		{
			GraphStructure structure = null;

			ExtractedDataset asExtractedDataset = dataset as ExtractedDataset;

			if (asExtractedDataset != null)
			{
				IRecordExtractor currentExtractor = asExtractedDataset.RecordExtractors.First();
				IRecordExtractor rootExtractor = currentExtractor;
				IList<IRecordExtractor> extractors = new List<IRecordExtractor>();

				do
				{
					extractors.Add(currentExtractor);
				} while ((currentExtractor = currentExtractor.ParentExtractor) != null);

				extractors = extractors.Reverse().ToArray();

				IRecordReader reader = extractors[0].Reader;
				IDataSource source = reader.Source;

				GraphNode resourceNode = new GraphNode($"\"{source.ResourceName}\"");
				GraphNode sourceNode = new GraphNode(source.GetType().Name.ToLower());
				GraphNode prevNode = new GraphNode(reader.GetType().Name.ToLower());

				structure = new GraphStructure(resourceNode);
				structure.AddNode(resourceNode, "resource", sourceNode, "source");
				structure.AddNode(sourceNode, "origin", prevNode, "out (buffer)");

				IRecordExtractor prevExtractor = null;
				foreach (IRecordExtractor extractor in extractors)
				{
					GraphNode node = new GraphNode(extractor.GetType().Name.ToLower());

					string outSectionNames = prevExtractor == null ? "buffer" : $"buffer_{prevExtractor.SectionNames.Length}";
					string inSectionNames = $"buffer_{extractor.SectionNames.Length}";

					structure.AddNode(prevNode, $"out ({outSectionNames})", node, $"in ({inSectionNames})");

					prevNode = node;
					prevExtractor = extractor;
				}

				structure.AddNode(prevNode, "out", new GraphNode(dataset.Name), "dataset");
			}
			else
			{
				GraphNode resourceNode = new GraphNode("internal");
				structure = new GraphStructure(resourceNode);
				structure.AddNode(resourceNode, "resource", new GraphNode(dataset.Name), "dataset");
			}

			Init(structure, 100);
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

					structure.AddNode(prevNode, $"out ({input})", node, $"in ({output})");
				}

				prevNode = node;
				prevLayerConstruct = layerConstruct;
			}

			Init(structure, 100);
		}

		private void Init(IGraphStructure graphStructure, int nodeDistance = 200)
		{
			GraphStructure = graphStructure;
			Content = new NetLayout();
			PopulateNetLayout(Content, graphStructure, nodeDistance);
		}

		/// <summary>
		/// Fill a given layout with a given graph structure.
		/// </summary>
		/// <param name="layout">The layout that will contain the definition for the nodes.</param>
		/// <param name="structure">The structure of the nodes. </param>
		/// <param name="nodeDistance">The distance between nodes (in pixels).</param>
		protected virtual void PopulateNetLayout(NetLayout layout, IGraphStructure structure, int nodeDistance = 200)
		{
			_logger.Debug($"Populating netlayout with a graph (root node: {structure.Root})");

			GraphViewMapping = new Dictionary<GraphNode, NodeViewModel>();

			PopulateForward(structure.Root, nodeDistance);

			GraphViewMapping = null;

			//NodeViewModel node1 = viewModel.CreateNode("Test node", 100, 100, false);
			//NodeViewModel node2 = viewModel.CreateNode("Test node", 400, 100, false);

			//viewModel.Connect(node1, node2, 0, 0);

			_logger.Debug($"Finished populating netlayout with a graph (root node: {structure.Root})");
		}

		private int horizontalOffset = 30;

		protected virtual NodeViewModel PopulateForward(GraphNode node, int nodeDistance = 200)
		{
			NodeViewModel root = new NodeViewModel(node.Name);
			root.X = horizontalOffset;
			root.Y = 450;
			horizontalOffset += nodeDistance;

			int maxStringLength = node.Name.Length;

			GraphViewMapping.Add(node, root);

			foreach (GraphConnection connection in node.Connections)
			{
				// if its an outgoing connection
				if (connection.SourceNode == node)
				{
					// if it is not circular
					if (connection.DestinationNode != node)
					{
						maxStringLength = Math.Max(maxStringLength, connection.SourceName.Length + connection.DestinationName.Length);
						horizontalOffset += maxStringLength * 7;

						NodeViewModel next;
						if (!GraphViewMapping.TryGetValue(connection.DestinationNode, out next))
						{
							next = PopulateForward(connection.DestinationNode, nodeDistance);
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