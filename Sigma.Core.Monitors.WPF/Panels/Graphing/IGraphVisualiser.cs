using Sigma.Core.Monitors.WPF.NetView.Graphing;

namespace Sigma.Core.Monitors.WPF.Panels.Graphing
{
	/// <summary>
	/// A class that can visaulise a graph structure.
	/// </summary>
	public interface IGraphVisualiser
	{
		/// <summary>
		/// Get the graph structure that is being represented.
		/// </summary>
		IGraphStructure GraphStructure { get; }
	}
}