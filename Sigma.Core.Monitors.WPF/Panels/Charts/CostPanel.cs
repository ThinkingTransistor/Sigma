namespace Sigma.Core.Monitors.WPF.Panels.Charts
{
	/// <summary>
	/// Visualises the cost of a given trainer automatically. 
	/// </summary>
	public class CostPanel : CartesianChartPanel
	{
		/// <summary>
		///     Create a CostPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public CostPanel(string title, object headerContent = null) : base(title, headerContent)
		{
			
		}
	}
}