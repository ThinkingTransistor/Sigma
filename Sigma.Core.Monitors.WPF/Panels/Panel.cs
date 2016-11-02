/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.Panels
{
	public interface IPanel
	{
		
	}

	/// <summary>
	/// This panel can be seen as a "subwindow".
	/// One Window consists of one or multiple <see cref="Panel"/>(s) per tab arranged in a grid.
	/// </summary>
	public abstract class Panel : IPanel
	{

	}
}
