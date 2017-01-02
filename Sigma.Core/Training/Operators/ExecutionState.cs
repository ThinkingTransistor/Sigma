/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Training.Operators
{
	/// <summary>
	/// An state for diverse training tasks.
	/// </summary>
	public enum ExecutionState
	{
		None,
		Running,
		Paused,
		Stopped
	}
}