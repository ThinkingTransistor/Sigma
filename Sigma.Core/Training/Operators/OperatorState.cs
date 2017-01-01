/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Training.Operators
{
	/// <summary>
	/// An operator state.
	/// </summary>
	public enum OperatorState
	{
		None,
		Running,
		Paused,
		Stopped
	}
}