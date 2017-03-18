/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	/// <summary>
	/// An interface for the label that describes the names of parameters. 
	/// </summary>
	public interface IParameterLabel
	{
		/// <summary>
		/// Set the state of a visualiser to pending.
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		void Pending(IParameterVisualiser visualiser);

		/// <summary>
		/// Set the state of a visualiser to errored.
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		void Errored(IParameterVisualiser visualiser);

		/// <summary>
		/// Set the state of a visualiser to success (i.e. don't care anymore).
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		void Success(IParameterVisualiser visualiser);
	}
}