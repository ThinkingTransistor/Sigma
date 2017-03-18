/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	/// <summary>
	/// The priority for a given <see cref="IParameterVisualiserInfo"/>. It is used in order to provide an easy way to force
	/// certain <see cref="IParameterVisualiser"/>s (e.g. setting the priority higher than default (<see cref="Higher"/>)).
	/// </summary>
	public enum VisualiserPriority
	{
		/// <summary>
		/// The lowest possible priority.
		/// </summary>
		Lowest,
		/// <summary>
		/// The second lowest priority. Use this to mark classes that may be exchanged by the user
		/// </summary>
		Lower,
		/// <summary>
		/// The default priority that is automatically assigned.
		/// </summary>
		Normal,
		/// <summary>
		/// The second highes priority. Use this for a framework that should replace all default visualisers.
		/// </summary>
		Higher,
		/// <summary>
		/// The highers priority. Use this to hard-override visualisers.
		/// </summary>
		Highest
	}

	/// <summary>
	/// This interface specifies additional information for a <see cref="IParameterVisualiser"/>.
	/// It is requried in order to add a visualiser to the manager (<see cref="IParameterVisualiserManager"/>).
	/// </summary>
	public interface IParameterVisualiserInfo
	{
		/// <summary>
		/// The type this visualiser visualises.
		/// </summary>
		Type Type { get; }

		/// <summary>
		/// The priority of the <see cref="IParameterVisualiserInfo"/>. If another priority with a lower priority has already been added, the
		/// higher priority will override the settings. 
		/// </summary>
		VisualiserPriority Priority { get; set; }
	}
}