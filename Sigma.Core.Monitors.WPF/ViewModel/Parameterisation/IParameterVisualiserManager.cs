/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Monitors.WPF.Annotations;
using Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults;

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	/// <summary>
	/// A manager that keeps track of all <see cref="IParameterVisualiser"/> and the types the represent. 
	/// </summary>
	public interface IParameterVisualiserManager
	{
		/// <summary>
		/// Add a new <see cref="IParameterVisualiser"/> to the mapping. This requires the class itself and some additional information.
		/// </summary>
		/// <param name="visualiserClass">The class that represents the type (e.g. <see cref="SigmaCheckBox"/>).</param>
		/// <param name="parameterInfo">The info for the visualiser (type it represents ... ).</param>
		/// <returns><c>True</c> if the type could be added successfully. <c>False</c> otherwise (e.g. an <see cref="IParameterVisualiser"/> with a higher
		/// priority has already been added).</returns>
		bool Add([NotNull]Type visualiserClass, [NotNull]IParameterVisualiserInfo parameterInfo);

		/// <summary>
		/// Remove a <see cref="IParameterVisualiser"/> from the mapping. Since there can always only be one parameter per type, 
		/// the <see cref="Type"/> that is visualised is enough.
		/// </summary>
		/// <param name="type">The type that is being visualised (e.g. <c>typeof(bool)</c>)</param>
		/// <returns><c>True</c> if the mapping could be removed successfully; <c>False</c> otherwise. </returns>
		bool Remove([NotNull]Type type);

		/// <summary>
		/// Get the type which is used to visualise given type (e.g. <c>typeof(bool)</c>). 
		/// </summary>
		/// <param name="type">The object type which will be displayed.</param>
		/// <returns>The closest type for visualisation. <c>null</c> if not found.</returns>
		/// <exception cref="ArgumentNullException">If the given type is null.</exception>
		Type VisualiserType([NotNull] Type type);

		/// <summary>
		/// Get the type which is used to visualise given object (reference not type). 
		/// </summary>
		/// <param name="obj">The object which will be displayed.</param>
		/// <returns>The closest type for visualisation. <c>null</c> if not found.</returns>
		Type VisualiserTypeByReference([NotNull] object obj);

		/// <summary>
		/// Instantiate a visualiser that can represent given type.
		/// </summary>
		/// <param name="type">The type that will be visualies</param>
		/// <returns>An instance of a visualiser.</returns>
		IParameterVisualiser InstantiateVisualiser(Type type);
	}
}