/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// This marks an <see cref="IParameterVisualiser"/>. It contains information which type this visualiser implements
	/// and reduces the amount of work required to define a new type. 
	/// Multiple attributes can be specified (to display <c>string</c>s and <c>object</c>s for example).
	/// </summary>
	[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = true)]
	public class ParameterVisualiserAttribute : Attribute, IParameterVisualiserInfo
	{
		/// <summary>
		/// The type this visualiser visualises.
		/// </summary>
		public Type Type { get; }

		/// <summary>
		/// The priority of the <see cref="IParameterVisualiserInfo"/>. If another priority with a lower priority has already been added, the
		/// higher priority will override the settings. 
		/// </summary>
		public VisualiserPriority Priority { get; set; } = VisualiserPriority.Normal;

		/// <summary>
		/// Define that the class visualises given type. 
		/// </summary>
		/// <param name="type">The type that is being represented.</param>
		public ParameterVisualiserAttribute(Type type)
		{
			Type = type;
		}

		/// <summary>
		/// Determinse whether the given visualiser is generic or not.
		/// </summary>
		public bool IsGeneric { get; protected set; }
	}

	/// <summary>
	/// This marks an <see cref="IParameterVisualiser"/>. It contains information which type this visualiser implements
	/// and reduces the amount of work required to define a new type. Differently from <see cref="ParameterVisualiserAttribute"/>, the class implementing this
	/// attribute has to be a generic class with a single attribute, which will be the given type.
	/// Multiple attributes can be specified (to display <c>string</c>s and <c>object</c>s for example).
	/// </summary>
	public class GenericParameterVisualiserAttribute : ParameterVisualiserAttribute
	{
		/// <summary>
		/// Define that the class visualises given type. 
		/// </summary>
		/// <param name="type">The type that is being represented.</param>
		public GenericParameterVisualiserAttribute(Type type) : base(type)
		{
			IsGeneric = true;
		}
	}
}