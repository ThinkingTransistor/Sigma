using System;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// This marks an <see cref="IParameterVisualiser"/>. It contains information which type this visualiser implements
	/// and reduces the amount of work required to define a new type. 
	/// Multiple attributes can be specified (to display <c>string</c>s and <c>object</c>s for example).
	/// </summary>
	[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = true)]
	public class ParameterVisualiserAttribute : Attribute
	{
		/// <summary>
		/// The type this visualiser visualises.
		/// </summary>
		public Type Type { get; }

		public VisualiserPriority Priority { get; set; } = VisualiserPriority.Normal;

		/// <summary>
		/// Define that the class visualises given type. 
		/// </summary>
		/// <param name="type"></param>
		public ParameterVisualiserAttribute(Type type)
		{
			Type = type;
		}

		public enum VisualiserPriority
		{
			Lowest,
			Lower,
			Normal,
			Higher,
			Highest
		}
	}
}