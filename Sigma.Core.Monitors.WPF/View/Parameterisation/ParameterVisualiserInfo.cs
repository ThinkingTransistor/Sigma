/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.ComponentModel;
using Sigma.Core.Monitors.WPF.Annotations;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// The default implementation for <see cref="IParameterVisualiserInfo"/> so it is easy to add new items (and nobody implements it badly).
	/// </summary>
	public class ParameterVisualiserInfo : IParameterVisualiserInfo
	{
		/// <inheritdoc />
		public Type Type { get; }

		/// <inheritdoc />
		public VisualiserPriority Priority { get; set; }

		/// <inheritdoc />
		public bool IsGeneric { get; }

		/// <summary>
		/// Initializes a new instance of the <see cref="IParameterVisualiserInfo" /> class.
		/// </summary>
		/// <param name="type">The type the <see cref="IParameterVisualiser"/> is responsible for.</param>
		/// <param name="priority">The priority of the info. (higher prioriuty overrides lower ones).</param>
		/// <param name="isGeneric">Determinse whether the given visualiser is generic or not.</param>
		/// <exception cref="InvalidEnumArgumentException">If bad enum is passed.</exception>
		/// <exception cref="ArgumentNullException">If <see ref="type"/> is <c>null</c>.</exception>
		public ParameterVisualiserInfo([NotNull] Type type, VisualiserPriority priority = VisualiserPriority.Normal, bool isGeneric = false)
		{
			if (type == null) throw new ArgumentNullException(nameof(type));
			if (!Enum.IsDefined(typeof(VisualiserPriority), priority))
			{
				throw new InvalidEnumArgumentException(nameof(priority), (int)priority, typeof(VisualiserPriority));
			}

			Type = type;
			Priority = priority;
			IsGeneric = isGeneric;
		}

		
	}
}