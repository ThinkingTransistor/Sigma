/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Sigma.Core.Persistence.Selectors
{
	/// <summary>
	/// A selector used to selectively keep and discards parts of compatible objects (e.g. get trainer without runtime data to store on disk).
	/// Note: All selector operations must provide a result object with legal object state (the <see cref="Result"/> object must always be valid and usable).
	/// </summary>
	/// <typeparam name="T">The type this selector is selecting from.</typeparam>
	public interface ISelector<out T>
	{
		/// <summary>
		/// The current result object of type <see cref="T"/>.
		/// </summary>
		T Result { get; }

		/// <summary>
		/// Get the "emptiest" available version of this <see cref="T"/> while still retaining a legal object state for type <see cref="T"/>.
		/// Note: Any parameters that are fixed per-object (such as unique name) must be retained.
		/// </summary>
		/// <returns>An empty version of the object.</returns>
		ISelector<T> Empty();

		/// <summary>
		/// Get an uninitialised version of this <see cref="T"/> without any runtime information, but ready to be re-initialised.
		/// </summary>
		/// <returns></returns>
		ISelector<T> Uninitialised();
	}

	/// <summary>
	/// A selector component and its id. 
	/// </summary>
	public abstract class SelectorComponent
	{
		/// <summary>
		/// The component id of this component. 
		/// </summary>
		public int Id { get; }

		/// <summary>
		/// The selected sub components of this component (if any).
		/// </summary>
		public SelectorComponent[] SubComponents { get; }

		/// <summary>
		/// Create a selector component with a certain id. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		protected SelectorComponent(int id)
		{
			if (id < 0) throw new ArgumentOutOfRangeException($"Sub id must be >= 0 but was {id}.");

			Id = id;
		}

		/// <summary>
		/// Create a selector component with a certain id and certain sub-components. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		/// <param name="subComponents">The selected sub components.</param>
		protected SelectorComponent(int id, params SelectorComponent[] subComponents) : this(id)
		{
			if (subComponents == null) throw new ArgumentNullException(nameof(subComponents));

			SubComponents = subComponents;
		}
	}
}
