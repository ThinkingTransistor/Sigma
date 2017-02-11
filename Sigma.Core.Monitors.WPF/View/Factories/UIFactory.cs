/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;

// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.View.Factories
{
	public interface IUIFactory<out T>
	{
		/// <summary>
		/// Create an element of the specified type T and set all required parameters (normally it is an <see cref="UIElement"/>). 
		/// If additional parameters are required, use <see cref="parameters"/>.
		/// </summary>
		/// <param name="app">The <see cref="Application"/> in which the newly generated item will be.</param>
		/// <param name="window">The <see cref="Window"/> in which the newly generated item will be.</param>
		/// <param name="parameters">The parameters that may or may not be required. Often none are required.</param>
		/// <returns>The newly created item.</returns>
		T CreateElement(Application app, Window window, params object[] parameters);
	}

	public abstract class UIElementFactory<T> : IUIFactory<T> where T : FrameworkElement
	{
		/// <summary>
		///	The style that will be applied to every created <see cref="FrameworkElement"/>.
		/// </summary>
		public Style Style { get; set; }

		/// <summary>
		/// Create an <see cref="FrameworkElement"/> of the specified type T and set all required parameters and style. 
		/// If additional parameters are required, use <see cref="parameters"/>.
		/// </summary>
		/// <param name="app">The <see cref="Application"/> in which the newly generated item will be.</param>
		/// <param name="window">The <see cref="Window"/> in which the newly generated item will be.</param>
		/// <param name="parameters">The parameters that may or may not be required. Often none are required.</param>
		/// <returns>The newly created item.</returns>
		public T CreateElement(Application app, Window window, params object[] parameters)
		{
			T element = CreateFrameworkElement(app, window, parameters);

			element.Style = Style;

			return element;
		}

		protected abstract T CreateFrameworkElement(Application app, Window window, params object[] parameters);
	}

	/// <summary>
	/// Implementation of <see cref="IUIFactory{T}"/> that allows a lambda-function that creates an <see cref="UIElement"/> to be passed as factory. 
	/// </summary>
	public class LambdaUIFactory : LambdaUIFactory<UIElement>
	{
		public LambdaUIFactory(Func<Application, Window, object[], UIElement> create) : base(create) { }
	}

	/// <summary>
	/// Implementation of <see cref="IUIFactory{T}"/> that allows a generic lambda-function to be passed as factory. 
	/// </summary>
	public class LambdaUIFactory<T> : IUIFactory<T>
	{
		private readonly Func<Application, Window, object[], T> _create;

		public LambdaUIFactory(Func<Application, Window, object[], T> create)
		{
			if (create == null)
			{
				throw new ArgumentNullException(nameof(create));
			}

			_create = create;
		}

		/// <summary>
		/// Create an element of the specified type T and set all required parameters (normally it is an <see cref="UIElement"/>) based on the lambda function specified in the constructor. 
		/// If additional parameters are required, use <see cref="parameters"/>.
		/// </summary>
		/// <param name="app">The <see cref="Application"/> in which the newly generated item will be.</param>
		/// <param name="window">The <see cref="Window"/> in which the newly generated item will be.</param>
		/// <param name="parameters">The parameters that may or may not be required. Often none are required.</param>
		/// <returns>The newly created item.</returns>
		public T CreateElement(Application app, Window window, params object[] parameters)
		{
			return _create(app, window, parameters);
		}
	}
}