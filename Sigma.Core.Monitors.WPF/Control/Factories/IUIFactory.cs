using System;
using System.Windows;
// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.Control.Factories
{
	public interface IUIFactory
	{
		UIElement CreatElement(App app, Window window);
	}

	public interface IUIFactory<out T> : IUIFactory where T : UIElement
	{
		new T CreatElement(App app, Window window);
	}

	public class LambdaUIFactory : LambdaUIFactory<UIElement>
	{
		public LambdaUIFactory(Func<UIElement> create) : base(create) { }
	}

	public class LambdaUIFactory<T> : IUIFactory<T> where T : UIElement
	{
		private readonly Func<T> _create;

		public LambdaUIFactory(Func<T> create)
		{
			if (create == null) throw new ArgumentNullException(nameof(create));

			_create = create;
		}

		public T CreatElement(App app, Window window)
		{
			return _create();
		}

		UIElement IUIFactory.CreatElement(App app, Window window)
		{
			return CreatElement(app, window);
		}
	}
}