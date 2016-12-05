using System;
using System.Windows;
// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.Control.Factories
{
	public interface IUIFactory<out T>
	{
		T CreatElement(App app, Window window);
	}

	public class LambdaUIFactory : LambdaUIFactory<UIElement>
	{
		public LambdaUIFactory(Func<App, Window, UIElement> create) : base(create) { }
	}

	public class LambdaUIFactory<T> : IUIFactory<T>
	{
		private readonly Func<App, Window, T> _create;

		public LambdaUIFactory(Func<App, Window, T> create)
		{
			if (create == null) throw new ArgumentNullException(nameof(create));

			_create = create;
		}

		public T CreatElement(App app, Window window)
		{
			return _create(app, window);
		}
	}
}