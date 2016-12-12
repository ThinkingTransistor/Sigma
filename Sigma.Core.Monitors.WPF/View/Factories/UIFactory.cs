using System;
using System.Windows;

// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.View.Factories
{
	public interface IUIFactory<out T>
	{
		T CreatElement(Application app, Window window, params object[] parameters);
	}

	public class LambdaUIFactory : LambdaUIFactory<UIElement>
	{
		public LambdaUIFactory(Func<Application, Window, object[], UIElement> create) : base(create)
		{
		}
	}

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

		public T CreatElement(Application app, Window window, params object[] parameters)
		{
			return _create(app, window, parameters);
		}
	}
}