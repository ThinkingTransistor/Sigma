using System.Windows;
using System.Windows.Data;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public static class BindUtils
	{
		public static Binding Bind(object source, string sourcePropertyName, DependencyObject sourceObject, DependencyProperty targetProperty, BindingMode mode = BindingMode.Default)
		{
			Binding binding = new Binding
			{
				Source = source,
				Path = new PropertyPath(sourcePropertyName),
				Mode = mode,
				UpdateSourceTrigger = UpdateSourceTrigger.PropertyChanged
			};
			BindingOperations.SetBinding(sourceObject, targetProperty, binding);

			return binding;
		}
	}
}