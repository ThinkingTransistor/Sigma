using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View
{
	internal abstract class UIWrapper<T> where T : ContentControl, new()
	{
		protected T content;

		public UIWrapper() : this(new T()) { }

		public UIWrapper(T content)
		{
			this.content = content;
		}

		public T Content
		{
			get
			{
				return content;
			}
			set
			{
				content = value;
			}
		}

		public static explicit operator T(UIWrapper<T> wrapper)
		{
			return wrapper.content;
		}
	}
}
