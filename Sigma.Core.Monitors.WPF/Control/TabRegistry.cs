using Sigma.Core.Monitors.WPF.Model;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Control
{
	public class TabRegistry : Registry, IRegistry
	{
		public TabRegistry(IRegistry parent = null) : base(parent) { }

		/// <summary>
		/// Add a single <see cref="Tab"/> to the <see cref="TabControl"/>.
		/// </summary>
		/// <param name="tab">The <see cref="Tab"/> to add.</param>
		public void AddTab(Tab tab)
		{
			Set(tab.Title, tab, typeof(Tab));
		}

		/// <summary>
		/// Add multiple <see cref="Tab"/>s at once to the <see cref="TabControl"/>.
		/// </summary>
		/// <param name="tabs">The <see cref="Tab"/>s to add. </param>
		public void AddTabs(params Tab[] tabs)
		{
			foreach (Tab tab in tabs)
			{
				AddTab(tab);
			}
		}

		public bool ContainsTab(Tab tab)
		{
			return Contains(tab.Title, tab);
		}

		new public Tab this[string identifier]
		{
			get
			{
				return Get<Tab>(identifier);
			}

			set
			{
				Set(identifier, value);
			}
		}
	}
}
