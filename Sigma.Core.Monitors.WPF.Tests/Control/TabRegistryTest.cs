using NUnit.Framework;
using Sigma.Core.Monitors.WPF.Control;

namespace Sigma.Core.Monitors.WPF.Tests.Control
{
	public class TabRegistryTest
	{
		[TestCase]
		public void TestContainsTab()
		{
			TabRegistry registry = new TabRegistry();

			Assert.False(registry.ContainsTab("1"));

			registry.AddTab("1");

			Assert.True(registry.ContainsTab("1"));
		}

		[TestCase]
		public void TestAddTab()
		{
			TabRegistry registry = new TabRegistry();

			registry.AddTab("Test1");

			Assert.True(registry.ContainsTab("Test1"));	
		}

		[TestCase]
		public void TestAddTabs()
		{
			TabRegistry registry = new TabRegistry();

			registry.AddTabs("Test1", "Test2");

			Assert.True(registry.ContainsTab("Test1") && registry.ContainsTab("Test2"));
		}
	}
}
