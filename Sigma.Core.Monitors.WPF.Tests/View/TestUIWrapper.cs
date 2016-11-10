using NUnit.Framework;
using Sigma.Core.Monitors.WPF.View;
using System.Threading;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Tests.View
{
	public class TestUIWrapper
	{
		[TestCase]
		public void TestUIWrapperCreation()
		{
			Thread fred = new Thread(() =>
			{
				TestUIWrapperClass wrapper = new TestUIWrapperClass();

				Assert.IsTrue(wrapper.Content is TestControl);

				wrapper.Content.Test = "hello";

				Assert.AreEqual(wrapper.Content.Test, "hello");

				wrapper.Content = new TestControl() { Test = "world" };

				Assert.AreNotEqual(wrapper.Content.Test, "hello");
				Assert.AreEqual(wrapper.Content.Test, "world");
			});

			fred.SetApartmentState(ApartmentState.STA);
			fred.Priority = ThreadPriority.Highest;
			fred.Start();

			fred.Join();
		}

		private class TestControl : ContentControl
		{
			public string Test { get; set; }
		}

		private class TestUIWrapperClass : UIWrapper<TestControl>
		{

		}
	}
}