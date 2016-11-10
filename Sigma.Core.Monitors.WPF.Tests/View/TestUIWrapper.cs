using System.Threading;
using System.Windows.Controls;
using NUnit.Framework;
using Sigma.Core.Monitors.WPF.View;

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

				Assert.IsTrue(wrapper.WrappedContent is TestControl);

				wrapper.WrappedContent.Test = "hello";

				Assert.AreEqual(wrapper.WrappedContent.Test, "hello");

				wrapper.WrappedContent = new TestControl() { Test = "world" };

				Assert.AreNotEqual(wrapper.WrappedContent.Test, "hello");
				Assert.AreEqual(wrapper.WrappedContent.Test, "world");
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