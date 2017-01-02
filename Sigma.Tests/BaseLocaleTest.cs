using System.Globalization;
using System.Threading;
using NUnit.Framework;

namespace Sigma.Tests
{
	public class BaseLocaleTest
	{
		private static readonly CultureInfo DefaultCultureInfo = new CultureInfo("en-GB");

		[SetUp]
		public void SetUp()
		{
			SetDefaultCulture(DefaultCultureInfo);
		}

		private static void SetDefaultCulture(CultureInfo culture)
		{
			Thread.CurrentThread.CurrentCulture = culture;
			CultureInfo.DefaultThreadCurrentCulture = culture;
		}
	}
}