/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Sources;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data.Sources
{
	public class TestURLSource
	{
		private static bool checkedInternetConnection;
		private static bool internetConnectionValid;

		private static void AsserIgnoreIfNoInternetConnection()
		{
			if (!checkedInternetConnection)
			{
				internetConnectionValid = false;

				try
				{
					using (WebClient client = new WebClient())
					{
						//this framework will be irrelevant long before google goes down
						using (Stream stream = client.OpenRead("http://www.google.com"))
						{
							internetConnectionValid = true;
						}
					}
				}
				catch
				{
					checkedInternetConnection = false;
				}
			}

			if (!internetConnectionValid)
			{
				Assert.Ignore();
			}
		}

		[TestCase]
		public void TestURLSourceCheckNonExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			URLSource source = new URLSource("http://thisisnotavalidwebsiteisitnoitisnt.noitisnt/notexistingfile.dat");

			Assert.IsFalse(source.Exists());
		}

		[TestCase]
		public void TestURLSourceAccessNonExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			URLSource source = new URLSource("http://thisisnotavalidwebsiteisitnoitisnt.noitisnt/notexistingfile.dat");

			Assert.Throws<InvalidOperationException>(() => source.Prepare());

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());
		}

		[TestCase]
		public void TestURLSourceCheckExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			URLSource source = new URLSource("https://www.google.com/robots.txt");

			Assert.IsTrue(source.Exists());
		}

		[TestCase]
		public void TestURLSourceAccessExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			URLSource source = new URLSource("https://www.google.com/robots.txt", Path.GetTempPath() + ".unittestfileurltest1");

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());

			source.Prepare();

			Stream stream = source.Retrieve();

			Assert.IsNotNull(stream);

			stream.Dispose();

			File.Delete(Path.GetTempPath() + ".unittestfileurltest1");
		}
	}
}
