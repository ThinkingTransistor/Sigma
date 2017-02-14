/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using System.Net;
using NUnit.Framework;
using Sigma.Core.Data.Sources;

namespace Sigma.Tests.Data.Sources
{
	public class TestUrlSource
	{
		private static bool _checkedInternetConnection;
		private static bool _internetConnectionValid;

		private static void AsserIgnoreIfNoInternetConnection()
		{
			if (!_checkedInternetConnection)
			{
				_internetConnectionValid = false;

				try
				{
					using (WebClient client = new WebClient())
					{
						//this framework will be irrelevant long before google goes down
						using (Stream stream = client.OpenRead("http://www.google.com"))
						{
							_internetConnectionValid = true;
						}
					}
				}
				catch
				{
					_checkedInternetConnection = false;
				}
			}

			if (!_internetConnectionValid)
			{
				Assert.Ignore();
			}
		}

		[TestCase]
		public void TestUrlSourceCheckNonExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			UrlSource source = new UrlSource("http://thisisnotavalidwebsiteisitnoitisnt.noitisnt/notexistingfile.dat");

			Assert.IsFalse(source.Exists());
		}

		[TestCase]
		public void TestUrlSourceAccessNonExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			UrlSource source = new UrlSource("http://thisisnotavalidwebsiteisitnoitisnt.noitisnt/notexistingfile.dat");

			Assert.Throws<InvalidOperationException>(() => source.Prepare());

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());
		}

		[TestCase]
		public void TestUrlSourceCheckExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			UrlSource source = new UrlSource("https://www.google.com/robots.txt");

			Assert.IsTrue(source.Exists());
		}

		[TestCase]
		public void TestUrlSourceAccessExisting()
		{
			AsserIgnoreIfNoInternetConnection();

			UrlSource source = new UrlSource("https://www.google.com/robots.txt", Path.GetTempPath() + ".unittestfileurltest1");

			Assert.Throws<InvalidOperationException>(() => source.Retrieve());

			source.Prepare();

			Stream stream = source.Retrieve();

			Assert.IsNotNull(stream);

			stream.Dispose();

			File.Delete(Path.GetTempPath() + ".unittestfileurltest1");
		}
	}
}
