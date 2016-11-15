/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Utils
{
	public class TestTaskObserver
	{
		[TestCase]
		public void TestTaskObserverCreate()
		{
			Assert.Throws<ArgumentNullException>(() => new TaskObserver(null));

			TaskObserver observer = new TaskObserver(TaskType.PREPARE, "description", false);

			Assert.AreSame(TaskType.PREPARE, observer.Type);
			Assert.AreEqual("description", observer.Description);
			Assert.IsFalse(observer.Exposed);
		}
	}
}
