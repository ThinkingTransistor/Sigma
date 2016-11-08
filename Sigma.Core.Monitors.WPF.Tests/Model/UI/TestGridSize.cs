using System;
using NUnit.Framework;
using Sigma.Core.Monitors.WPF.Model.UI;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;

namespace Sigma.Core.Monitors.WPF.Tests.Model.UI
{
	public class TestGridSize
	{
		[TestCase]
		public void TestGridSizeCreationByConstructor()
		{
			GridSize grid = new GridSize(1, 2);

			Assert.AreEqual(grid.Rows, 1);
			Assert.AreEqual(grid.Columns, 2);
		}

		[TestCase]
		public void TestGridSizeCreationByIntArray()
		{
			GridSize grid = new[] { 1, 2 };

			Assert.AreEqual(grid.Rows, 1);
			Assert.AreEqual(grid.Columns, 2);
		}

		[TestCase]
		public void TestGridSizeAssign()
		{
			GridSize grid = new GridSize(1, 1);
			grid.Rows = 2;
			grid.Columns = 3;

			Assert.AreEqual(grid.Rows, 2);
			Assert.AreEqual(grid.Columns, 3);

			grid.Set(4, 5);
			Assert.AreEqual(grid.Rows, 4);
			Assert.AreEqual(grid.Columns, 5);
		}


		[TestCase]
		public void TestGridSizeIllegalArguments()
		{
			GridSize grid;

			Assert.Throws<ArgumentException>(() => grid = new GridSize(0, 1));
			Assert.Throws<ArgumentException>(() => grid = new GridSize(1, 0));
			Assert.Throws<ArgumentException>(() => grid = new GridSize(-1, 1));
			Assert.Throws<ArgumentException>(() => grid = new GridSize(1, -1));

			grid = new GridSize(1, 1);

			Assert.Throws<ArgumentException>(() => grid.Columns = 0);
			Assert.Throws<ArgumentException>(() => grid.Rows = 0);
			Assert.Throws<ArgumentException>(() => grid.Columns = -1);
			Assert.Throws<ArgumentException>(() => grid.Rows = -1);

			Assert.Throws<ArgumentException>(() => grid.Set(0, 1));
			Assert.Throws<ArgumentException>(() => grid.Set(1, 0));
			Assert.Throws<ArgumentException>(() => grid.Set(-1, 1));
			Assert.Throws<ArgumentException>(() => grid.Set(1, -1));
		}
	}
}
