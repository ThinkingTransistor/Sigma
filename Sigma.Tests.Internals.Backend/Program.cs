using System;
using Sigma.Core;
using Sigma.Core.Math;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			NDArray<int> array = new NDArray<int>(ArrayUtils.Range(1, 12), 2, 6);

			Console.WriteLine("originalshape: " + ArrayUtils.ToString(array.Shape));
			Console.WriteLine(array);

			array.ReshapeSelf(4, 3);

			Console.WriteLine("array.ReshapeSelf(4, 3)");

			Console.WriteLine(array);

			array.TransposeSelf();

			Console.WriteLine("array.TransposeSelf()");

			Console.WriteLine(array);

			Console.ReadKey();
		}
	}
}
