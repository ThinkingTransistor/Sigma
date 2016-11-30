/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// A custom initialiser adapter for simple initialiser assignment with lambdas. 
	/// </summary>
	public class CustomInitialiser : BaseInitialiser
	{
		private readonly Func<long[], long[], Random, object> _valueFunc;

		public CustomInitialiser(Func<long[], long[], Random, object> valueFunc)
		{
			if (valueFunc == null) throw new ArgumentNullException(nameof(valueFunc));

			_valueFunc = valueFunc;
		}

		public override object GetValue(long[] indices, long[] shape, Random random)
		{
			return _valueFunc(indices, shape, random);
		}
	}
}
