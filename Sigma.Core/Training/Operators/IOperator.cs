/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Training.Operators
{
	public interface IOperator
	{
		IComputationHandler Handler { get; }

		ITrainer Trainer { get; }

		INetwork Network { get; }

		void Run();
	}
}
