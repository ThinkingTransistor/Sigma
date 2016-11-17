using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Training.Initialisers
{
	public interface IInitialiser
	{
		void Initialise(INDArray array);
	}
}
