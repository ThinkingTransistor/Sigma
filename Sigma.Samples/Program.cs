using log4net;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Samples
{
	class Program
	{
		static void Main(string[] args)
		{
			ILog log = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

			log.Info("Test");

			Console.ReadKey();
		}
	}
}
