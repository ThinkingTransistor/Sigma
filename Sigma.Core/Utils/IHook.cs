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

namespace Sigma.Core.Utils
{
	public interface IHook
	{
		ITimeStep TimeStep { get; }

		ISet<string> RequiredRegistryEntries { get; }

		void Execute(IRegistry registry);		
	}

	public interface ITimeStep
	{
		TimeScale TimeScale { get; }

		int Interval { get; }

		int LiveTime { get; }
	}

	public class TimeScale
	{
		public static readonly TimeScale EPOCH = new TimeScale();
		public static readonly TimeScale UPDATE = new TimeScale();
	}
}