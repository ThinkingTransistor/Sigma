/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	/// <summary>
	/// A worker that executes its operations on the cpu with a cpu backend (default behavior in base worker, separate class for future proofing).
	/// </summary>
	public class CpuWorker : BaseWorker
	{
		private ILog Logger => _logger ?? (_logger = LogManager.GetLogger(GetType()));
		private ILog _logger;

		public CpuWorker(IOperator @operator) : base(@operator) { }

		public CpuWorker(IOperator @operator, IComputationHandler handler) : base(@operator, handler) { }

		public CpuWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority) : base(@operator, handler, priority) { }

		/// <inheritdoc />
		protected override void OnPause()
		{
		}

		/// <inheritdoc />
		protected override void OnResume()
		{
		}

		/// <inheritdoc />
		protected override void OnStop()
		{
		}
	}
}