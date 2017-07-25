/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Threading;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeGpu.Workers
{
	public class CudaWorker : BaseWorker
	{
		private ILog Logger => _logger ?? (_logger = LogManager.GetLogger(GetType()));
		private ILog _logger;

		private bool _requireContextBinding;

		public CudaWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority = ThreadPriority.Highest) : base(@operator, handler, priority)
		{
		}

		/// <summary>
		/// This method will be called every time the worker will start from a full stop.
		/// </summary>
		protected override void Initialise()
		{
			base.Initialise();

			_requireContextBinding = true;
		}

		/// <summary>
		/// This method gets updated as long as this worker is not paused or stopped. Maybe only once. 
		/// </summary>
		protected override void DoWork()
		{
			if (_requireContextBinding)
			{
				Logger.Debug($"");

				_requireContextBinding = false;
			}

			base.DoWork();
		}

		/// <summary>
		/// This method gets called when the worker is about to pause. It will also be called when the worker
		/// is stopped.
		/// </summary>
		protected override void OnPause()
		{
		}

		/// <summary>
		/// This method gets called when the worker resumes from its paused state. 
		/// </summary>
		protected override void OnResume()
		{
		}

		/// <summary>
		/// This method gets called when the worker comes to a halt.
		/// </summary>
		protected override void OnStop()
		{
		}
	}
}
