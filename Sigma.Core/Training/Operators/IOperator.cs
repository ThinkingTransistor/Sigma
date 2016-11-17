/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Training.Operators
{
	/// <summary>
	/// An operator that operates (executes) the training process defined in a trainer.
	/// Operators typically split the workload into multiple workers and backends for CPU, GPU and inter-device cooperation are provided. 
	/// </summary>
	public interface IOperator
	{
		/// <summary>
		/// The sigma environment this operator runs in and communicates with. 
		/// </summary>
		SigmaEnvironment Sigma { get; set; }

		/// <summary>
		/// The handler used to compute everything in this operator.
		/// </summary>
		IComputationHandler Handler { get; }

		/// <summary>
		/// The trainer that is being trained in this operators training process.
		/// </summary>
		ITrainer Trainer { get; }

		/// <summary>
		/// The network the training process is operated on.
		/// </summary>
		INetwork Network { get; }

		/// <summary>
		/// The number of workers used in this operator in parallel. 
		/// </summary>
		int WorkerCount { get; }

		/// <summary>
		/// Attach a hook to this operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		void AttachHook(IHook hook);

		/// <summary>
		/// Start this operator in a separate thread (return immediately). 
		/// </summary>
		void Start();
	}
}
