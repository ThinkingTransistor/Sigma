/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators
{
	/// <summary>
	///     An operator that operates (executes) the training process defined in a trainer.
	///     Operators typically split the workload into multiple workers and backends for CPU, GPU and inter-device cooperation
	///     are provided.
	/// </summary>
	public interface IOperator
	{
		/// <summary>
		///     The <see cref="SigmaEnvironment" /> this operator runs in and communicates with.
		///     It will be automatically set by the <see cref="ITrainer" />.
		/// </summary>
		SigmaEnvironment Sigma { get; set; }

		/// <summary>
		///     The current <see cref="ExecutionState" /> of the <see cref="IOperator" />. <see cref="ExecutionState.None" />
		///     if the operator has not been started yet.
		/// </summary>
		ExecutionState State { get; }

		/// <summary>
		///     The <see cref="IComputationHandler" /> used to compute everything in
		///     this <see cref="IOperator" />. It will be automatically set by the
		///     <see cref="ITrainer" /> if not specified.
		/// </summary>
		IComputationHandler Handler { get; set; }

		/// <summary>
		///     The <see cref="ITrainer" /> that is being trained in this operators training process.
		///     This is automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		ITrainer Trainer { get; set; }

		/// <summary>
		///     The <see cref="INetwork" /> the training process is operated on.
		///     This is automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		INetwork Network { get; set; }

		/// <summary>
		///		This merger is used to merge multiple networks after they are
		///		submitted to the <see cref="IOperator"/>.
		/// </summary>
		INetworkMerger Merger { get; set; }

		/// <summary>
		///     The number of <see cref="Workers.IWorker" />s (threads) used in this
		///     <see cref="IOperator" /> in parallel.
		/// </summary>
		int WorkerCount { get; }

		/// <summary>
		///     Attach a hook to this operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		void AttachHook(IHook hook);

		/// <summary>
		///     Detach a hook from this operator.
		/// </summary>
		/// <param name="hook">The hook to detach.</param>
		void DetachHook(IHook hook);

		/// <summary>
		/// A <see cref="IWorker"/> calls this method to report its current progress 
		/// to the <see cref="IOperator"/>. 
		/// </summary>
		/// <param name="worker"></param>
		void ReportProgress(IWorker worker);

		/// <summary>
		///     Start this operator in a separate thread (return immediately).
		/// </summary>
		void Start();

		/// <summary>
		///		Start this operator for a single time only (return immediately).
		/// </summary>
		void StartOnce();

		/// <summary>
		///     Signal this operator to pause as soon as possible.
		/// </summary>
		void SignalPause();

		/// <summary>
		///     Signal this operator to resume as soon as possible.
		/// </summary>
		void SignalResume();

		/// <summary>
		///     Signal this operator to stop as soon as possible.
		/// </summary>
		void SignalStop();

		/// <summary>
		///     This method blocks until the last state change has been fully performed.
		///     Returns immediately if not implemented.
		/// </summary>
		void WaitForStateChanged();
	}
}