/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu;
using Sigma.Core.Persistence.Selectors;
using Sigma.Core.Training.Operators.Backends.NativeGpu.Workers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeGpu
{
	public class CudaSinglethreadedOperator : BaseOperator
	{
		public int DeviceId { get; }

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> with a specified <see cref="IComputationHandler" />.
		///     The <see cref="IComputationHandler" /> will <c>not</c> be modified by the <see cref="ITrainer" />.
		/// </summary>
		public CudaSinglethreadedOperator(int deviceId = 0) : base(new CudaFloat32Handler(deviceId), 1)
		{
			DeviceId = deviceId;
		}

		/// <summary>
		/// Create an instance of this operator with the same parameters.
		/// Used for shallow-copying state to another operator (e.g. for persistence / selection).
		/// </summary>
		/// <returns></returns>
		protected override BaseOperator CreateDuplicateInstance()
		{
			return new CudaSinglethreadedOperator(DeviceId);
		}

		/// <summary>
		///     This method creates an <see cref="IWorker" />.
		/// </summary>
		/// <returns>The newly created <see cref="IWorker" />.</returns>
		protected override IWorker CreateWorker()
		{
			return new CudaWorker(this, new CudaFloat32Handler(DeviceId));
		}

		/// <summary>
		/// Get an operator selector for this operator.
		/// </summary>
		/// <returns>The selector for this operator.</returns>
		public override IOperatorSelector<IOperator> Select()
		{
			throw new System.NotImplementedException();
		}
	}
}
