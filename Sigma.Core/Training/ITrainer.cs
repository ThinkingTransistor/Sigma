/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;
using System.Collections.Generic;
using Sigma.Core.Training.Hooks;

namespace Sigma.Core.Training
{
	/// <summary>
	/// A trainer that defines how a network should be trained, denoting initialisers, optimiser, operator, custom hooks and data to apply and use. 
	/// </summary>
	public interface ITrainer
	{
		/// <summary>
		/// The unique name of this trainer. 
		/// </summary>
		string Name { get; }

		/// <summary>
		/// The network to be trained with this trainer. 
		/// </summary>
		INetwork Network { get; set; }

		/// <summary>
		/// The initialisers used in this trainer by layer and then by parameter name. 
		/// Registry resolve notation may be used as the initialiser will be executed on all ndarrays which resolve to a match in a certain layer and match identifier. 
		/// </summary>
		IReadOnlyDictionary<string, IReadOnlyDictionary<string, IInitialiser>> Initialisers { get; set; }

		/// <summary>
		/// The optimiser used in this trainer (e.g. Stochastic gradient descent, momentum).
		/// </summary>
		IOptimiser Optimiser { get; set; }

		/// <summary>
		/// The operator which controls the training process and effectively operates this trainer at runtime. 
		/// </summary>
		IOperator Operator { get; set; }

		/// <summary>
		/// The primary training data iterator, used to yield training data for the network to train on.
		/// </summary>
		IDataIterator TrainingDataIterator { get; set; }

		/// <summary>
		/// The hooks attached to this trainer. 
		/// </summary>
		IReadOnlyCollection<IHook> Hooks { get; }

		/// <summary>
		/// Add a secondary named data iterator to this trainer.
		/// Note: Secondary data iterators can for example be used for model validation with separate data.
		/// </summary>
		/// <param name="name"></param>
		/// <param name="iterator"></param>
		void AddNamedDataIterator(string name, IDataIterator iterator);

		/// <summary>
		/// Add an active hook to this trainer, which will be executed during runtime directly in the operator. 
		/// </summary>
		/// <param name="hook">The active hook to add to this trainer.</param>
		void AddActiveHook(IActiveHook hook);

		/// <summary>
		/// Add a passive hook to this trainer, which will be executed asynchronously or in the owning monitor.
		/// </summary>
		/// <param name="hook">The passive hook to add to this trainer.</param>
		void AddPassiveHook(IPassiveHook hook);

		/// <summary>
		/// Initialise this trainer and the network to be trained using the set initialisers.
		/// </summary>
		void Initialise();
	}
}
