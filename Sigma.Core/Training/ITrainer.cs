/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Optimisers;
using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

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
		/// The sigma environment this trainer is associated with.
		/// </summary>
		SigmaEnvironment Sigma { get; set; }

		/// <summary>
		/// The network to be trained with this trainer. 
		/// </summary>
		INetwork Network { get; set; }

		/// <summary>
		/// The initialisers used in this trainer by registry resolve string (e.g. FC*.weights, *.weights, Layer1.biases, Layer2.*).
		/// Registry resolve notation may be used as the initialiser will be executed on all ndarrays which resolve to a match in a certain layer and match identifier. 
		/// </summary>
		IReadOnlyDictionary<string, IInitialiser> Initialisers { get; }

		/// <summary>
		/// The optimiser used in this trainer (e.g. Stochastic gradient descent, momentum - one instance per trainer).
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
		/// The additional named data iterators, apart from the primary training data iterator.
		/// </summary>
		IReadOnlyDictionary<string, IDataIterator> AdditionalNameDataIterators { get; }

		/// <summary>
		/// The hooks attached to this trainer. 
		/// </summary>
		IReadOnlyCollection<IHook> Hooks { get; }

		/// <summary>
		/// The active hooks attached to this trainer.
		/// </summary>
		IReadOnlyCollection<IPassiveHook> PassiveHooks { get; }

		/// <summary>
		/// The passive hooks attached to this trainer.
		/// </summary>
		IReadOnlyCollection<IActiveHook> ActiveHooks { get; }

		/// <summary>
		/// A registry containing all relevant sub-registries (e.g. network, layers, operator).
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Add an initialiser by registry resolve string (e.g. FC*.weights, *.weights, Layer1.biases, Layer2.*).
		/// Registry resolve notation may be used as the initialiser will be executed on all ndarrays which resolve to a match in a certain layer and match identifier. 
		/// </summary>
		/// <param name="identifier">The identifier (registry resolve string).</param>
		/// <param name="initialiser">The initialiser.</param>
		void AddInitialiser(string identifier, IInitialiser initialiser);

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
		/// Initialise this trainer and the network to be trained using the set initialisers. Set up all handlers and constructs used to run the trainer. 
		/// </summary>
		/// <param name="handler">The computation handler to initialise for (must be the interchangeable with the one used for running the network).</param>
		void Initialise(IComputationHandler handler);

		/// <summary>
		/// Run a training iteration on a prepared network (does not have to match the trainer's network but must have interchangeable architecture).
		/// Note: The network's external data inputs and outputs must already be linked and supplied. 
		/// </summary>
		/// <param name="localNetwork">The network to train.</param>
		/// <param name="handler">The computation handler to use.</param>
		void RunTrainingIteration(INetwork localNetwork, IComputationHandler handler);
	}
}
