/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reflection;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training
{
	/// <summary>
	/// The default <see cref="ITrainer"/> implementation.
	/// A trainer that defines how a network should be trained, denoting initialisers, optimiser, operator, custom hooks and data to apply and use. 
	/// </summary>
	public class Trainer : ITrainer
	{
		private readonly ILog _logger = LogManager.GetLogger(MethodBase.GetCurrentMethod().DeclaringType);
		private readonly IList<IHook> _localHooks;
		private readonly IList<IHook> _globalHooks;
		private readonly Dictionary<string, IDataIterator> _additionalNameDataIterators;
		private readonly IList<IHook> _allHooks;
		private readonly IDictionary<string, IInitialiser> _initialisers;
		private bool _initialised;

		public string Name { get; }
		public SigmaEnvironment Sigma { get; set; }
		public INetwork Network { get; set; }
		public IOptimiser Optimiser { get; set; }
		public IOperator Operator { get; set; } = new CpuSinglethreadedOperator();
		public IDataProvider DataProvider { get; set; } = new DefaultDataProvider();

		public IReadOnlyDictionary<string, IInitialiser> Initialisers { get; }
		public IDataIterator TrainingDataIterator { get; set; }
		public IReadOnlyDictionary<string, IDataIterator> AdditionalNameDataIterators { get; }
		public IReadOnlyCollection<IHook> Hooks { get; }
		public IReadOnlyCollection<IHook> GlobalHooks { get; }
		public IReadOnlyCollection<IHook> LocalHooks { get; }
		public IRegistry Registry { get; }

		internal Trainer(string name)
		{
			if (name == null) { throw new ArgumentNullException(nameof(name)); }

			Name = name;

			_allHooks = new List<IHook>();
			_localHooks = new List<IHook>();
			_globalHooks = new List<IHook>();
			_additionalNameDataIterators = new Dictionary<string, IDataIterator>();
			_initialisers = new Dictionary<string, IInitialiser>();

			Hooks = new ReadOnlyCollection<IHook>(_allHooks);
			GlobalHooks = new ReadOnlyCollection<IHook>(_globalHooks);
			LocalHooks = new ReadOnlyCollection<IHook>(_localHooks);
			AdditionalNameDataIterators = new ReadOnlyDictionary<string, IDataIterator>(_additionalNameDataIterators);
			Initialisers = new ReadOnlyDictionary<string, IInitialiser>(_initialisers);
			Registry = new Registry(tags: "trainer");
		}

		public void AddNamedDataIterator(string name, IDataIterator iterator)
		{
			if (_additionalNameDataIterators.ContainsKey(name))
			{
				throw new ArgumentException($"Named data iterator with name {name} already registered in this trainer ({Name}).");
			}

			_additionalNameDataIterators.Add(name, iterator);
		}

		public void AddInitialiser(string identifier, IInitialiser initialiser)
		{
			if (identifier == null) { throw new ArgumentNullException(nameof(identifier)); }
			if (initialiser == null) { throw new ArgumentNullException(nameof(initialiser)); }

			if (_initialisers.ContainsKey(identifier))
			{
				throw new InvalidOperationException($"Cannot add duplicate identifier {identifier} for initialiser {initialiser}, identifier is already bound to initialiser {_initialisers[identifier]}");
			}

			_initialisers.Add(identifier, initialiser);
		}

		public void AddGlobalHook(IHook hook)
		{
			if (Hooks.Contains(hook))
			{
				throw new ArgumentException($"Duplicate hook {hook}, hook already registered in this trainer ({Name}).");
			}

			_allHooks.Add(hook);
			_localHooks.Add(hook);
		}

		public void AddLocalHook(IHook hook)
		{
			if (Hooks.Contains(hook))
			{
				throw new ArgumentException($"Duplicate hook {hook}, hook already registered in this trainer ({Name}).");
			}

			_allHooks.Add(hook);
			_globalHooks.Add(hook);
		}

		public void Initialise(IComputationHandler handler)
		{
			ValidateAssignedComponents();

			_logger.Info($"Initialising trainer \"{Name}\" with handler {handler}...");

			ITaskObserver prepareTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare, $"Preparing trainer {Name}");

			Network.Initialise(handler);

			RegistryResolver networkResolver = new RegistryResolver(Network.Registry.Get<IRegistry>("layers"));
			int initialisedNDArrayCount = 0, initialisedNumberCount = 0;

			// TODO maybe sort by most specific ascending?
			foreach (string identifier in _initialisers.Keys)
			{
				object[] values = networkResolver.ResolveGet(identifier, new object[0]);
				IInitialiser initialiser = _initialisers[identifier];

				foreach (object value in values)
				{
					INDArray array = value as INDArray;

					if (array != null)
					{
						initialiser.Initialise(array, handler, Sigma.Random);
						initialisedNDArrayCount++;
					}
					else
					{
						INumber number = value as INumber;

						if (number != null)
						{
							initialiser.Initialise(number, handler, Sigma.Random);
							initialisedNumberCount++;
						}
					}
				}
			}

			Operator.Sigma = Sigma;
			Operator.Handler = Operator.Handler ?? handler;
			Operator.Network = Network;
			Operator.Trainer = this;

			// attach all given hooks
			foreach (IHook hook in _globalHooks)
			{
				if (!Operator.AttachGlobalHook(hook))
				{
					_logger.Debug($"Skipped attaching global hook {hook} in trainer \"{Name}\", operator refused to attach it.");
				}
			}

			foreach (IHook hook in _localHooks)
			{
				if (!Operator.AttachLocalHook(hook))
				{
					_logger.Debug($"Skipped attaching local hook {hook} in trainer \"{Name}\", operator refused to attach it.");
				}
			}

			UpdateRegistry();

			_initialised = true;

			SigmaEnvironment.TaskManager.EndTask(prepareTask);

			_logger.Info($"Done initialising trainer \"{Name}\" for handler {handler}, initialised {initialisedNDArrayCount} ndarrays and {initialisedNumberCount} numbers.");
		}

		protected virtual void UpdateRegistry()
		{
			Registry["name"] = Name;
			Registry["network"] = Network?.Registry;
			Registry["optimiser"] = Optimiser?.Registry;

			Registry initialiserRegistry = new Registry(Registry, tags: "initialisers");
			Registry["initialisers"] = initialiserRegistry;

			foreach (string initialiserMatchIdentifier in Initialisers.Keys)
			{
				initialiserRegistry[initialiserMatchIdentifier] = Initialisers[initialiserMatchIdentifier];
			}
		}

		private void ValidateAssignedComponents()
		{
			if (Network == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer {Name} before assigning a network.");
			}

			if (Sigma == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer {Name} before assigning a sigma environment.");
			}

			if (Operator == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer {Name} before assigning an operator.");
			}

			if (DataProvider == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer {Name} before assigning a data provider.");
			}

			Network.Validate();
		}

		private void CheckInitialised()
		{
			if (!_initialised)
			{
				throw new InvalidOperationException($"Trainer {Name} has not been initialised yet. Call {nameof(Initialise)}!");
			}
		}

		public void Start()
		{
			_logger.Info($"Validating trainer state of trainer {Name} before start...");

			ValidateAssignedComponents();
			CheckInitialised();

			_logger.Info($"Starting operator {Operator} with trainer {Name}...");

			Operator.Network = Network;
			Operator.Trainer = this;

			Operator.Start();
		}

		public void RunTrainingIteration(INetwork localNetwork, IOptimiser localOptimiser, IComputationHandler handler)
		{
			CheckInitialised();

			localOptimiser.PrepareRun(localNetwork, handler);
			localNetwork.Run(handler, trainingPass: true);
			localOptimiser.Run(localNetwork, handler);
		}

		/// <summary>
		/// Provide the external data to a network given the current record block (typically as given by the training data iterator).
		/// </summary>
		/// <param name="localNetwork">The network to provide the data with.</param>
		/// <param name="currentBlock">The current record block.</param>
		public void ProvideExternalData(INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			CheckInitialised();

			foreach (ILayerBuffer layerBuffer in localNetwork.YieldExternalInputsLayerBuffers())
			{
				foreach (string externalInputAlias in layerBuffer.ExternalInputs)
				{
					DataProvider.ProvideExternalInput(externalInputAlias, layerBuffer.Inputs[externalInputAlias], layerBuffer.Layer, currentBlock);
				}
			}

			foreach (ILayerBuffer layerBuffer in localNetwork.YieldExternalOutputsLayerBuffers())
			{
				foreach (string externalOutputAlias in layerBuffer.ExternalOutputs)
				{
					DataProvider.ProvideExternalOutput(externalOutputAlias, layerBuffer.Outputs[externalOutputAlias], layerBuffer.Layer, currentBlock);
				}
			}
		}
	}
}