/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training
{
	public class Trainer : ITrainer
	{
		public string Name { get; }
		public SigmaEnvironment Sigma { get; set; }
		public INetwork Network { get; set; }
		public IOptimiser Optimiser { get; set; }
		public IOperator Operator { get; set; }

		public IReadOnlyDictionary<string, IInitialiser> Initialisers { get; }
		public IDataIterator TrainingDataIterator { get; set; }
		public IReadOnlyDictionary<string, IDataIterator> AdditionalNameDataIterators { get; }
		public IReadOnlyCollection<IHook> Hooks { get; }
		public IReadOnlyCollection<IPassiveHook> PassiveHooks { get; }
		public IReadOnlyCollection<IActiveHook> ActiveHooks { get; }
		public IRegistry Registry { get; }

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private readonly IList<IHook> _allHooks;
		private readonly IList<IActiveHook> _activeHooks;
		private readonly IList<IPassiveHook> _passiveHooks;
		private readonly Dictionary<string, IDataIterator> _additionalNameDataIterators;
		private readonly IDictionary<string, IInitialiser> _initialisers;

		public Trainer(string name)
		{
			if (name == null) throw new ArgumentNullException(nameof(name));

			Name = name;

			_allHooks = new List<IHook>();
			_activeHooks = new List<IActiveHook>();
			_passiveHooks = new List<IPassiveHook>();
			_additionalNameDataIterators = new Dictionary<string, IDataIterator>();
			_initialisers = new Dictionary<string, IInitialiser>();

			Hooks = new ReadOnlyCollection<IHook>(_allHooks);
			PassiveHooks = new ReadOnlyCollection<IPassiveHook>(_passiveHooks);
			ActiveHooks = new ReadOnlyCollection<IActiveHook>(_activeHooks);
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
			if (identifier == null) throw new ArgumentNullException(nameof(identifier));
			if (initialiser == null) throw new ArgumentNullException(nameof(initialiser));

			if (_initialisers.ContainsKey(identifier))
			{
				throw new InvalidOperationException($"Cannot add duplicate identifier {identifier} for initialiser {initialiser}, identifier is already bound to initialiser {_initialisers[identifier]}");
			}

			_initialisers.Add(identifier, initialiser);
		}

		public void AddActiveHook(IActiveHook hook)
		{
			if (Hooks.Contains(hook))
			{
				throw new ArgumentException($"Duplicate hook {hook}, hook already registered in this trainer ({Name}).");
			}

			_allHooks.Add(hook);
			_activeHooks.Add(hook);
		}

		public void AddPassiveHook(IPassiveHook hook)
		{
			if (Hooks.Contains(hook))
			{
				throw new ArgumentException($"Duplicate hook {hook}, hook already registered in this trainer ({Name}).");
			}

			_allHooks.Add(hook);
			_passiveHooks.Add(hook);
		}

		public void Initialise(IComputationHandler handler)
		{
			if (Network == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer before assigning a network.");
			}

			if (Sigma == null)
			{
				throw new InvalidOperationException($"Cannot initialise trainer before assigning a sigma environment.");
			}

			Network.Validate();

			_logger.Info($"Initialising trainer \"{Name}\" for handler {handler}...");

			ITaskObserver prepareTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare, "Preparing trainer");

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

			SigmaEnvironment.TaskManager.EndTask(prepareTask);

			_logger.Info($"Done initialising trainer \"{Name}\" for handler {handler}, initialised {initialisedNDArrayCount} ndarrays and {initialisedNumberCount} numbers.");
		}

		public void RunTrainingIteration(INetwork localNetwork, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}
	}
}
