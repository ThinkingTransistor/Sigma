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
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Modifiers;
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
	[Serializable]
	public class Trainer : ITrainer
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(MethodBase.GetCurrentMethod().DeclaringType);
		private readonly IList<IHook> _localHooks;
		private readonly IList<IHook> _globalHooks;
		private readonly Dictionary<string, IDataIterator> _additionalNameDataIterators;
		private readonly IList<IHook> _allHooks;
		private readonly IDictionary<string, IInitialiser> _initialisers;
		private readonly IDictionary<string, ISet<IValueModifier>> _valueModifiers;
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
		public IReadOnlyDictionary<string, ISet<IValueModifier>> ValueModifiers { get; }
		public IRegistry Registry { get; }

		public Trainer(string name)
		{
			if (name == null) throw new ArgumentNullException(nameof(name));

			Name = name;

			_allHooks = new List<IHook>();
			_localHooks = new List<IHook>();
			_globalHooks = new List<IHook>();
			_additionalNameDataIterators = new Dictionary<string, IDataIterator>();
			_initialisers = new Dictionary<string, IInitialiser>();
			_valueModifiers = new Dictionary<string, ISet<IValueModifier>>();

			Hooks = new ReadOnlyCollection<IHook>(_allHooks);
			GlobalHooks = new ReadOnlyCollection<IHook>(_globalHooks);
			LocalHooks = new ReadOnlyCollection<IHook>(_localHooks);
			AdditionalNameDataIterators = new ReadOnlyDictionary<string, IDataIterator>(_additionalNameDataIterators);
			ValueModifiers = new ReadOnlyDictionary<string, ISet<IValueModifier>>(_valueModifiers);
			Initialisers = new ReadOnlyDictionary<string, IInitialiser>(_initialisers);
			Registry = new Registry(tags: "trainer");
			Registry["self"] = this;
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
				throw new InvalidOperationException($"Cannot add duplicate identifier {identifier} for initialiser {initialiser}," +
													$" identifier is already bound to initialiser {_initialisers[identifier]}");
			}

			_initialisers.Add(identifier, initialiser);
		}

		public void AddValueModifier(string identifier, IValueModifier modifier)
		{
			_valueModifiers.TryGetValue(identifier, () => new HashSet<IValueModifier>()).Add(modifier);
		}

		public void AddHook(IHook hook)
		{
			if (hook.DefaultTargetMode == TargetMode.Local)
			{
				AddLocalHook(hook);
			}
			else if (hook.DefaultTargetMode == TargetMode.Global)
			{
				AddGlobalHook(hook);
			}
			else
			{
				throw new InvalidOperationException($"Ambiguous add hook call for hook {hook} with target mode {hook.DefaultTargetMode}. " +
													$"Target mode must be explicitly {nameof(TargetMode.Local)} or {nameof(TargetMode.Global)} for implicit hook add to work" +
													$" (i.e. unable to determine where to add this hook, specify it explicitly in the caller).");
			}
		}

		public void AddLocalHook(IHook hook)
		{
			if (Hooks.Contains(hook))
			{
				throw new ArgumentException($"Duplicate hook {hook}, hook already registered in this trainer ({Name}).");
			}

			_allHooks.Add(hook);
			_localHooks.Add(hook);
		}

		public void AddGlobalHook(IHook hook)
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
			List<string> orderedInitialiserIdentifiers = _initialisers.Keys.ToList();
			orderedInitialiserIdentifiers.Sort(RegistryUtils.CompareIdentifierSpecificityAscending);

			foreach (string identifier in orderedInitialiserIdentifiers)
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
			Registry["initialised"] = _initialised;
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

		public void RunTrainingIteration(INetwork localNetwork, IOptimiser localOptimiser, IRegistry localRegistry, IComputationHandler handler)
		{
			if (localNetwork == null) throw new ArgumentNullException(nameof(localNetwork));
			if (localOptimiser == null) throw new ArgumentNullException(nameof(localOptimiser));
			if (localRegistry == null) throw new ArgumentNullException(nameof(localRegistry));
			if (handler == null) throw new ArgumentNullException(nameof(handler));

			CheckInitialised();

			localOptimiser.PrepareRun(localNetwork, handler);
			localNetwork.Run(handler, trainingPass: true);
			localOptimiser.Run(localNetwork, handler);
			ApplyValueModifiers(localRegistry, handler);
		}

		private void ApplyValueModifiers(IRegistry localRegistry, IComputationHandler handler)
		{
			if (_valueModifiers.Count == 0)
			{
				return;
			}

			RegistryResolver resolver = new RegistryResolver(localRegistry);

			foreach (string identifier in _valueModifiers.Keys)
			{
				string[] fullyResolvedIdentifiers;
				object[] values = resolver.ResolveGet<object>(identifier, out fullyResolvedIdentifiers);

				for (int i = 0; i < values.Length; i++)
				{
					object value = values[i];
					INDArray asNDArray = value as INDArray;
					INumber asNumber = value as INumber;

					if (asNDArray != null)
					{
						foreach (IValueModifier modifier in _valueModifiers[identifier])
						{
							asNDArray = modifier.Modify(fullyResolvedIdentifiers[i], asNDArray, handler);
						}
						values[i] = asNDArray;
					}
					else if (asNumber != null)
					{
						foreach (IValueModifier modifier in _valueModifiers[identifier])
						{
							asNumber = modifier.Modify(fullyResolvedIdentifiers[i], asNumber, handler);
						}
						values[i] = asNumber;
					}
					else
					{
						double? asDouble = value as double?;

						if (asDouble != null)
						{
							foreach (IValueModifier modifier in _valueModifiers[identifier])
							{
								asDouble = modifier.Modify(fullyResolvedIdentifiers[i], asDouble.Value, handler);
							}
							values[i] = asDouble.Value;
						}
					}

					resolver.ResolveSet(fullyResolvedIdentifiers[i], values[i]);
				}
			}
		}

		/// <summary>
		/// Provide the external data to a network given the current record block (typically as given by the training data iterator).
		/// </summary>
		/// <param name="localNetwork">The network to provide the data with.</param>
		/// <param name="currentBlock">The current record block.</param>
		public void ProvideExternalInputData(INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			CheckInitialised();

			DataProviderUtils.ProvideExternalInputData(DataProvider, localNetwork, currentBlock);
		}

		/// <summary>
		/// Provide the external output data from network to the data provider.
		/// </summary>
		/// <param name="localNetwork">The network to get the data from.</param>
		/// <param name="currentBlock">The current record block.</param>
		public void ProvideExternalOutputData(INetwork localNetwork, IDictionary<string, INDArray> currentBlock)
		{
			DataProviderUtils.ProvideExternalOutputData(DataProvider, localNetwork, currentBlock);
		}

		/// <summary>
		/// Reset this trainer to an un-initialised state, discard all progress information. If necessary, stop the operator (and wait for that).
		/// </summary>
		public void Reset()
		{
			_logger.Info($"Resetting trainer \"{Name}\" to un-initialised state, discarding all progress data...");

			if (Operator?.State != ExecutionState.None)
			{
				_logger.Info($"Signalling operator to stop and reset, waiting for state change signal to continue trainer reset...");

				Operator.SignalReset();
				Operator.WaitForStateChanged();
			}

			Network?.Reset();
			_initialised = false;

			UpdateRegistry();

			_logger.Info($"Done resetting trainer \"{Name}\" to un-initialised state, discarded all progress data and stopped operator.");
		}

		public override string ToString()
		{
			return $"trainer \"{Name}\"";
		}
	}
}