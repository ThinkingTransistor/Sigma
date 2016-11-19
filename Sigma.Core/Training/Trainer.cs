using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Optimisers;

namespace Sigma.Core.Training
{
	public class Trainer : ITrainer
	{
		public string Name { get; }
		public INetwork Network { get; set; }
		public IReadOnlyDictionary<string, IReadOnlyDictionary<string, IInitialiser>> Initialisers { get; set; }
		public IOptimiser Optimiser { get; set; }
		public IOperator Operator { get; set; }
		public IDataIterator TrainingDataIterator { get; set; }
		public IReadOnlyDictionary<string, IDataIterator> AdditionalNameDataIterators { get; }
		public IReadOnlyCollection<IHook> Hooks { get; }
		public IReadOnlyCollection<IPassiveHook> PassiveHooks { get; }
		public IReadOnlyCollection<IActiveHook> ActiveHooks { get; }

		private readonly IList<IHook> _allHooks;
		private readonly IList<IActiveHook> _activeHooks;
		private readonly IList<IPassiveHook> _passiveHooks;
		private readonly Dictionary<string, IDataIterator> _additionalNameDataIterators;

		public Trainer(string name)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			Name = name;

			_allHooks = new List<IHook>();
			_activeHooks = new List<IActiveHook>();
			_passiveHooks = new List<IPassiveHook>();
			_additionalNameDataIterators = new Dictionary<string, IDataIterator>();

			Hooks = new ReadOnlyCollection<IHook>(_allHooks);
			PassiveHooks = new ReadOnlyCollection<IPassiveHook>(_passiveHooks);
			ActiveHooks = new ReadOnlyCollection<IActiveHook>(_activeHooks);
			AdditionalNameDataIterators = new ReadOnlyDictionary<string, IDataIterator>(_additionalNameDataIterators);
		}

		public void AddNamedDataIterator(string name, IDataIterator iterator)
		{
			if (_additionalNameDataIterators.ContainsKey(name))
			{
				throw new ArgumentException($"Named data iterator with name {name} already registered in this trainer ({Name}).");
			}

			_additionalNameDataIterators.Add(name, iterator);
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

		public void InitialiseNetwork()
		{
			
		}
	}
}
