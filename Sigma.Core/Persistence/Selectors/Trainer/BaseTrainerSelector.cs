/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Modifiers;

namespace Sigma.Core.Persistence.Selectors.Trainer
{
	/// <summary>
	/// A base trainer selector that handles basic <see cref="TrainerComponent"/> selection independent of actual trainer type.
	/// </summary>
	/// <typeparam name="TTrainer">The type of the selected trainer.</typeparam>
	public abstract class BaseTrainerSelector<TTrainer> : ITrainerSelector<TTrainer> where TTrainer : ITrainer
	{
		/// <inheritdoc />
		public TTrainer Result { get; }

		/// <summary>
		/// Create a base trainer selector for a certain trainer.
		/// </summary>
		/// <param name="trainer">The type of the selected trainer.</param>
		public BaseTrainerSelector(TTrainer trainer)
		{
			if (trainer == null) throw new ArgumentNullException(nameof(trainer));

			Result = trainer;
		}

		/// <inheritdoc />
		public ISelector<TTrainer> Empty()
		{
			return Keep(TrainerComponent.None);
		}

		/// <inheritdoc />
		public ISelector<TTrainer> Uninitialised()
		{
			return Keep(TrainerComponent.DataIterators, TrainerComponent.DataProvider, TrainerComponent.Hooks, TrainerComponent.Initialisers,
						TrainerComponent.Network(NetworkComponent.Architecture), TrainerComponent.Optimiser, TrainerComponent.ValueModifiers, TrainerComponent.Operator(OperatorComponent.None));
		}

		/// <inheritdoc />
		public ISelector<TTrainer> Keep(params TrainerComponent[] components)
		{
			TTrainer trainer = CreateTrainer(Result.Name);

			// TODO also check by id flags, maybe write extension method for dictionary that also checks for selector component ids

			if (components.Contains(TrainerComponent.DataIterators))
			{
				trainer.TrainingDataIterator = Result.TrainingDataIterator;

				foreach (var namedIterator in Result.AdditionalNameDataIterators)
				{
					trainer.AddNamedDataIterator(namedIterator.Key, namedIterator.Value);
				}
			}

			if (components.Contains(TrainerComponent.DataProvider))
			{
				trainer.DataProvider = Result.DataProvider;
			}

			if (components.Contains(TrainerComponent.Hooks))
			{
				foreach (IHook hook in Result.LocalHooks)
				{
					trainer.AddLocalHook(hook);
				}

				foreach (IHook hook in Result.GlobalHooks)
				{
					trainer.AddGlobalHook(hook);
				}
			}

			if (components.Contains(TrainerComponent.Initialisers))
			{
				foreach (var identifierInitialiser in Result.Initialisers)
				{
					trainer.AddInitialiser(identifierInitialiser.Key, identifierInitialiser.Value);
				}
			}

			if (components.Contains(TrainerComponent.ValueModifiers))
			{
				foreach (var identifierModifiers in Result.ValueModifiers)
				{
					foreach (IValueModifier modifier in identifierModifiers.Value)
					{
						trainer.AddValueModifier(identifierModifiers.Key, modifier);
					}
				}
			}

			if (components.Contains(TrainerComponent.Optimiser))
			{
				trainer.Optimiser = Result.Optimiser;
			}

			if (components.Contains(TrainerComponent.Network()))
			{
				// TODO add network selector implementation and select method to interface, call that here with cast sub components
			}

			var operatorComponent = TrainerComponent.Operator();
			if (components.Contains(operatorComponent))
			{
				var actualOperatorComponent = components.First(c => operatorComponent.Equals(c));

				trainer.Operator = Result.Operator.Select().Keep((OperatorComponent[]) actualOperatorComponent.SubComponents).Result;
			}

			return CreateSelector(trainer);
		}

		/// <inheritdoc />
		public ISelector<TTrainer> Discard(params TrainerComponent[] componentsToDiscard)
		{
			if (componentsToDiscard.Length == 0) throw new ArgumentException($"Discarded components must be > 0 but none were given (empty array).");

			IList<TrainerComponent> componentsToKeep = SelectorComponent.AllComponentsByIdByType[componentsToDiscard[0].GetType()].Values.Cast<TrainerComponent>().ToList();

			foreach (TrainerComponent component in componentsToDiscard)
			{
				if (!component.HasSubComponents)
				{
					componentsToKeep.Remove(component);
				}
				else
				{
					throw new NotImplementedException($"Discarding components with sub components is currently not implemented.");
					// Discarding including sub components could be done recursively by getting each types components and their ids (they use bitwise flags)
					// Taking the "All" components and negating the ones we don't want out, then getting the appropriate components via the static selector components map
					// Or alternatively just ignoring the individual ids and getting all sub components that are not the discarded components (is that logical? it's simpler..)
				}
			}

			return Keep(componentsToKeep.ToArray());
		}

		/// <summary>
		/// Create a trainer with a certain name.
		/// </summary>
		/// <param name="name">The name.</param>
		/// <returns>The trainer.</returns>
		public abstract TTrainer CreateTrainer(string name);

		/// <summary>
		/// Create a selector for a certain trainer.
		/// </summary>
		/// <param name="trainer">The trainer.</param>
		/// <returns>The selector.</returns>
		public abstract ITrainerSelector<TTrainer> CreateSelector(TTrainer trainer);
	}
}
