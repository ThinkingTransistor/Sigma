/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Training;

namespace Sigma.Core.Persistence.Selectors.Trainer
{
	/// A default network selector, assuming the availability of a Trainer(string name) constructor.
	public class DefaultTrainerSelector<TTrainer> : BaseTrainerSelector<TTrainer> where TTrainer : ITrainer 
	{
		/// <summary>
		/// Create a base trainer selector for a certain trainer.
		/// Note: The trainer type must have a Trainer(string name) constructor.
		/// </summary>
		/// <param name="trainer">The type of the selected trainer.</param>
		public DefaultTrainerSelector(TTrainer trainer) : base(trainer)
		{
		}

		/// <summary>
		/// Create a trainer with a certain name.
		/// </summary>
		/// <param name="name">The name.</param>
		/// <returns>The trainer.</returns>
		public override TTrainer CreateTrainer(string name)
		{
			return (TTrainer) Activator.CreateInstance(Result.GetType(), name);
		}

		/// <summary>
		/// Create a selector for a certain trainer.
		/// </summary>
		/// <param name="trainer">The trainer.</param>
		/// <returns>The selector.</returns>
		public override ITrainerSelector<TTrainer> CreateSelector(TTrainer trainer)
		{
			return new DefaultTrainerSelector<TTrainer>(trainer);
		}
	}
}
