/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before Execute is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	public interface IHook
	{
		ITimeStep TimeStep { get; }

		ISet<string> RequiredRegistryEntries { get; }

		void Execute(IRegistry registry);		
	}

	/// <summary>
	/// An active hook that actively influences the calling operator by modifying the given parameters.
	/// Note: The distinction between "active" and "passive" depends on the kind of hook. 
	///		  For example a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IActiveHook : IHook
	{
	}

	/// <summary>
	/// A passive hook that passively executes code independent from the operand (without modifying the given parameters or dependency on the operator). 
	/// Note: The distinction between "active" and "passive" depends on the kind of hook. 
	///		  For example a hook that gets a networks weights and visualises them in a monitor is passive, as it does not influence the operator. 
	///		  A hook that stops the training process after a certain epoch or decreases the learning rate every update is active, as it actively influences the operator. 	
	/// </summary>
	public interface IPassiveHook : IHook
	{
	}

	/// <summary>
	/// The default implementation of the <see cref="IHook"/> interface.
	/// Represents a hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before Execute is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	public abstract class Hook : IHook
	{
		public ITimeStep TimeStep { get; private set; }

		public ISet<string> RequiredRegistryEntries { get; private set; }

		public abstract void Execute(IRegistry registry);

		public Hook(ITimeStep timestep, ISet<string> requiredRegistryEntries)
		{
			if (timestep == null)
			{
				throw new ArgumentNullException("Timestep cannot be null.");
			}

			if (requiredRegistryEntries == null)
			{
				throw new ArgumentNullException("Required registry entries cannot be null.");
			}

			this.TimeStep = timestep;
			this.RequiredRegistryEntries = requiredRegistryEntries;
		}
	}

	/// <summary>
	/// The default implementation of the <see cref="IActiveHook"/> interface.
	/// Represents an active hook that actively influences the calling operator by modifying the given parameters.
	/// </summary>
	public abstract class ActiveHook : Hook, IActiveHook
	{
		public ActiveHook(ITimeStep timestep, ISet<string> requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}
	}

	/// <summary>
	/// The default implementation of the <see cref="IPassiveHook"/> interface.
	/// A passive hook that passively executes code independent from the operand (without modifying the given parameters or dependency on the operator). 
	/// </summary>
	public abstract class PassiveHook : Hook, IPassiveHook
	{
		public PassiveHook(ITimeStep timestep, ISet<string> requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}
	}
}