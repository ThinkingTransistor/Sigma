/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Sigma.Core.Training.Operators;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// The base implementation of the <see cref="IHook"/> interface.
	/// Represents a hook which can be used to "hook" into operations and execute custom code at a certain time step. 
	/// The required parameters from the callers registry must be denoted before <see cref="Invoke"/> is first called (so that the operator can fetch the requested parameters).
	/// </summary>
	[Serializable]
	public abstract class BaseHook : IHook
	{
		private readonly IList<IHook> _requiredHooks;
		private readonly IList<string> _requiredRegistryEntries;

		/// <summary>
		/// The time step at which to execute this hook.
		/// </summary>
		public ITimeStep TimeStep { get; }

		/// <summary>
		/// The global registry entries required for the execution of this hook.
		/// </summary>
		public IReadOnlyCollection<string> RequiredRegistryEntries { get; }

		/// <summary>
		/// The hooks that are required for this hook (i.e. the hooks this hook depends on).
		/// Required hooks are prioritised and executed before the dependent hook.
		/// If multiple required hooks are functionally equivalent, only one will be invoked. 
		/// </summary>
		public IReadOnlyCollection<IHook> RequiredHooks { get; }

		/// <summary>
		/// The invoke priority of this hook (i.e. invoke as first or last hook).
		/// An invoke priority of 0 is the default, smaller than 0 means invoke earlier, larger than 0 means invoke later.
		/// Note:	The <see cref="IHook.RequiredHooks"/> take precedence over invoke priority.
		///			Invoke priorities are only a recommendation and cannot be guaranteed. 
		/// </summary>
		public int InvokePriority { get; protected set; } = 0;

		/// <summary>
		/// Flag whether this hook should be invoked by the owner (worker/operator) or in a separate background task.
		/// By default this is set to false (no background execution). Hooks that are not invoked in the background should execute as fast as possible. Otherwise...
		/// Set this to true if performance intensive operations (e.g. storing to disk, processing large arrays) are used in this hook.
		/// Note: When invoked in background, hooks received a complete copy of all required registry entries and can therefore not directly modify the parameters of a worker/operator.
		/// </summary>
		public bool InvokeInBackground { get; protected set; } = false;

		/// <summary>
		/// The operator that owns this hook and dispatched it for execution. 
		/// </summary>
		public IOperator Operator { get; set; }

		/// <summary>
		/// The default target mode of this hook (i.e. where to invoke it if not explicitly specified).
		/// </summary>
		public TargetMode DefaultTargetMode { get; protected set; } = TargetMode.Any;

		/// <summary>
		/// The internal parameter registry of this hook.
		/// </summary>
		public IRegistry ParameterRegistry { get; }

		/// <summary>
		/// The optional hook invoke criteria.
		/// </summary>
		public HookInvokeCriteria InvokeCriteria { get; private set; }

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseHook(ITimeStep timestep, params string[] requiredRegistryEntries) : this(timestep, new HashSet<string>(requiredRegistryEntries))
		{
		}

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseHook(ITimeStep timestep, ISet<string> requiredRegistryEntries)
		{
			if (timestep == null)
			{
				throw new ArgumentNullException(nameof(timestep));
			}

			if (requiredRegistryEntries == null)
			{
				throw new ArgumentNullException(nameof(requiredRegistryEntries));
			}

			TimeStep = timestep;
			_requiredHooks = new List<IHook>();
			_requiredRegistryEntries = new List<string>(requiredRegistryEntries.ToArray());

			RequiredRegistryEntries = new ReadOnlyCollection<string>(_requiredRegistryEntries);
			RequiredHooks = new ReadOnlyCollection<IHook>(_requiredHooks);
			ParameterRegistry = new Registry();
		}

		/// <summary>
		/// Invoke this hook only when a certain hook invoke criteria is satisfied.
		/// </summary>
		/// <param name="criteria"></param>
		/// <returns></returns>
		public BaseHook On(HookInvokeCriteria criteria)
		{
			if (criteria == null) throw new ArgumentNullException(nameof(criteria));

			InvokeCriteria = criteria;

			return this;
		}

		/// <summary>
		/// Require one or more registry entries (to be added to the required hooks field).
		/// Note: This is only meant to be used within the constructor, invocation in other areas may lead to illegal / inconsistent state.
		/// </summary>
		/// <param name="requiredRegistryEntries">The required registry entries.</param>
		protected BaseHook RequireRegistryEntry(params string[] requiredRegistryEntries)
		{
			if (requiredRegistryEntries == null) throw new ArgumentNullException(nameof(requiredRegistryEntries));

			foreach (string entry in requiredRegistryEntries)
			{
				_requiredRegistryEntries.Add(entry);
			}

			return this;
		}

		/// <summary>
		/// Require one or more hooks (to be added to the required hooks field).
		/// Note: This is only meant to be used within the constructor, invocation in other areas may lead to illegal / inconsistent state.
		/// </summary>
		/// <param name="requiredHooks">The required hooks.</param>
		protected BaseHook RequireHook(params IHook[] requiredHooks)
		{
			if (requiredHooks == null) throw new ArgumentNullException(nameof(requiredHooks));

			foreach (IHook hook in requiredHooks)
			{
				_requiredHooks.Add(hook);
			}

			return this;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			if (InvokeCriteria == null || InvokeCriteria.CheckCriteria(registry, resolver))
			{
				SubInvoke(registry, resolver);
			}
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public abstract void SubInvoke(IRegistry registry, IRegistryResolver resolver);

		/// <summary>
		/// Check if this hook's functionality is equal to that of another. 
		/// Used when deciding which hooks can be omitted (optimised out).
		/// Note: Different parameters typically infer different functionalities, here only the parameters in the <see cref="ParameterRegistry"/> are checked.
		///		  If your custom hook for any reason requires any additional parameters that alter its behaviour you should add your own checks to this method.
		/// </summary>
		/// <param name="other">The hook to check.</param>
		/// <returns>A boolean indicating whether or not the other hook does the same that this one does.</returns>
		public bool FunctionallyEquals(IHook other)
		{
			if (other == null)
			{
				throw new ArgumentNullException(nameof(other));
			}

			BaseHook otherAsBaseHook = other as BaseHook;
			bool otherParametersEqual = otherAsBaseHook == null || ParameterRegistry.RegistryContentEquals(otherAsBaseHook.ParameterRegistry);

			return GetType() == other.GetType() && TimeStep.StepEquals(other.TimeStep) && otherParametersEqual;
		}
	}

	/// <summary>
	/// The base implementation of the <see cref="ICommand"/> interface.
	/// A local hook that is only invoked one time on global and local scope(s), regardless of TimeStep (though live time should be 1 for consistency).
	/// </summary>
	public abstract class BaseCommand : BaseHook, ICommand
	{
		/// <summary>
		/// The time step for commands (intermediate time scale, interval and live time are 1 for single invocation).
		/// </summary>
		protected static readonly TimeStep TimeStepCommand = new TimeStep(TimeScale.Indeterminate, 1, 1);

		/// <summary>
		/// The callback method that will be executed when the command finishes execution.
		/// <c>null</c> if not required.
		/// </summary>
		public Action OnFinish { get; set; }

		/// <summary>
		/// Create a command with a set of required global registry entries.
		/// </summary>
		/// <param name="onFinish">Set a callback for when the command is finsihed. <c>null</c> if not required.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseCommand(Action onFinish = null, params string[] requiredRegistryEntries) : base(TimeStepCommand, requiredRegistryEntries)
		{
			OnFinish = onFinish;
		}

		/// <summary>
		/// Create a command with a set of required global registry entries.
		/// </summary>
		/// <param name="onFinish">Set a callback for when the command is finsihed. <c>null</c> if not required.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		protected BaseCommand(ISet<string> requiredRegistryEntries, Action onFinish = null) : base(TimeStepCommand, requiredRegistryEntries)
		{
			OnFinish = onFinish;
		}
	}
}
