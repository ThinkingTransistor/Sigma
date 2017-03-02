/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Training.Hooks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility methods for hook management. 
	/// </summary>
	public static class HookUtils
	{
		/// <summary>
		/// Get a registry copy containing a copy of all by the given hooks required values.
		/// </summary>
		/// <param name="registry">The registry root to copy from.</param>
		/// <param name="hooks">The hooks.</param>
		/// <param name="bufferRegistryEntries">The optional buffer to use for temporarily storing unresolved registry entries.</param>
		/// <param name="bufferResolvedRegistryEntries">The optional buffer to use for temporarily storing resolved registry entries.</param>
		/// <returns>A registry copy containing a copy of all by the given hooks required values.</returns>
		public static IRegistry GetRegistryCopyForHooks(IRegistry registry, IEnumerable<IHook> hooks, ISet<string> bufferRegistryEntries = null, ISet<string> bufferResolvedRegistryEntries = null)
		{
			if (bufferRegistryEntries == null)
			{
				bufferRegistryEntries = new HashSet<string>();
			}

			if (bufferResolvedRegistryEntries == null)
			{
				bufferResolvedRegistryEntries = new HashSet<string>();
			}

			IRegistryResolver resolver = new RegistryResolver(registry);


			FetchAllRequiredRegistryEntries(hooks, bufferRegistryEntries);
			ResolveAllRequiredRegistry(resolver, bufferRegistryEntries, bufferResolvedRegistryEntries);

			return GetRegistryCopyForResolvedEntries(registry, bufferResolvedRegistryEntries);
		}

		/// <summary>
		/// Get a registry copy containing a copy of given required values.
		/// </summary>
		/// <param name="registry">The registry.</param>
		/// <param name="resolvedRegistryEntries">The RESOLVED registry entries.</param>
		/// <returns>A registry copy containing a copy of given required values.</returns>
		public static IRegistry GetRegistryCopyForResolvedEntries(IRegistry registry, ISet<string> resolvedRegistryEntries)
		{
			IRegistry rootCopy = new Registry(tags: registry.Tags.ToArray());

			foreach (string entry in resolvedRegistryEntries)
			{
				string[] parts = entry.Split('.');

				IRegistry currentRoot = registry;
				IRegistry currentRootCopy = rootCopy;

				for (int i = 0; i < parts.Length - 1; i++)
				{
					string part = parts[i];

					if (!currentRoot.ContainsKey(part))
					{
						throw new InvalidOperationException($"Cannot access non-existing registry \"{part}\" from full entry \"{entry}\" (level {i}).");
					}

					IRegistry nextRoot = currentRoot[part] as IRegistry;

					if (nextRoot == null)
					{
						throw new InvalidOperationException($"Cannot access non-registry entry \"{part}\" from full entry \"{entry}\" (level {i}), should be instance of {nameof(IRegistry)}.");
					}

					if (!currentRootCopy.ContainsKey(part))
					{
						currentRootCopy[part] = new Registry(parent: currentRoot, tags: nextRoot.Tags.ToArray());
					}

					currentRoot = nextRoot;
				}

				string lastPart = parts[parts.Length - 1];
				object copiedValue = RegistryUtils.DeepestCopy(currentRoot[lastPart]);

				currentRootCopy[lastPart] = copiedValue;
			}

			return rootCopy;
		}

		/// <summary>
		/// Resolve all registry entries using a given registry resolver.
		/// </summary>
		/// <param name="registryResolver">The registry resolver to use.</param>
		/// <param name="allRegistryEntries">The registry entries to resolve.</param>
		/// <param name="resultAllResolvedRegistryEntries">The resulting resolved registry entries.</param>
		public static void ResolveAllRequiredRegistry(IRegistryResolver registryResolver, IEnumerable<string> allRegistryEntries, ISet<string> resultAllResolvedRegistryEntries)
		{
			resultAllResolvedRegistryEntries.Clear();

			foreach (string registryEntry in allRegistryEntries)
			{
				string[] resolvedEntries;

				registryResolver.ResolveGet<object>(registryEntry, out resolvedEntries, null);
			}
		}

		/// <summary>
		/// Fetch all required registry entries from a set of hooks.
		/// </summary>
		/// <param name="hooks">The hooks.</param>
		/// <param name="resultAllRequiredRegistryEntries">The resulting required registry entries.</param>
		public static void FetchAllRequiredRegistryEntries(IEnumerable<IHook> hooks, ISet<string> resultAllRequiredRegistryEntries = null)
		{
			if (resultAllRequiredRegistryEntries == null)
			{
				resultAllRequiredRegistryEntries = new HashSet<string>();
			}
			else
			{
				resultAllRequiredRegistryEntries.Clear();
			}

			foreach (IHook hook in hooks)
			{
				foreach (string registryEntry in hook.RequiredRegistryEntries)
				{
					if (!resultAllRequiredRegistryEntries.Contains(registryEntry))
					{
						resultAllRequiredRegistryEntries.Add(registryEntry);
					}
				}
			}
		}

		/// <summary>
		/// Fetch all hooks that should be invoke in background from a set of given hooks (given order is retained).
		/// </summary>
		/// <param name="hooks">The hooks to fetch from.</param>
		/// <param name="resultHooks">The resulting set of hooks to invoke in background.</param>
		public static void FetchOrderedBackgroundHooks(IEnumerable<IHook> hooks, IList<IHook> resultHooks)
		{
			resultHooks.Clear();
			foreach (IHook hook in hooks)
			{
				if (hook.InvokeInBackground)
				{
					resultHooks.Add(hook);
				}
			}
		}

		/// <summary>
		/// Validate a hook and its dependencies.
		/// </summary>
		/// <param name="hook">The hook to evaluate.</param>
		public static void ValidateHook(IHook hook)
		{
			if (hook == null) throw new ArgumentNullException(nameof(hook));
			if (hook.TimeStep == null) throw new ArgumentException($"Hook {hook} has invalid time step: null");
			if (hook.RequiredHooks == null) throw new ArgumentException($"Hook {hook} has invalid required hooks field: null");
			if (hook.RequiredRegistryEntries == null) throw new ArgumentException($"Hook {hook} has invalid required registry entries field: null");

			foreach (IHook requiredHook in hook.RequiredHooks)
			{
				IHook culprit;
				if (HasCircularDependency(requiredHook, requiredHook, out culprit))
				{
					throw new IllegalHookDependencyException($"Hook {hook} has illegal dependencies, detected circular dependency of required hook {requiredHook} via {culprit}.");
				}

				if (requiredHook.InvokeInBackground != hook.InvokeInBackground)
				{
					throw new IllegalHookDependencyException($"Hook {hook} has inconsistent dependencies, {nameof(IHook.InvokeInBackground)} field must be the same for all required hooks, " +
															 $"but field was {requiredHook.InvokeInBackground} in the required hook and {hook.InvokeInBackground} in the dependent hook.");
				}

				if (requiredHook.TimeStep.CompareTo(hook.TimeStep) > 0)
				{
					throw new IllegalHookDependencyException($"Hook {hook} has illegal dependencies, detected illegal time step dependency with other hook {requiredHook}. " +
					                                         $"Time step of required hook must be <= than dependent hook time step, but required hook time step was {requiredHook.TimeStep}" +
					                                         $" and dependent hook time step was {hook.TimeStep}.");
				}
			}
		}

		private static bool HasCircularDependency(IHook current, IHook root, out IHook culprit)
		{
			foreach (IHook requiredHook in current.RequiredHooks)
			{
				if (requiredHook.FunctionallyEquals(root))
				{
					culprit = requiredHook;

					return true;
				}
				else if (HasCircularDependency(requiredHook, root, out culprit))
				{
					return true;
				}
			}

			culprit = null;

			return false;
		}

		/// <summary>
		/// Get the current interval of a certain time scale out of a registry.
		/// Note: Time scale can only be epoch or iteration.
		/// </summary>
		/// <param name="registry">The registry.</param>
		/// <param name="timeScale">The time scale.</param>
		/// <returns>The current interval of the given time scale as it is in the given registry.</returns>
		public static int GetCurrentInterval(IRegistry registry, TimeScale timeScale)
		{
			if (registry == null) throw new ArgumentNullException(nameof(registry));

			if (timeScale == TimeScale.Epoch)
			{
				return registry.Get<int>("epoch");
			}
			else if (timeScale == TimeScale.Iteration)
			{
				return registry.Get<int>("iteration");
			}
			else
			{
				throw new ArgumentException($"Cannot get current interval of time scale {timeScale}, must be either {nameof(TimeScale.Epoch)} or {nameof(TimeScale.Iteration)}.");
			}
		}
	}
}
