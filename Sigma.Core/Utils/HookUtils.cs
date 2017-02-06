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
				object copiedValue = Registry.DeepestCopy(currentRoot[lastPart]);

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
		/// Fetch all hooks that should be invoke in background from a set of given hooks.
		/// </summary>
		/// <param name="hooks">The hooks to fetch from.</param>
		/// <param name="resultHooks">The resulting set of hooks to invoke in background.</param>
		public static void FetchBackgroundHooks(IEnumerable<IHook> hooks, ISet<IHook> resultHooks)
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
	}
}
