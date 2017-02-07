/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

#pragma warning disable 1570

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A default implementation of the registry resolver interface.
	/// A registry resolver that resolves layered identifiers. Implementations are expected but not required to cache all resolved identifiers for better performance.
	/// The supported notation syntax is:
	///		-	'.' separates registries hierarchically
	///			Example: "trainer2.training.accuracy"
	///		-	'*' indicates a wild-card mask, match any name - similar to regex's '.*'
	///			Example: "trainer*.training.accuracy" match all sub-registries whose name starts with trainer
	///		-	'*<tag>' conditionally matching wild-card mask, match any name if the conditional tag
	///			Example: "*<trainer>.training.accuracy" match all sub-registries whose tags include the tag "trainer"	
	/// </summary>
	public class RegistryResolver : IRegistryResolver, IRegistryHierarchyChangeListener
	{
		public IRegistry Root { get; }

		private readonly Dictionary<string, MatchIdentifierRequestCacheEntry> _matchIdentifierCache;
		private readonly ISet<string> _fullIdentifiersToInvalidate;

		/// <summary>
		/// Create a registry resolver with a certain root registry.
		/// </summary>
		/// <param name="root">The root registry.</param>
		public RegistryResolver(IRegistry root) : this(root, true)
		{
		}

		/// <summary>
		/// Create a registry resolver with a certain root registry.
		/// </summary>
		/// <param name="root">The root registry.</param>
		/// <param name="updateCacheEntries">This boolean decides, whether updates in the registry should also update cached entries.
		/// Normally this should be <c>true</c>.</param>
		internal RegistryResolver(IRegistry root, bool updateCacheEntries)
		{
			if (root == null)
			{
				throw new ArgumentNullException(nameof(root));
			}

			if (updateCacheEntries)
			{
				root.HierarchyChangeListeners.Add(this);
			}

			Root = root;
			_matchIdentifierCache = new Dictionary<string, MatchIdentifierRequestCacheEntry>();
			_fullIdentifiersToInvalidate = new HashSet<string>();
		}

		public void OnChildHierarchyChanged(string identifier, IRegistry previousChild, IRegistry newChild)
		{
			lock (_matchIdentifierCache)
			{
				foreach (string fullIdentifier in _matchIdentifierCache.Keys)
				{
					MatchIdentifierRequestCacheEntry entry = _matchIdentifierCache[fullIdentifier];

					if (entry.AllReferredRegistries.Contains<IRegistry>(previousChild))
					{
						_fullIdentifiersToInvalidate.Add(fullIdentifier);
					}
				}

				foreach (string fullIdentifier in _fullIdentifiersToInvalidate)
				{
					_matchIdentifierCache.Remove(fullIdentifier);
				}
			}

			RemoveChildHierarchyListener(previousChild);
			AddChildHierarchyListener(newChild);
		}

		private ISet<IRegistry> GetReferredRegistries(List<IRegistry> usedRegistries)
		{
			ISet<IRegistry> referredRegistries = new HashSet<IRegistry>();

			foreach (IRegistry usedRegistry in usedRegistries)
			{
				IRegistry referredRegistry = usedRegistry;

				do
				{
					referredRegistries.Add(referredRegistry);
				} while ((referredRegistry = referredRegistry.Parent) != null);
			}

			return referredRegistries;
		}

		private void RemoveChildHierarchyListener(IRegistry child)
		{
			if (child == null)
			{
				return;
			}

			child.HierarchyChangeListeners.Remove(this);

			foreach (object value in child.Values)
			{
				IRegistry childRegistry = value as IRegistry;
				if (childRegistry != null)
				{
					RemoveChildHierarchyListener(childRegistry);
				}
			}
		}

		private void AddChildHierarchyListener(IRegistry child)
		{
			if (child == null)
			{
				return;
			}

			child.HierarchyChangeListeners.Add(this);

			foreach (object value in child.Values)
			{
				IRegistry childRegistry = value as IRegistry;
				if (childRegistry != null)
				{
					AddChildHierarchyListener(childRegistry);
				}
			}
		}

		private static void CheckMatchIdentifier(string matchIdentifier)
		{
			if (matchIdentifier == null)
			{
				throw new ArgumentNullException(nameof(matchIdentifier));
			}

			if (matchIdentifier.Length == 0)
			{
				throw new ArgumentException("Match identifier cannot be of length 0.");
			}
		}

		public T[] ResolveGet<T>(string matchIdentifier, T[] values = null)
		{
			string[] emptyArrayThrowaway;

			return ResolveGet(matchIdentifier, out emptyArrayThrowaway, values);
		}


		public T[] ResolveGet<T>(string matchIdentifier, out string[] fullMatchedIdentifierArray, T[] values = null)
		{
			CheckMatchIdentifier(matchIdentifier);

			MatchIdentifierRequestCacheEntry cacheEntry = GetOrCreateCacheEntry(matchIdentifier);

			if (values == null || values.Length < cacheEntry.FullMatchedIdentifierArray.Length)
			{
				values = new T[cacheEntry.FullMatchedIdentifierArray.Length];
			}

			for (int i = 0; i < cacheEntry.FullMatchedIdentifierArray.Length; i++)
			{
				string fullIdentifier = cacheEntry.FullMatchedIdentifierArray[i];

				values[i] = cacheEntry.FullMatchedIdentifierRegistries[fullIdentifier].Get<T>(cacheEntry.FullMatchedIdentifierLocals[fullIdentifier]);
			}

			fullMatchedIdentifierArray = cacheEntry.FullMatchedIdentifierArray;

			return values;
		}

		public string[] ResolveSet<T>(string matchIdentifier, T value, Type associatedType = null)
		{
			CheckMatchIdentifier(matchIdentifier);

			MatchIdentifierRequestCacheEntry cacheEntry = GetOrCreateCacheEntry(matchIdentifier);

			string localIdentifier = matchIdentifier.Contains('.') ? matchIdentifier.Substring(matchIdentifier.LastIndexOf('.') + 1) : matchIdentifier;

			foreach (string fullIdentifier in cacheEntry.FullMatchedIdentifierArray)
			{
				IRegistry currentRegistry = cacheEntry.FullMatchedIdentifierRegistries[fullIdentifier];

				associatedType = associatedType ?? currentRegistry.GetAssociatedType(localIdentifier);

				currentRegistry.Set(localIdentifier, value, associatedType);
			}

			return cacheEntry.FullMatchedIdentifierArray;
		}

		private MatchIdentifierRequestCacheEntry GetOrCreateCacheEntry(string matchIdentifier)
		{
			CheckMatchIdentifier(matchIdentifier);

			lock (_matchIdentifierCache)
			{
				if (_matchIdentifierCache.ContainsKey(matchIdentifier))
				{
					return _matchIdentifierCache[matchIdentifier];
				}
			}

			string[] matchIdentifierParts = matchIdentifier.Split('.');

			ISet<string>[] conditionalTagsPerLevel = new ISet<string>[matchIdentifierParts.Length];

			for (int i = 0; i < matchIdentifierParts.Length; i++)
			{
				matchIdentifierParts[i] = ParseMatchIdentifier(i, matchIdentifierParts[i], conditionalTagsPerLevel);
			}

			Dictionary<string, IRegistry> fullMatchedIdentifierRegistries = new Dictionary<string, IRegistry>();

			MatchIdentifierRequestCacheEntry newCacheEntry = new MatchIdentifierRequestCacheEntry(matchIdentifier, fullMatchedIdentifierRegistries, new Dictionary<string, string>(), null, null);

			AddMatchingIdentifiersFromRegistryTree(0, matchIdentifierParts.Length - 1, Root, "", matchIdentifierParts, conditionalTagsPerLevel, newCacheEntry);

			newCacheEntry.FullMatchedIdentifierArray = fullMatchedIdentifierRegistries.Keys.ToArray();

			newCacheEntry.AllReferredRegistries = GetReferredRegistries(newCacheEntry.FullMatchedIdentifierRegistries.Values.ToList());

			//only cache if the last identifier level did not contain a blank unrestricted wildcard (wildcard without conditional tags, meaning we can't cache anything as it could be any value)
			int lastLevel = matchIdentifierParts.Length - 1;
			bool shouldCache = !matchIdentifierParts[lastLevel].Contains(".*") || conditionalTagsPerLevel[lastLevel]?.Count > 0;

			lock (_matchIdentifierCache)
			{
				if (shouldCache)
				{
					_matchIdentifierCache.Add(matchIdentifier, newCacheEntry);
				}
			}

			return newCacheEntry;
		}

		private void AddMatchingIdentifiersFromRegistryTree(int hierarchyLevel, int lastHierarchySearchLevel, IRegistry currentRootAtLevel, string currentFullIdentifier, string[] parsedMatchIdentifierParts, ISet<string>[] conditionalTagsPerLevel, MatchIdentifierRequestCacheEntry newCacheEntry)
		{
			Regex regex = new Regex(parsedMatchIdentifierParts[hierarchyLevel]);

			foreach (string identifier in currentRootAtLevel.Keys)
			{
				if (regex.IsMatch(identifier))
				{
					object value = currentRootAtLevel.Get(identifier);

					if (hierarchyLevel < lastHierarchySearchLevel && value is IRegistry)
					{
						IRegistry subRegistry = (IRegistry) value;

						if (RegistryMatchesAllTags(subRegistry, conditionalTagsPerLevel[hierarchyLevel]))
						{
							string nextFullIdentifier = String.IsNullOrEmpty(currentFullIdentifier) ? identifier : (currentFullIdentifier + "." + identifier);

							AddMatchingIdentifiersFromRegistryTree(hierarchyLevel + 1, lastHierarchySearchLevel, subRegistry, nextFullIdentifier, parsedMatchIdentifierParts, conditionalTagsPerLevel, newCacheEntry);
						}
					}
					else if (hierarchyLevel == lastHierarchySearchLevel)
					{
						if (value is IRegistry && !RegistryMatchesAllTags((IRegistry) value, conditionalTagsPerLevel[hierarchyLevel]))
						{
							continue;
						}

						string globalFullIdentifier = (String.IsNullOrEmpty(currentFullIdentifier) ? "" : currentFullIdentifier + ".") + identifier;

						newCacheEntry.FullMatchedIdentifierRegistries.Add(globalFullIdentifier, currentRootAtLevel);
						newCacheEntry.FullMatchedIdentifierLocals.Add(globalFullIdentifier, identifier);
					}
				}
			}
		}

		private bool RegistryMatchesAllTags(IRegistry registry, ISet<string> tags)
		{
			bool matchesAllTags = true;

			if (tags?.Count > 0)
			{
				foreach (string tag in tags)
				{
					if (!registry.Tags.Contains(tag))
					{
						matchesAllTags = false;

						break;
					}
				}
			}

			return matchesAllTags;
		}

		private string ParseMatchIdentifier(int hierarchyLevel, string partialMatchIdentifier, ISet<string>[] conditionalTagsPerLevel)
		{
			if (partialMatchIdentifier.Contains('*'))
			{
				partialMatchIdentifier = partialMatchIdentifier.Replace("*", ".*");

				if (partialMatchIdentifier.Contains('<'))
				{
					int conditionStart = partialMatchIdentifier.IndexOf('<');
					int conditionEnd = partialMatchIdentifier.IndexOf('>');

					//condition start after condition end or no condition end at all
					if (conditionStart > conditionEnd)
					{
						throw new ArgumentException($"Malformed partial match identifier {partialMatchIdentifier.Replace(".*", "*")} at hierarchy level {hierarchyLevel}.");
					}

					string tag = partialMatchIdentifier.Substring(conditionStart + 1, conditionEnd - conditionStart - 1);
					conditionalTagsPerLevel[hierarchyLevel] = new HashSet<string>();

					if (tag.Contains(','))
					{
						conditionalTagsPerLevel[hierarchyLevel].UnionWith(tag.Split(','));
					}
					else
					{
						conditionalTagsPerLevel[hierarchyLevel].Add(tag);
					}

					partialMatchIdentifier = partialMatchIdentifier.Substring(0, conditionStart);
				}
			}

			return @"^\s*" + partialMatchIdentifier + @"\s*$";
		}

		private class MatchIdentifierRequestCacheEntry
		{
			internal string MatchIdentifier;
			internal readonly Dictionary<string, IRegistry> FullMatchedIdentifierRegistries;
			internal readonly Dictionary<string, string> FullMatchedIdentifierLocals;
			internal string[] FullMatchedIdentifierArray;
			internal ISet<IRegistry> AllReferredRegistries;

			internal MatchIdentifierRequestCacheEntry(string matchIdentifier, Dictionary<string, IRegistry> fullMatchedIdentifierRegistries, Dictionary<string, string> fullMatchedIdentifierLocals, string[] fullMatchedIdentifierArray, ISet<IRegistry> allReferredRegistries)
			{
				MatchIdentifier = matchIdentifier;
				FullMatchedIdentifierRegistries = fullMatchedIdentifierRegistries;
				FullMatchedIdentifierLocals = fullMatchedIdentifierLocals;
				FullMatchedIdentifierArray = fullMatchedIdentifierArray;
				AllReferredRegistries = allReferredRegistries;
			}
		}
	}
}