/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	public class RegistryResolver : IRegistryResolver, IRegistryHierarchyChangeListener
	{
		public IRegistry Root
		{
			get; private set;
		}

		private static string[] EMPTY_ARRAY = new string[0] { };

		private Dictionary<string, MatchIdentifierRequestCacheEntry> matchIdentifierCache;
		private ISet<string> fullIdentifiersToInvalidate;

		public RegistryResolver(IRegistry root)
		{
			if (root == null)
			{
				throw new ArgumentNullException("Registry root may not be null.");
			}

			root.HierarchyChangeListeners.Add(this);

			this.Root = root;
			this.matchIdentifierCache = new Dictionary<string, MatchIdentifierRequestCacheEntry>(8);
			this.fullIdentifiersToInvalidate = new HashSet<string>();
		}

		public void OnChildHierarchyChanged(string identifier, IRegistry previousChild, IRegistry newChild)
		{
			foreach (string fullIdentifier in matchIdentifierCache.Keys)
			{
				MatchIdentifierRequestCacheEntry entry = matchIdentifierCache[fullIdentifier];

				if (entry.allReferredRegistries.Contains<IRegistry>(previousChild))
				{
					fullIdentifiersToInvalidate.Add(fullIdentifier);
				}
			}

			foreach (string fullIdentifier in fullIdentifiersToInvalidate)
			{
				matchIdentifierCache.Remove(fullIdentifier);
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
				}
				while ((referredRegistry = referredRegistry.Parent) != null);
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
				if (value is IRegistry)
				{
					RemoveChildHierarchyListener((IRegistry) value);
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
				if (child is IRegistry)
				{
					AddChildHierarchyListener((IRegistry) value);
				}
			}
		}

		private void CheckMatchIdentifier(String matchIdentifier)
		{
			if (matchIdentifier == null)
			{
				throw new ArgumentNullException("Match identifier cannot be null.");
			}

			if (matchIdentifier.Length == 0)
			{
				throw new ArgumentException("Match identifier cannot be of length 0.");
			}
		}

		public T[] ResolveGet<T>(string matchIdentifier, T[] values = null)
		{
			string[] emptyArrayThrowaway = EMPTY_ARRAY;

			return ResolveGet<T>(matchIdentifier, ref emptyArrayThrowaway, values);
		}

		public T[] ResolveGet<T>(string matchIdentifier, ref string[] fullMatchedIdentifierArray, T[] values = null)
		{
			CheckMatchIdentifier(matchIdentifier);

			MatchIdentifierRequestCacheEntry cacheEntry = GetOrCreateCacheEntry(matchIdentifier);

			if (values == null || values.Length < cacheEntry.fullMatchedIdentifierArray.Length)
			{
				values = new T[cacheEntry.fullMatchedIdentifierArray.Length];
			}

			for (int i = 0; i < cacheEntry.fullMatchedIdentifierArray.Length; i++)
			{
				string fullIdentifier = cacheEntry.fullMatchedIdentifierArray[i];

				values[i] = cacheEntry.fullMatchedIdentifierRegistries[fullIdentifier].Get<T>(cacheEntry.fullMatchedIdentifierLocals[fullIdentifier]);
			}

			fullMatchedIdentifierArray = cacheEntry.fullMatchedIdentifierArray;

			return values;
		}

		public string[] ResolveSet<T>(string matchIdentifier, T value, System.Type associatedType = null)
		{
			CheckMatchIdentifier(matchIdentifier);

			MatchIdentifierRequestCacheEntry cacheEntry = GetOrCreateCacheEntry(matchIdentifier);

			string localIdentifier = matchIdentifier.Contains<char>('.') ? matchIdentifier.Substring(matchIdentifier.LastIndexOf('.') + 1) : matchIdentifier;

			for (int i = 0; i < cacheEntry.fullMatchedIdentifierArray.Length; i++)
			{
				string fullIdentifier = cacheEntry.fullMatchedIdentifierArray[i];

				cacheEntry.fullMatchedIdentifierRegistries[fullIdentifier].Set(localIdentifier, value, associatedType);
			}

			return cacheEntry.fullMatchedIdentifierArray;
		}

		private MatchIdentifierRequestCacheEntry GetOrCreateCacheEntry(string matchIdentifier)
		{
			CheckMatchIdentifier(matchIdentifier);

			if (!matchIdentifierCache.ContainsKey(matchIdentifier))
			{
				string[] matchIdentifierParts = matchIdentifier.Split('.');

				ISet<string>[] conditionalTagsPerLevel = new HashSet<string>[matchIdentifierParts.Length];

				for (int i = 0; i < matchIdentifierParts.Length; i++)
				{
					matchIdentifierParts[i] = ParseMatchIdentifier(i, matchIdentifierParts[i], conditionalTagsPerLevel);
				}

				Dictionary<string, IRegistry> fullMatchedIdentifierRegistries = new Dictionary<string, IRegistry>();

				MatchIdentifierRequestCacheEntry newCacheEntry = new MatchIdentifierRequestCacheEntry(matchIdentifier, fullMatchedIdentifierRegistries, new Dictionary<string, string>(), null, null);

				AddMatchingIdentifiersFromRegistryTree(0, matchIdentifierParts.Length - 1, Root, "", matchIdentifierParts, conditionalTagsPerLevel, newCacheEntry);

				newCacheEntry.fullMatchedIdentifierArray = fullMatchedIdentifierRegistries.Keys.ToArray<string>();

				newCacheEntry.allReferredRegistries = GetReferredRegistries(newCacheEntry.fullMatchedIdentifierRegistries.Values.ToList<IRegistry>());

				//only cache if the last identifier level did not contain a blank unrestricted wildcard (wildcard without conditional tags, meaning we can't cache anything as it could be any value)
				int lastLevel = matchIdentifierParts.Length - 1;
				bool shouldCache = !matchIdentifierParts[lastLevel].Contains(".*") || conditionalTagsPerLevel[lastLevel]?.Count > 0;

				if (shouldCache)
				{
					matchIdentifierCache.Add(matchIdentifier, newCacheEntry);
				}

				return newCacheEntry;				
			}

			return matchIdentifierCache[matchIdentifier];
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

						newCacheEntry.fullMatchedIdentifierRegistries.Add(globalFullIdentifier, currentRootAtLevel);
						newCacheEntry.fullMatchedIdentifierLocals.Add(globalFullIdentifier, identifier);
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
			if (partialMatchIdentifier.Contains<char>('*'))
			{
				partialMatchIdentifier = partialMatchIdentifier.Replace("*", ".*");

				if (partialMatchIdentifier.Contains<char>('<'))
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

					if (tag.Contains<char>(','))
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
			internal string matchIdentifier;
			internal Dictionary<string, IRegistry> fullMatchedIdentifierRegistries;
			internal Dictionary<string, string> fullMatchedIdentifierLocals;
			internal string[] fullMatchedIdentifierArray;
			internal ISet<IRegistry> allReferredRegistries;

			internal MatchIdentifierRequestCacheEntry(string matchIdentifier, Dictionary<string, IRegistry> fullMatchedIdentifierRegistries, Dictionary<string, string> fullMatchedIdentifierLocals, string[] fullMatchedIdentifierArray, ISet<IRegistry> allReferredRegistries)
			{
				this.matchIdentifier = matchIdentifier;
				this.fullMatchedIdentifierRegistries = fullMatchedIdentifierRegistries;
				this.fullMatchedIdentifierLocals = fullMatchedIdentifierLocals;
				this.fullMatchedIdentifierArray = fullMatchedIdentifierArray;
				this.allReferredRegistries = allReferredRegistries;
			}
		}
	}
}
