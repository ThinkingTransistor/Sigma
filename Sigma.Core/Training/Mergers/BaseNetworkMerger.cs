/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Mergers
{
	/// <summary>
	///     This is the <see cref="BaseNetworkMerger" /> - it provides functionality that
	///     should make it easier to implement the <see cref="INetworkMerger" /> interface.
	/// </summary>
	[Serializable]
	public abstract class BaseNetworkMerger : INetworkMerger
	{
		/// <summary>
		///     The matching keys.
		/// </summary>
		protected readonly ICollection<string> MatchIdentifier;

		[NonSerialized]
		private ILog _log;

		protected ILog Log => _log ?? (_log = LogManager.GetLogger(GetType()));

		protected BaseNetworkMerger(params string[] matchIdentifiers)
		{
			MatchIdentifier = new List<string>(matchIdentifiers);
		}

		/// <summary>
		///     Specify how multiple networks are merged into a single one. <see cref="root" /> is <em>not</em>
		///     considered for the calculation. It is merely the storage container. (Although root can also be in
		///     <see cref="networks" />).
		/// </summary>
		/// <param name="root">
		///     The root network that will be modified. Since the <see cref="INetworkMerger" /> does not know how
		///     to create a <see cref="INetwork" />, it will be passed not returned.
		/// </param>
		/// <param name="networks">
		///     The networks that will be merged into the <see cref="root" />. Can contain <see cref="root" />
		///     itself.
		/// </param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		public void Merge(INetwork root, IEnumerable<INetwork> networks, IComputationHandler handler = null)
		{
			IRegistryResolver rootResolver = new RegistryResolver(root.Registry);
			string[] mergeKeys = CopyMatchIdentifiers();

			if (mergeKeys.Length == 0)
			{
				Log.Warn($"Attempted merge network {root} with networks {networks} using handler {handler} but no merge keys were set so nothing will happen. This is probably not intended.");
			}

			// mapping of resolved mergeEnetry and all data
			IDictionary<string, IList<object>> resolvedDataArrays = new Dictionary<string, IList<object>>(mergeKeys.Length);
			int numNetworks = 0;

			// fill the mapping of all values
			foreach (INetwork network in networks)
			{
				IRegistryResolver resolver = new RegistryResolver(network.Registry);

				foreach (string mergeKey in mergeKeys)
				{
					string[] fullyResolvedIdentifiers;
					object[] values = resolver.ResolveGet<object>(mergeKey, out fullyResolvedIdentifiers);

					Debug.Assert(fullyResolvedIdentifiers.Length == values.Length);

					for (int i = 0; i < values.Length; i++)
					{
						IList<object> allValuesAtKey = resolvedDataArrays.TryGetValue(fullyResolvedIdentifiers[i], () => new List<object>());

						allValuesAtKey.Add(values[i]);
					}
				}

				numNetworks++;
			}

			foreach (KeyValuePair<string, IList<object>> keyDataPair in resolvedDataArrays)
			{
				int numObjects = keyDataPair.Value.Count;

				if (numObjects != numNetworks)
				{
					_log.Warn($"Inconsistent network states for identifier \"{keyDataPair.Key}\", only {keyDataPair.Value.Count} have it but there are {numNetworks} networks.");
				}

				object merged = Merge(keyDataPair.Value.ToArray(), handler);
				rootResolver.ResolveSet(keyDataPair.Key, merged);
			}
		}

		/// <summary>
		///     Returns from a list, from every object[], the ith index.
		///     e.g. 3 passed objects[] => get index 2, returns an object[] with a length of 3
		///     and from everyone the second index.
		/// </summary>
		/// <param name="list">The list the action will be performed on. (Normally amount of networks).</param>
		/// <param name="index">The index we are looking for.</param>
		/// <returns>The object[] specified previously how it is generated.</returns>
		private object[] GetAllObjectsWithIndex(IList<object[]> list, int index)
		{
			object[] objectsWithIndex = new object[list.Count];

			for (int i = 0; i < objectsWithIndex.Length; i++)
			{
				if (index >= list[i].Length)
				{
					return null;
				}

				objectsWithIndex[i] = list[i][index];
			}

			return objectsWithIndex;
		}

		/// <summary>
		///     Specify how two networks are merged into a single one (root = root + other). This method can be achieved with
		///     <see cref="Merge(INetwork,IEnumerable{INetwork},IComputationHandler)" /> (internally
		///     implemented this way), but it may be easier to call this function.
		/// </summary>
		/// <param name="root">The root network that will be modified and contain the result.</param>
		/// <param name="other">The unchanged other network.</param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		public void Merge(INetwork root, INetwork other, IComputationHandler handler = null)
		{
			Merge(root, new[] {root, other}, handler);
		}

		/// <summary>
		///     Specify the registry keys (match identifiers) that will be merged.
		///     This supports the full
		///     <see cref="Utils.IRegistryResolver" /> syntax.
		/// </summary>
		/// <param name="matchIdentifier">The key of the registry.</param>
		public void AddMergeEntry(string matchIdentifier)
		{
			lock (MatchIdentifier)
			{
				MatchIdentifier.Add(matchIdentifier);
			}
		}

		/// <summary>
		///     Remove a previously specified match identifier.
		/// </summary>
		/// <param name="matchIdentifier">The key to remove from the targeted registries.</param>
		public void RemoveMergeEntry(string matchIdentifier)
		{
			lock (MatchIdentifier)
			{
				MatchIdentifier.Remove(matchIdentifier);
			}
		}

		/// <summary>
		///     The method that will be called when no other method suits.
		/// </summary>
		/// <param name="objects">The objects.</param>
		/// <param name="handler">The handler (may be null). </param>
		/// <returns></returns>
		protected virtual object MergeDefault(object[] objects, IComputationHandler handler)
		{
			// default policy is just return first value if not mergeable
			return objects[0];
		}

		/// <summary>
		///     This method is used to merge doubles.
		/// </summary>
		/// <param name="doubles">The doubles.</param>
		/// <returns>The merged value.</returns>
		protected abstract double MergeDoubles(double[] doubles);

		/// <summary>
		///     This method is used to merge floats. Override it as you need,
		///     but per default it calls <see cref="MergeDoubles" />.
		/// </summary>
		/// <param name="floats">The floats.</param>
		/// <returns>The merged value.</returns>
		protected virtual float MergeFloats(float[] floats)
		{
			return (float) MergeDoubles(Array.ConvertAll(floats, x => (double) x));
		}

		/// <summary>
		///     This method is used to merge floats. Override it as you need,
		///     but per default it calls <see cref="MergeDoubles" />.
		/// </summary>
		/// <param name="ints">The ints.</param>
		/// <returns>The merged value.</returns>
		protected virtual int MergeInts(int[] ints)
		{
			return (int) MergeDoubles(Array.ConvertAll(ints, x => (double) x));
		}

		/// <summary>
		///     This method is used to merge floats. Override it as you need,
		///     but per default it calls <see cref="MergeDoubles" />.
		/// </summary>
		/// <param name="shorts">The shorts.</param>
		/// <returns>The merged value.</returns>
		protected virtual short MergeShorts(short[] shorts)
		{
			return (short) MergeDoubles(Array.ConvertAll(shorts, x => (double) x));
		}

		/// <summary>
		///     This method is used to merge floats. Override it as you need,
		///     but per default it calls <see cref="MergeDoubles" />.
		/// </summary>
		/// <param name="longs">The longs.</param>
		/// <returns>The merged value.</returns>
		protected virtual long MergeLongs(long[] longs)
		{
			return (long) MergeDoubles(Array.ConvertAll(longs, x => (double) x));
		}

		/// <summary>
		///     This method is used to merge <see cref="INDArray" />s.
		/// </summary>
		/// <param name="arrays">The arrays to merge. </param>
		/// <param name="handler">The handler that may or may not be specified.</param>
		/// <returns>A merged <see cref="INDArray" />.</returns>
		protected abstract INDArray MergeNDArrays(INDArray[] arrays, IComputationHandler handler);

		/// <summary>
		///     This method is used to merge <see cref="INDArray" />s.
		/// </summary>
		/// <param name="numbers">The numbers to merge. </param>
		/// <param name="handler">The handler that may or may not be specified.</param>
		/// <returns>A merged <see cref="INumber" />.</returns>
		protected abstract INumber MergeNumbers(INumber[] numbers, IComputationHandler handler);

		/// <summary>
		///     This method gets called before merge is called. So if your class has attributes
		///     that have to be checked before every merge, feel free to override this method.
		///     Per default it is an empty method. (i.e. no back calling required)
		/// </summary>
		/// <param name="objects">The objects that will be merged.</param>
		protected virtual void CheckObjects(object[] objects)
		{
		}

		/// <summary>
		///     Checks every object in <see cref="objects" /> if it can be casted to a type,
		///     and if everything can be casted, calls the correct method. As last resort, <see cref="MergeDefault" />
		///     will be called.
		/// </summary>
		/// <param name="objects">The objects that will be merged.</param>
		/// <param name="handler">The handler that may or may not be specified.</param>
		/// <returns>A merged object.</returns>
		protected virtual object Merge(object[] objects, IComputationHandler handler)
		{
			CheckObjects(objects);

			object merged;

			if (CastToAndCall<double>(objects, MergeDoubles, out merged))
			{
				return merged;
			}

			if (CastToAndCall<float>(objects, MergeFloats, out merged))
			{
				return merged;
			}

			if (CastToAndCall<int>(objects, MergeInts, out merged))
			{
				return merged;
			}

			if (CastToAndCall<short>(objects, MergeShorts, out merged))
			{
				return merged;
			}

			if (CastToAndCall<long>(objects, MergeLongs, out merged))
			{
				return merged;
			}

			if (CastToAndCall<INDArray>(objects, o => MergeNDArrays(o, handler), out merged))
			{
				return merged;
			}

			if (CastToAndCall<INumber>(objects, o => MergeNumbers(o, handler), out merged))
			{
				return merged;
			}

			return objects[0]; // TODO fix default behavior for merging non-mergeable objects... I suggest just ignoring
			//return MergeDefault(objects, handler);
		}

		/// <summary>
		///     Copy the <see cref="MatchIdentifier" />s into a <see cref="string" />[], so it can be used
		///     without a lock.
		/// </summary>
		/// <returns>A copy of all elements currently in <see cref="CopyMatchIdentifiers" />.</returns>
		private string[] CopyMatchIdentifiers()
		{
			string[] copiedStrings;

			lock (MatchIdentifier)
			{
				copiedStrings = new string[MatchIdentifier.Count];
				MatchIdentifier.CopyTo(copiedStrings, 0);
			}

			return copiedStrings;
		}

		/// <summary>
		///     This method tries to cast every object of <see cref="objects" /> to T.
		///     If all objects can be casted, it calls the passed calculate function.
		/// </summary>
		/// <typeparam name="T">The object to cast to.</typeparam>
		/// <param name="objects">The objects that will be passed.</param>
		/// <param name="calculate">The method that will be called if all objects can be converted.</param>
		/// <param name="mergedObject">The merged object (initialised with a simple call of <see cref="calculate" />).</param>
		/// <returns>
		///     If any of those objects cannot be casted (and therefore the creation of an T[] is impossible), <c>false</c>.
		///     <c>True</c> otherwise.
		/// </returns>
		protected bool CastToAndCall<T>(object[] objects, Func<T[], T> calculate, out object mergedObject)
		{
			mergedObject = null;

			if (objects.Any(t => !(t is T)))
			{
				return false;
			}

			T[] castedObjects = Array.ConvertAll(objects, x => (T) x);

			mergedObject = calculate(castedObjects);

			return true;
		}
	}
}