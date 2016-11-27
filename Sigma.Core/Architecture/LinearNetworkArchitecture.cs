/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A linear network architecture which only allows linear single connections between layer constructs.
	/// </summary>
	public class LinearNetworkArchitecture : INetworkArchitecture
	{
		public int LayerCount => _layerConstructs.Count;

		private readonly List<LayerConstruct> _layerConstructs;

		/// <summary>
		/// Create a linear network architecture consisting of a certain array of layer constructs.
		/// </summary>
		/// <param name="layerConstructs">The initial ordered layer constructs.</param>
		public LinearNetworkArchitecture(params LayerConstruct[] layerConstructs)
		{
			_layerConstructs = new List<LayerConstruct>(layerConstructs);
		}

		/// <summary>
		/// Create a linear network architecture consisting of a certain array of layer constructs.
		/// </summary>
		/// <param name="layerConstructs">The initial ordered layer constructs.</param>
		public LinearNetworkArchitecture(IEnumerable<LayerConstruct> layerConstructs)
		{
			_layerConstructs = new List<LayerConstruct>(layerConstructs);
		}

		public IEnumerable<LayerConstruct> YieldLayerConstructs()
		{
			return _layerConstructs;
		}

		public void Validate()
		{
			throw new NotImplementedException();
		}

		public LayerConstruct[] ResolveAllNames()
		{
			for (int i = 0; i < _layerConstructs.Count; i++)
			{
				if (_layerConstructs[i].Name.Contains('#'))
				{
					_layerConstructs[i].Name = _layerConstructs[i].Name.Replace("#", i.ToString());
				}
			}

			return _layerConstructs.ToArray();
		}

		public LinearNetworkArchitecture AppendEnd(LinearNetworkArchitecture other)
		{
			if (other == null)
			{
				throw new ArgumentNullException(nameof(other));
			}

			if (other.LayerCount == 0)
			{
				return this;
			}

			if (LayerCount > 0)
			{
				LayerConstruct lastOwn = _layerConstructs.Last();
				LayerConstruct firstOther = other._layerConstructs.First();

				lastOwn.AddOutput(firstOther);
				firstOther.AddInput(lastOwn);
			}

			_layerConstructs.AddRange(other._layerConstructs);

			return this;
		}

		public LinearNetworkArchitecture AppendStart(LinearNetworkArchitecture other)
		{
			if (other == null)
			{
				throw new ArgumentNullException(nameof(other));
			}

			if (other.LayerCount == 0)
			{
				return this;
			}

			if (LayerCount > 0)
			{
				LayerConstruct firstOwn = _layerConstructs.First();
				LayerConstruct lastOther = _layerConstructs.Last();

				lastOther.AddOutput(firstOwn);
				firstOwn.AddInput(lastOther);
			}

			for (int i = other._layerConstructs.Count; i >= 0; i--)
			{
				_layerConstructs.Insert(0, other._layerConstructs[i]);
			}

			return this;
		}

		public static LinearNetworkArchitecture operator +(LinearNetworkArchitecture self, LinearNetworkArchitecture other)
		{
			return self.AppendEnd(other);
		}

		public static LinearNetworkArchitecture operator +(LinearNetworkArchitecture self, LayerConstruct other)
		{
			return self.AppendEnd(new LinearNetworkArchitecture(other));
		}

		public static LinearNetworkArchitecture operator *(int multiplier, LinearNetworkArchitecture self)
		{
			if (multiplier <= 0)
			{
				throw new ArgumentException($"Multiplier must be >= 1, but multiplier was {multiplier}.");
			}

			if (self.LayerCount == 0)
			{
				return self;
			}

			for (int i = 0; i < multiplier; i++)
			{
				LinearNetworkArchitecture copy = new LinearNetworkArchitecture(self._layerConstructs.ConvertAll(x =>
				{
					if (!x.Name.Contains('#'))
					{
						throw new ArgumentException("Attempted to multiply linear network architecture containing layer construct with static name, which cannot be multiplied. Include '#' in layer name for dynamic auto naming.");
					}

					return x.Copy();
				}));

				LayerConstruct lastOwn = self._layerConstructs.Last();
				LayerConstruct firstOther = copy._layerConstructs.First();

				lastOwn.AddOutput(firstOther);
				firstOther.AddInput(lastOwn);
			}

			return self;
		}
	}
}
