/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using log4net;

namespace Sigma.Core.Architecture.Linear
{
	/// <summary>
	/// A linear network architecture which only allows linear single connections between layer constructs.
	/// </summary>
	public class LinearNetworkArchitecture : INetworkArchitecture
	{
		public IRegistry Registry { get; }
		public int LayerCount => _layerConstructs.Count;

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private readonly List<LayerConstruct> _layerConstructs;

		/// <summary>
		/// Create a linear network architecture consisting of a certain array of layer constructs.
		/// </summary>
		/// <param name="layerConstructs">The initial ordered layer constructs.</param>
		public LinearNetworkArchitecture(params LayerConstruct[] layerConstructs)
		{
			_layerConstructs = new List<LayerConstruct>(layerConstructs);
			Registry = new Registry(tags: "architecture");

			UpdateRegistry();
		}

		/// <summary>
		/// Create a linear network architecture consisting of a certain array of layer constructs.
		/// </summary>
		/// <param name="layerConstructs">The initial ordered layer constructs.</param>
		public LinearNetworkArchitecture(IEnumerable<LayerConstruct> layerConstructs)
		{
			_layerConstructs = new List<LayerConstruct>(layerConstructs);
			Registry = new Registry(tags: "architecture");

			UpdateRegistry();
		}

		protected virtual void UpdateRegistry()
		{
			Registry["type"] = "linear";
			Registry["layercount"] = LayerCount;
		}

		protected virtual LinearNetworkArchitecture Copy()
		{
			Dictionary<LayerConstruct, LayerConstruct> mappedConstructCopies = new Dictionary<LayerConstruct, LayerConstruct>();
			List<LayerConstruct> copiedConstructs = _layerConstructs.ConvertAll(construct =>
			{
				LayerConstruct copy = construct.Copy();

				mappedConstructCopies.Add(construct, copy);

				return copy;
			});

			foreach (LayerConstruct original in _layerConstructs)
			{
				LayerConstruct copy = mappedConstructCopies[original];

				foreach (string inputAlias in original.Inputs.Keys)
				{
					LayerConstruct input = original.Inputs[inputAlias];

					if (mappedConstructCopies.ContainsKey(input))
					{
						input = mappedConstructCopies[input];
					}

					copy.AddInput(input, inputAlias);
				}

				foreach (string outputAlias in original.Outputs.Keys)
				{
					LayerConstruct output = original.Outputs[outputAlias];

					if (mappedConstructCopies.ContainsKey(output))
					{
						output = mappedConstructCopies[output];
					}

					copy.AddOutput(output, outputAlias);
				}
			}

			return new LinearNetworkArchitecture(copiedConstructs);
		}

		public void Validate()
		{
			if (_layerConstructs.Count != _layerConstructs.Distinct().Count())
			{
				throw new InvalidNetworkArchitectureException($"All layer constructs in the network architecture must be unique.");
			}

			HashSet<LayerConstruct> previousConstructs = new HashSet<LayerConstruct>();

			foreach (LayerConstruct construct in YieldLayerConstructsOrdered())
			{
				if (construct.InputsExternal && construct.Inputs.Count > 0)
				{
					throw new InvalidNetworkArchitectureException($"Layer constructs marked with external inputs cannot have any internal inputs, but construct {construct} has {construct.Inputs.Count}.");
				}

				if (construct.OutputsExternal && construct.Outputs.Count > 0)
				{
					throw new InvalidNetworkArchitectureException($"Layer constructs marked with external outputs cannot have any internal outputs, but construct {construct} has {construct.Outputs.Count}.");
				}

				foreach (LayerConstruct previousConstruct in previousConstructs)
				{
					if (construct.Outputs.Values.Contains(previousConstruct))
					{
						throw new InvalidNetworkArchitectureException($"Linear networks can only have acyclic linear connections, but connection from layer construct {previousConstruct} connects back to itself through {construct}.");
					}
				}

				previousConstructs.Add(construct);
			}

			
		}

		public void ResolveAllNames()
		{
			string formatString = "D" + Math.Ceiling(Math.Log10(LayerCount));

			for (int i = 0; i < _layerConstructs.Count; i++)
			{
				if (_layerConstructs[i].UnresolvedName.Contains('#'))
				{
					_layerConstructs[i].Name = _layerConstructs[i].UnresolvedName.Replace("#", i.ToString(formatString));
				}
			}
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

			UpdateRegistry();

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
				LayerConstruct lastOther = other._layerConstructs.Last();

				lastOther.AddOutput(firstOwn);
				firstOwn.AddInput(lastOther);
			}

			for (int i = other._layerConstructs.Count - 1; i >= 0; i--)
			{
				_layerConstructs.Insert(0, other._layerConstructs[i]);
			}

			UpdateRegistry();

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

		public static LinearNetworkArchitecture operator +(LayerConstruct other, LinearNetworkArchitecture self)
		{
			return self.AppendStart(new LinearNetworkArchitecture(other));
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

			if (multiplier >= 2)
			{
				if (self._layerConstructs.Any(construct => !construct.Name.Contains('#')))
				{
					throw new ArgumentException("Attempted to multiply linear network architecture containing layer construct with static name, which cannot be multiplied. Include '#' in layer name for dynamic auto naming.");
				}
			}

			LinearNetworkArchitecture multipliedSelf = new LinearNetworkArchitecture();

			for (int i = 0; i < multiplier; i++)
			{
				LinearNetworkArchitecture copy = self.Copy();

				multipliedSelf.AppendEnd(copy);
			}

			return multipliedSelf;
		}

		public IEnumerable<LayerConstruct> YieldLayerConstructsOrdered()
		{
			return _layerConstructs;
		}
	}
}
