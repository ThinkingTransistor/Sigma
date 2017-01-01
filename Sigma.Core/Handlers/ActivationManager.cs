/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using log4net;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Handlers
{
	/// <summary>
	/// The global activation manager for managing all globally used activation functions and their implementations.
	/// </summary>
	public static class ActivationManager
	{
		public static IEnumerable<string> Activations => ActivationHandles.Keys;
		public static readonly IReadOnlyDictionary<string, IActivationHandle> ActivationHandles;

		private static readonly ILog Logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private static readonly IDictionary<string, IActivationHandle> InternalActivationHandles;

		static ActivationManager()
		{
			InternalActivationHandles = new Dictionary<string, IActivationHandle>();
			ActivationHandles = new ReadOnlyDictionary<string, IActivationHandle>(InternalActivationHandles);

			AddDefaultActivations();
		}

		private static void AddDefaultActivations()
		{
			MapActivation("rel", new LambdaActivationHandle((n, h) => h.ReL(n), (a, h) => h.ReL(a)));
			MapActivation("sigmoid", new LambdaActivationHandle((n, h) => h.Sigmoid(n), (a, h) => h.Sigmoid(a)));
			MapActivation("tanh", new LambdaActivationHandle((n, h) => h.Tanh(n), (a, h) => h.Tanh(a)));
			MapActivation("softplus", new LambdaActivationHandle((n, h) => h.SoftPlus(n), (a, h) => h.SoftPlus(a)));
		}

		/// <summary>
		/// Map an activation function name to an activation handle.
		/// If an existing function is to be explicitly overwritten, set the forceOverwrite flag (use with caution).
		/// </summary>
		/// <param name="activation">The activation to map.</param>
		/// <param name="handle">The handle to map to.</param>
		/// <param name="forceOverwrite">Indicate if existing mappings should be explicitly overwritten.</param>
		public static void MapActivation(string activation, IActivationHandle handle, bool forceOverwrite = false)
		{
			if (activation == null) throw new ArgumentNullException(nameof(activation));
			if (handle == null) throw new ArgumentNullException(nameof(handle));

			if (InternalActivationHandles.ContainsKey(activation))
			{
				if (forceOverwrite)
				{
					InternalActivationHandles[activation] = handle;

					Logger.Info($"Overwrote activation {activation} with custom handle {handle}.");
				}
				else
				{
					throw new InvalidOperationException($"Cannot map activation {activation} to handle {handle}, activation is already mapped to handle {InternalActivationHandles[activation]} and forceOverwrite flag was not set.");
				}
			}

			InternalActivationHandles.Add(activation, handle);
		}

		/// <summary>
		/// Apply a certain activation defined in this activation manager to a number.
		/// </summary>
		/// <param name="activation">The activation to apply.</param>
		/// <param name="number">The number-</param>
		/// <param name="handler">The handler.</param>
		/// <returns>A number with the activation function applied to it.</returns>
		public static INumber ApplyActivation(string activation, INumber number, IComputationHandler handler)
		{
			if (!ActivationHandles.ContainsKey(activation))
			{
				throw new ArgumentException($"Activation {activation} is not mapped to any activation handle.");
			}

			return ActivationHandles[activation].Apply(number, handler);
		}

		/// <summary>
		/// Apply a certain activation defined in this activation manager to a number using a certain handler.
		/// </summary>
		/// <param name="activation">The activation to apply.</param>
		/// <param name="array">The number.</param>
		/// <param name="handler">The handler.</param>
		/// <returns>An array with the activation function applied to it.</returns>
		public static INDArray ApplyActivation(string activation, INDArray array, IComputationHandler handler)
		{
			if (!ActivationHandles.ContainsKey(activation))
			{
				throw new ArgumentException($"Activation {activation} is not mapped to any activation handle.");
			}

			return ActivationHandles[activation].Apply(array, handler);
		}

		/// <summary>
		/// An activation handle for numbers and ndarrays, representing a single activation function.
		/// </summary>
		public interface IActivationHandle
		{
			/// <summary>
			/// Apply this activation handle to a number using a certain handler.
			/// </summary>
			/// <param name="number">The number.</param>
			/// <param name="handler">The handler.</param>
			/// <returns>A number with the activation function applied to it.</returns>
			INumber Apply(INumber number, IComputationHandler handler);

			/// <summary>
			/// Apply this activation handle to a number using a certain handler.
			/// </summary>
			/// <param name="array">The number.</param>
			/// <param name="handler">The handler.</param>
			/// <returns>An array with the activation function applied to it.</returns>
			INDArray Apply(INDArray array, IComputationHandler handler);
		}

		/// <summary>
		/// A lambda activation handle for easy lambda inline definition of activation handles.
		/// </summary>
		public class LambdaActivationHandle : IActivationHandle
		{
			private readonly Func<INumber, IComputationHandler, INumber> _numberActivation;
			private readonly Func<INDArray, IComputationHandler, INDArray> _arrayActivation;

			/// <summary>
			/// Create a new lambda activation handler with a number and ndarray activation.
			/// </summary>
			/// <param name="numberActivation">The number activation function (number, handler) => (number).</param>
			/// <param name="arrayActivation">The array activation function (array, handler) => (array).</param>
			public LambdaActivationHandle(Func<INumber, IComputationHandler, INumber> numberActivation, Func<INDArray, IComputationHandler, INDArray> arrayActivation)
			{
				if (numberActivation == null) throw new ArgumentNullException(nameof(numberActivation));
				if (arrayActivation == null) throw new ArgumentNullException(nameof(arrayActivation));

				_numberActivation = numberActivation;
				_arrayActivation = arrayActivation;
			}

			public INumber Apply(INumber number, IComputationHandler handler)
			{
				return _numberActivation.Invoke(number, handler);
			}

			public INDArray Apply(INDArray array, IComputationHandler handler)
			{
				return _arrayActivation.Invoke(array, handler);
			}
		}
	}
}
