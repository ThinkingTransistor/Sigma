using System;
using System.Collections.Generic;
using System.Linq;
using log4net;
using Sigma.Core.Monitors.WPF.Annotations;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Monitors.WPF.View.Parameterisation;

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	public class ParameterVisualisationManager : IParameterVisualisationManager
	{
		/// <summary>
		/// The logger.
		/// </summary>
		private readonly ILog _log = LogManager.GetLogger(typeof(ParameterVisualisationManager));

		/// <summary>
		/// Maps a given parameter type (e.g. <c>bool</c>) to a
		/// <see cref="IParameterVisualiser"/> implementation.
		/// </summary>
		protected readonly Dictionary<Type, Type> TypeMapping;

		/// <summary>
		/// Maps a given parameter type (e.g. <c>bool</c>) to a 
		/// <see cref="ParameterVisualiserAttribute"/> that contains 
		/// details like <see cref="ParameterVisualiserAttribute.VisualiserPriority"/>.
		/// </summary>
		protected readonly Dictionary<Type, ParameterVisualiserAttribute> AttributeMapping;


		public ParameterVisualisationManager(bool autoAssign = true)
		{
			TypeMapping = new Dictionary<Type, Type>();
			AttributeMapping = new Dictionary<Type, ParameterVisualiserAttribute>();

			if (autoAssign)
			{
				// ReSharper disable once VirtualMemberCallInConstructor
				AssignMarkedClasses();
			}
		}

		/// <summary>
		/// Assign all classes that are marked with <see cref="ParameterVisualiserAttribute"/>. 
		/// </summary>
		protected virtual void AssignMarkedClasses()
		{
			// get all classes that have the custom attribute
			IEnumerable<Type> classes = AttributeUtils.GetTypesWithAttribute(typeof(ParameterVisualiserAttribute));

			foreach (Type @class in classes)
			{
				ParameterVisualiserAttribute[] visualisers = (ParameterVisualiserAttribute[])Attribute.GetCustomAttributes(@class, typeof(ParameterVisualiserAttribute));

				foreach (ParameterVisualiserAttribute visualiser in visualisers)
				{
					Add(visualiser, @class);
				}
			}
		}

		public virtual bool Add(ParameterVisualiserAttribute parameterInfo, Type visualiserClass)
		{
			Type storedClass;
			ParameterVisualiserAttribute storedAttribte;

			// if the mapping has already been added 
			if (TypeMapping.TryGetValue(parameterInfo.Type, out storedClass) && AttributeMapping.TryGetValue(parameterInfo.Type, out storedAttribte))
			{
				// if the new values have a lower priority, we return false
				if (parameterInfo.Priority <= storedAttribte.Priority)
				{
					_log.Warn($"{parameterInfo.Type} is currently visualised by {storedClass.Name}; {visualiserClass.Name} tried to be the visualiser but has a lower priority ({parameterInfo.Priority} <= {storedAttribte.Priority}).");

					return false;
				}

				_log.Info($"{parameterInfo.Type} was visualised by {storedClass.Name}; {visualiserClass.Name} has a higher priority and is therefore the new visualiser ({parameterInfo.Priority} > {storedAttribte.Priority}).");
			}

			TypeMapping[parameterInfo.Type] = visualiserClass;
			AttributeMapping[parameterInfo.Type] = parameterInfo;

			return true;
		}

		public virtual bool Remove(Type type)
		{
			return TypeMapping.Remove(type) && AttributeMapping.Remove(type);
		}

		//TODO: HACK remove / move
		public Type VisualiserType([NotNull] object obj)
		{
			if (obj == null)
			{
				throw new ArgumentNullException(nameof(obj));
			}

			// check the type itself
			Type type = obj.GetType();

			// it is null when calling typeof(object).BaseType
			// prevent that everything with an interface is an object
			while (type.BaseType != null)
			{
				if (TypeMapping.ContainsKey(type))
				{
					return type;
				}

				type = type.BaseType;
			}

			// check all interfaces
			type = obj.GetType();

			Type[] interfaces = type.GetInterfaces();
			foreach (Type iface in interfaces)
			{
				if (TypeMapping.ContainsKey(iface))
				{
					return iface;
				}
			}

			return typeof(object);
		}
	}
}