/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public static class AttributeUtils
	{
		public static IEnumerable<Type> GetTypesWithAttribute(Type attribute)
		{
			return AppDomain.CurrentDomain.GetAssemblies().SelectMany(assembly => GetTypesWithAttribute(assembly, attribute));
		}

		public static IEnumerable<Type> GetTypesWithAttribute(Assembly assembly, Type attribute)
		{
			return assembly.GetTypes().Where(type => type.GetCustomAttributes(attribute, true).Length > 0);
		}
	}
}