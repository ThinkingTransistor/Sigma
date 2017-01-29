using System.Collections.Generic;

namespace Sigma.Core.Monitors.WPF.Panels.Charts.Definitions
{
	public interface IPointVisualiser<in T>
	{
		void Add(T value);
		void AddRange(IEnumerable<T> values);
	}
}