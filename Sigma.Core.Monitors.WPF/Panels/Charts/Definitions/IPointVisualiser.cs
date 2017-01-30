using System.Collections.Generic;

namespace Sigma.Core.Monitors.WPF.Panels.Charts.Definitions
{
	public interface IPointVisualiser
	{
		void Add(object value);
		void AddRange(IEnumerable<object> values);
	}
}