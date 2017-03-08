/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;

namespace Sigma.Core.Monitors.WPF.Panels.Charts.Definitions
{
	public interface IPointVisualiser
	{
		void Add(object value);
		void AddRange(IEnumerable<object> values);
	}
}