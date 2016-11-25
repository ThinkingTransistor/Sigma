/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// An alias for Dictionary|string, IRegistry| for cleaner typing. That's it.
	/// </summary>
	public class AliasRegistry : Dictionary<string, IRegistry>
	{
	}
}
