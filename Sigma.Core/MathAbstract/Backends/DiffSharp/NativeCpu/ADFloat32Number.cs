/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using DiffSharp.Interop.Float32;

namespace Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu
{
	public class ADFloat32Number : ADNumber<float>
	{
		public readonly DNumber _adNumberHandle;

		public ADFloat32Number(float value) : base(value)
		{
		}

		//public ADFloat32Number(DNumber handle) : base()
		//{
			
		//}
	}
}
