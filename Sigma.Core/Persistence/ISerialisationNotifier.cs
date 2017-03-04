/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Persistence
{
	/// <summary>
	/// An interface to indicate to the Sigma <see cref="Sigma.Core.Persistence"/> tools that your object should get notified on serialisation events. 
	/// Note: There is no OnDeserialising method because there is no guarantee that the <see cref="ISerialiser"/> used has access to the object before it's fully constructed.
	/// </summary>
	public interface ISerialisationNotifier
	{
		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		void OnSerialising();

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		void OnSerialised();

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		void OnDeserialised();
	}
}
