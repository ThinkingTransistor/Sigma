/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.IO;

namespace Sigma.Core.Persistence
{
	/// <summary>
	/// A utility class for creating serialisation targets (i.e. streams).
	/// </summary>
	public static class SerialisationTarget
	{
		public static FileStream File(string name)
		{
			return File(name, SigmaEnvironment.Globals.Get<string>("storage_path"));
		}

		public static FileStream File(string name, string directory)
		{
			directory = directory.Replace("\\", "/");
			if (!directory.EndsWith("/"))
			{
				directory = directory + "/";
			}

			return System.IO.File.Create(directory + name);
		}

		public static FileStream FileByPath(string path)
		{
			return System.IO.File.Create(path);
		}
	}
}
