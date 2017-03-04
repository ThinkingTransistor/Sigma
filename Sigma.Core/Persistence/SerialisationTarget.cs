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
	public static class Target
	{
		/// <summary>
		/// Get a file using a certain file name from the default <see cref="SigmaEnvironment"/> storage path.
		/// </summary>
		/// <param name="name">The file name within the storage.</param>
		/// <returns>A file stream to a file with the given file in the storage path.</returns>
		public static FileStream FileByName(string name)
		{
			return FileByName(name, SigmaEnvironment.Globals.Get<string>("storage_path"));
		}

		/// <summary>
		/// Get a file using a certain file name and directory (missing directories are automatically created).
		/// </summary>
		/// <param name="name">The file name.</param>
		/// <param name="directory">The containing directory (recommended with the '/', but we check for and correct it anyway).</param>
		/// <returns>A file stream to a file within the given containing directory.</returns>
		public static FileStream FileByName(string name, string directory)
		{
			directory = directory.Replace("\\", "/");
			if (!directory.EndsWith("/"))
			{
				directory = directory + "/";
			}

			if (!Directory.Exists(directory))
			{
				Directory.CreateDirectory(directory);
			}

			return File.Open(directory + name, FileMode.OpenOrCreate);
		}

		/// <summary>
		/// Get a file using a complete path (relative to the working directory or absolute).
		/// </summary>
		/// <param name="path">The file path.</param>
		/// <returns>A file stream to a file with the given path.</returns>
		public static FileStream FileByPath(string path)
		{
			string directory = new FileInfo(path).DirectoryName;

			if (directory != null && !Directory.Exists(directory)) // might be null in a root directory?
			{
				Directory.CreateDirectory(directory);
			}

			return File.Open(path, FileMode.OpenOrCreate);
		}
	}
}
