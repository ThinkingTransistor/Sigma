/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// The target mode for a particular hook. 
	/// Used to implicitly add it to local / global hooks if specified, otherwise (if marked as <see cref="TargetMode.Any"/> require explicit call.
	/// Note: Target mode does not have to match the actual target of a hook. It's merely a recommendation.
	/// </summary>
	public enum TargetMode
	{
		/// <summary>
		/// The local target mode, prefer per-worker invocation.
		/// </summary>
		Local,

		/// <summary>
		/// The global target mode, prefer per-operator invocation.
		/// </summary>
		Global,

		/// <summary>
		/// The default any target mode, no preference given. Requires explicit call to attach as hook.
		/// </summary>
		Any
	}
}
