using System;
using System.Threading;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// Some helping wrappers for threads. 
	/// </summary>
	public static class ThreadUtils
	{
		/// <summary>
		/// This <see cref="BlockingThread"/> allows the caller to wait for a certain event until the callee's control flow resumes.
		/// </summary>
		public class BlockingThread
		{
			/// <summary>
			/// The <see cref="ThreadStart"/> that the newly created <see cref="Thread"/> will receive.
			/// </summary>
			protected Action<ManualResetEvent> Action;

			/// <summary>
			/// Set the <see cref="Action"/> manually. 
			/// </summary>
			protected BlockingThread()
			{

			}

			/// <summary>
			/// Create a new blocking thread. After block it blocks as long as <see cref="EventWaitHandle.Set"/> is called in the <see cref="action"/>.
			/// </summary>
			/// <param name="action">The <see cref="ThreadStart"/> that the newly created <see cref="Thread"/> will receive. This action decides when to unlock the caller. </param>
			public BlockingThread(Action<ManualResetEvent> action) : this()
			{
				if (action == null)
				{
					throw new ArgumentNullException(nameof(action));
				}

				Action = action;
			}

			/// <summary>
			/// When called start, the calling thread blocks until <see cref="EventWaitHandle.Set"/> is called.
			/// The set may only be invoked once. 
			/// </summary>
			public void Start()
			{
				using (ManualResetEvent stateChangeStart = new ManualResetEvent(false))
				{
					// ReSharper disable once AccessToDisposedClosure
					new Thread(() => Action.Invoke(stateChangeStart)).Start();

					stateChangeStart.WaitOne();
				}
			}
		}

		/// <summary>
		/// This thread unlocks the calling thread as soon as it receives the lock for given object.
		/// </summary>
		public class BlockingLockingThread : BlockingThread
		{
			/// <summary>
			/// Specify that action that will be executed and the lock that is required.
			/// Once the <see cref="lockObject"/> is received, the calling thread is unlocked. 
			/// </summary>
			/// <param name="lockObject">The object a lock will acquired on.</param>
			/// <param name="action">The action that will be executed inside that object lock. </param>
			public BlockingLockingThread(object lockObject, Action action)
			{
				Action = reset =>
				{
					lock (lockObject)
					{
						reset.Set();
						action?.Invoke();
					}
				};
			}
		}
	}
}