/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Monitors.WPF.Model
{
	/// <summary>
	/// A <see cref="Tab"/> is uniquely identified by <see cref="Tab.Title"/>.
	/// </summary>
	public class Tab : IComparable, IComparable<Tab>
	{
		/// <summary>
		/// The name of the <see cref="Tab"/> in the TabView.
		/// </summary>
		public string Title { get; set; }

		/// <summary>
		/// Generate a <see cref="Tab"/> with a <b>unique</b> name.
		/// </summary>
		/// <param name="name">The name of the <see cref="Tab"/></param>
		public Tab(string name)
		{
			Title = name;
		}

		#region IComparable

		int IComparable.CompareTo(object obj)
		{
			if (obj == null)
			{
				return 1;
			}

			Tab other = obj as Tab;
			if (other != null)
			{
				return Title.CompareTo(other.Title);
			}
			else
			{
				throw new ArgumentException("Object is not a Tab");
			}
		}

		public int CompareTo(Tab other)
		{
			if (other == null)
			{
				return 1;
			}

			return Title.CompareTo(other.Title);
		}

		public override bool Equals(object obj)
		{
			return ((IComparable) this).CompareTo(obj) == 0;
		}

		public override int GetHashCode()
		{
			return Title.GetHashCode();
		}

		#endregion

		#region operators

		public static implicit operator Tab(string s)
		{
			return new Tab(s);
		}

		public static explicit operator string(Tab t)
		{
			return t.Title;
		}

		public static bool operator ==(Tab a, Tab b)
		{
			if (ReferenceEquals(a, null))
			{
				if (ReferenceEquals(b, null))
				{
					return true;
				}

				return false;
			}
			return a.CompareTo(b) == 0;
		}

		public static bool operator !=(Tab a, Tab b)
		{
			return !(a == b);
		}

		#endregion

		public override string ToString()
		{
			return Title;
		}
	}
}
