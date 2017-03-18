using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	//TODO: make single item parameter label for less overhead
	/// <summary>
	/// A descriptive information about other <see cref="IParameterVisualiser"/>(s). The style
	/// can change accordingly to the elements visualised. 
	/// </summary>
	public class ParameterLabel : Label, IParameterLabel
	{
		/// <summary>
		/// A set of visualisers that are currently pending.
		/// </summary>
		protected readonly ISet<IParameterVisualiser> PendingItems;
		/// <summary>
		/// A set of visualisers that are currently in the error state. 
		/// </summary>
		protected readonly ISet<IParameterVisualiser> ErroredItems;

		/// <summary>
		/// The style that will be applied to the label if all items are
		/// synchronised.
		/// </summary>
		public Style DefaultStyle { get; set; }

		/// <summary>
		/// The style that will be applied to the label if one or more items
		/// are pending.
		/// </summary>
		public Style PendingStyle { get; set; }

		/// <summary>
		/// The style that will be applied to the label if one or more items
		/// have errored.
		/// </summary>
		public Style ErroredStyle { get; set; }

		/// <summary>
		/// The default constructor of <see cref="ParameterLabel"/>. 
		/// All styles and the content is <c>null</c>.
		/// </summary>
		public ParameterLabel()
		{
			PendingItems = new HashSet<IParameterVisualiser>();
			ErroredItems = new HashSet<IParameterVisualiser>();
		}

		/// <summary>
		/// Create a <see cref="ParameterLabel"/> with a given content as text.
		/// All styles are <c>null</c>.
		/// </summary>
		/// <param name="text">The text that will become the labels content.</param>
		public ParameterLabel(string text) : this()
		{
			Content = text;
		}

		/// <summary>
		/// Set the state of a visualiser to pending.
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		public virtual void Pending(IParameterVisualiser visualiser)
		{
			PendingItems.Add(visualiser);
		}

		/// <summary>
		/// Set the state of a visualiser to errored.
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		public virtual void Errored(IParameterVisualiser visualiser)
		{
			PendingItems.Remove(visualiser);
			ErroredItems.Add(visualiser);
		}

		/// <summary>
		/// Set the state of a visualiser to success (i.e. don't care anymore).
		/// </summary>
		/// <param name="visualiser">The visualiser that has changed its state.</param>
		public virtual void Success(IParameterVisualiser visualiser)
		{
			PendingItems.Remove(visualiser);
			ErroredItems.Remove(visualiser);
		}

		/// <summary>
		/// Apply the errored style to this label.
		/// </summary>
		protected virtual void SetErroredStyle()
		{
			Style = ErroredStyle;
		}

		/// <summary>
		/// Apply the default style to this label.
		/// </summary>
		protected virtual void SetDefaultStyle()
		{
			Style = DefaultStyle;
		}

		/// <summary>
		/// Apply the pending style to this label.
		/// </summary>
		protected virtual void SetPendingStyle()
		{
			Style = PendingStyle;
		}

		/// <summary>
		/// Update the style to the correct current style (e.g. default, pending, error)
		/// </summary>
		protected virtual void UpdateStyle()
		{
			if (ErroredItems.Count > 0)
			{
				SetErroredStyle();
			} 
			else if (PendingItems.Count > 0)
			{
				SetPendingStyle();
			}
			else
			{
				SetDefaultStyle();
			}
		}
	}
}