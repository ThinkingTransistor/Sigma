using System;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	//TODO: editable time box (timepicker)
	//[ParameterVisualiser(typeof(DateTime), Priority = VisualiserPriority.Lower)]
	//public class SigmaTimeBox : SigmaTextBox
	//{
		
	//}

	/// <summary>
	/// A TimeBlock that allows to display the current time. 
	/// </summary>
	[ParameterVisualiser(typeof(DateTime), Priority = VisualiserPriority.Lower)]
	public class SigmaTimeBlock : SigmaTextBlock
	{
		/// <inheritdoc />
		public override object Object
		{
			get { return _Object; }
			set
			{
				if (_Object is DateTime)
				{
					_Object = value;
					TextBlock.Text = ((DateTime) Object).ToString(FormatString);
				}
				else
				{
					base.Object = value;
				}
			}
		}

		/// <summary>
		/// The string that is used to format the time. 
		/// <c>null</c>, if default formatting should be applied.
		/// </summary>
		public string FormatString { get; set; }

		/// <summary>
		/// Create a label that is capable of displaying a time.
		/// </summary>
		public SigmaTimeBlock()
		{
			base.IsReadOnly = true;
		}

		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public sealed override bool IsReadOnly
		{
			get { return base.IsReadOnly; }
			set { throw new NotImplementedException(); }
		}
	}
}