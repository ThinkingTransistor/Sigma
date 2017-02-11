/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.TitleBar
{
	/// <summary>
	/// This "box" can display an about message with an image, heading, text, close button.
	/// </summary>
	public class SigmaAboutBox : Control
	{
		/// <summary>
		/// The heading of the box. 
		/// </summary>
		public string Heading
		{
			get { return (string) GetValue(HeadingProperty); }
			set { SetValue(HeadingProperty, value); }
		}

		public static readonly DependencyProperty HeadingProperty =
			DependencyProperty.Register("Heading", typeof(string), typeof(SigmaAboutBox), new PropertyMetadata(""));

		/// <summary>
		/// A descriptive text (can be multiple sentences)
		/// </summary>
		public string Text
		{
			get { return (string) GetValue(TextProperty); }
			set { SetValue(TextProperty, value); }
		}

		public static readonly DependencyProperty TextProperty =
			DependencyProperty.Register("Text", typeof(string), typeof(SigmaAboutBox), new PropertyMetadata(""));


		/// <summary>
		/// An image (normally the sigma logo)
		/// </summary>
		public ImageSource Image
		{
			get { return (ImageSource) GetValue(ImageProperty); }
			set { SetValue(ImageProperty, value); }
		}

		public static readonly DependencyProperty ImageProperty =
			DependencyProperty.Register("Image", typeof(ImageSource), typeof(SigmaAboutBox), new PropertyMetadata(null));

		/// <summary>
		/// The width of the image. 
		/// </summary>
		public double ImageWidth
		{
			get { return (double) GetValue(ImageWidthProperty); }
			set { SetValue(ImageWidthProperty, value); }
		}

		public static readonly DependencyProperty ImageWidthProperty =
			DependencyProperty.Register("ImageWidth", typeof(double), typeof(SigmaAboutBox), new PropertyMetadata(double.NaN));

		/// <summary>
		/// The height of the image. 
		/// </summary>
		public double ImageHeight
		{
			get { return (double) GetValue(ImageHeightProperty); }
			set { SetValue(ImageHeightProperty, value); }
		}

		public static readonly DependencyProperty ImageHeightProperty =
			DependencyProperty.Register("ImageHeight", typeof(double), typeof(SigmaAboutBox), new PropertyMetadata(double.NaN));

		/// <summary>
		/// The content that is inside the "close window"-button. 
		/// </summary>
		public object ButtonContent
		{
			get { return GetValue(ButtonContentProperty); }
			set { SetValue(ButtonContentProperty, value); }
		}

		public static readonly DependencyProperty ButtonContentProperty =
			DependencyProperty.Register("ButtonContent", typeof(object), typeof(SigmaAboutBox), new PropertyMetadata(null));

		/// <summary>
		/// This command will be used, when the "close window"-button was pressed. 
		/// </summary>
		public CloseCommand Close
		{
			get { return (CloseCommand) GetValue(CloseProperty); }
			set { SetValue(CloseProperty, value); }
		}

		public static readonly DependencyProperty CloseProperty =
			DependencyProperty.Register("Close", typeof(CloseCommand), typeof(SigmaAboutBox), new PropertyMetadata(new CloseCommand()));

		static SigmaAboutBox()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(SigmaAboutBox), new FrameworkPropertyMetadata(typeof(SigmaAboutBox)));
		}

		/// <summary>
		/// The <see cref="DialogHost"/>, that is used to close the dialogue on user interaction.
		/// </summary>
		public DialogHost DialogHost { get; set; }

		public SigmaAboutBox()
		{
			Close.Box = this;
		}

		/// <summary>
		/// This method is called when the dialogue should be closed.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		public virtual void CloseDialogue(object sender, RoutedEventArgs e)
		{
			DialogHost.IsOpen = false;
		}

		public class CloseCommand : ICommand
		{
			/// <summary>
			/// The box this <see cref="SigmaAboutBox"/> belongs to.
			/// </summary>
			public SigmaAboutBox Box { get; set; }

			/// <summary>Occurs when changes occur that affect whether or not the command should execute.</summary>
			public event EventHandler CanExecuteChanged;

			/// <summary>Defines the method that determines whether the command can execute in its current state.</summary>
			/// <returns>true if this command can be executed; otherwise, false.</returns>
			/// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to null.</param>
			public virtual bool CanExecute(object parameter)
			{
				return true;
			}

			/// <summary>Defines the method to be called when the command is invoked.</summary>
			/// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to null.</param>
			public virtual void Execute(object parameter)
			{
				Box.DialogHost.IsOpen = false;
			}

		}
	}
}
