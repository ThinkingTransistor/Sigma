//
// ImpObservableCollection.cs
//
// Copyright (c) 2010, Ashley Davis, @@email@@, @@website@@
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted 
// provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this list of conditions 
//   and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//   and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

using System;
using System.Collections;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Sigma.Core.Monitors.WPF.NetView.NetworkModel;
using Sigma.Core.Monitors.WPF.NetView.PositionManagement;

namespace Sigma.Core.Monitors.WPF.NetView
{
	/// <summary>
	/// Defines the current state of the mouse handling logic.
	/// </summary>
	internal enum MouseHandlingMode
	{
		/// <summary>
		/// Not in any special mode.
		/// </summary>
		None,

		/// <summary>
		/// Panning has been initiated and will commence when the use drags the cursor further than the threshold distance.
		/// </summary>
		Panning,

		/// <summary>
		/// The user is left-mouse-button-dragging to pan the viewport.
		/// </summary>
		DragPanning,

		/// <summary>
		/// The user is holding down shift and left-clicking or right-clicking to zoom in or out.
		/// </summary>
		Zooming,

		/// <summary>
		/// The user is holding down shift and left-mouse-button-dragging to select a region to zoom to.
		/// </summary>
		DragZooming,
	}

	public partial class NetLayout
	{
		/// <summary>
		/// Specifies the current state of the mouse handling logic.
		/// </summary>
		private MouseHandlingMode mouseHandlingMode = MouseHandlingMode.None;

		/// <summary>
		/// The point that was clicked relative to the ZoomAndPanControl.
		/// </summary>
		private Point origZoomAndPanControlMouseDownPoint;

		/// <summary>
		/// The point that was clicked relative to the content that is contained within the ZoomAndPanControl.
		/// </summary>
		private Point origContentMouseDownPoint;

		/// <summary>
		/// Records which mouse button clicked during mouse dragging.
		/// </summary>
		private MouseButton mouseButtonDown;

		/// <summary>
		/// Saves the previous zoom rectangle, pressing the backspace key jumps back to this zoom rectangle.
		/// </summary>
		private Rect prevZoomRect;

		/// <summary>
		/// Save the previous content scale, pressing the backspace key jumps back to this scale.
		/// </summary>
		private double prevZoomScale;

		/// <summary>
		/// Set to 'true' when the previous zoom rect is saved.
		/// </summary>
		private bool prevZoomRectSet = false;

		/// <summary>
		/// Event raised on mouse down in the NetworkView.
		/// </summary> 
		private void networkControl_MouseDown(object sender, MouseButtonEventArgs e)
		{
			networkControl.Focus();
			Keyboard.Focus(networkControl);

			mouseButtonDown = e.ChangedButton;
			origZoomAndPanControlMouseDownPoint = e.GetPosition(zoomAndPanControl);
			origContentMouseDownPoint = e.GetPosition(networkControl);

			if ((Keyboard.Modifiers & ModifierKeys.Shift) != 0 &&
				(e.ChangedButton == MouseButton.Left ||
				e.ChangedButton == MouseButton.Right))
			{
				// Shift + left- or right-down initiates zooming mode.
				mouseHandlingMode = MouseHandlingMode.Zooming;
			}
			else if (mouseButtonDown == MouseButton.Left &&
					(Keyboard.Modifiers & ModifierKeys.Control) == 0)
			{
				//
				// Initiate panning, when control is not held down.
				// When control is held down left dragging is used for drag selection.
				// After panning has been initiated the user must drag further than the threshold value to actually start drag panning.
				//
				mouseHandlingMode = MouseHandlingMode.Panning;
			}

			if (mouseHandlingMode != MouseHandlingMode.None)
			{
				// Capture the mouse so that we eventually receive the mouse up event.
				networkControl.CaptureMouse();
				e.Handled = true;
			}
		}

		/// <summary>
		/// Event raised on mouse up in the NetworkView.
		/// </summary>
		private void networkControl_MouseUp(object sender, MouseButtonEventArgs e)
		{
			if (mouseHandlingMode != MouseHandlingMode.None)
			{
				if (mouseHandlingMode == MouseHandlingMode.Panning)
				{
					//
					// Panning was initiated but dragging was abandoned before the mouse
					// cursor was dragged further than the threshold distance.
					// This means that this basically just a regular left mouse click.
					// Because it was a mouse click in empty space we need to clear the current selection.
					//
				}
				else if (mouseHandlingMode == MouseHandlingMode.Zooming)
				{
					if (mouseButtonDown == MouseButton.Left)
					{
						// Shift + left-click zooms in on the content.
						ZoomIn(origContentMouseDownPoint);
					}
					else if (mouseButtonDown == MouseButton.Right)
					{
						// Shift + left-click zooms out from the content.
						ZoomOut(origContentMouseDownPoint);
					}
				}
				else if (mouseHandlingMode == MouseHandlingMode.DragZooming)
				{
					// When drag-zooming has finished we zoom in on the rectangle that was highlighted by the user.
					ApplyDragZoomRect();
				}

				//
				// Reenable clearing of selection when empty space is clicked.
				// This is disabled when drag panning is in progress.
				//
				networkControl.IsClearSelectionOnEmptySpaceClickEnabled = true;

				//
				// Reset the override cursor.
				// This is set to a special cursor while drag panning is in progress.
				//
				Mouse.OverrideCursor = null;

				networkControl.ReleaseMouseCapture();
				mouseHandlingMode = MouseHandlingMode.None;
				e.Handled = true;
			}
		}

		/// <summary>
		/// Event raised on mouse move in the NetworkView.
		/// </summary>
		private void networkControl_MouseMove(object sender, MouseEventArgs e)
		{
			if (mouseHandlingMode == MouseHandlingMode.Panning)
			{
				Point curZoomAndPanControlMousePoint = e.GetPosition(zoomAndPanControl);
				Vector dragOffset = curZoomAndPanControlMousePoint - origZoomAndPanControlMouseDownPoint;
				double dragThreshold = 10;
				if (Math.Abs(dragOffset.X) > dragThreshold ||
					Math.Abs(dragOffset.Y) > dragThreshold)
				{
					//
					// The user has dragged the cursor further than the threshold distance, initiate
					// drag panning.
					//
					mouseHandlingMode = MouseHandlingMode.DragPanning;
					networkControl.IsClearSelectionOnEmptySpaceClickEnabled = false;
					Mouse.OverrideCursor = Cursors.ScrollAll;
				}

				e.Handled = true;
			}
			else if (mouseHandlingMode == MouseHandlingMode.DragPanning)
			{
				//
				// The user is left-dragging the mouse.
				// Pan the viewport by the appropriate amount.
				//
				Point curContentMousePoint = e.GetPosition(networkControl);
				Vector dragOffset = curContentMousePoint - origContentMouseDownPoint;

				zoomAndPanControl.ContentOffsetX -= dragOffset.X;
				zoomAndPanControl.ContentOffsetY -= dragOffset.Y;

				e.Handled = true;
			}
			else if (mouseHandlingMode == MouseHandlingMode.Zooming)
			{
				Point curZoomAndPanControlMousePoint = e.GetPosition(zoomAndPanControl);
				Vector dragOffset = curZoomAndPanControlMousePoint - origZoomAndPanControlMouseDownPoint;
				double dragThreshold = 10;
				if (mouseButtonDown == MouseButton.Left &&
					(Math.Abs(dragOffset.X) > dragThreshold ||
					Math.Abs(dragOffset.Y) > dragThreshold))
				{
					//
					// When Shift + left-down zooming mode and the user drags beyond the drag threshold,
					// initiate drag zooming mode where the user can drag out a rectangle to select the area
					// to zoom in on.
					//
					mouseHandlingMode = MouseHandlingMode.DragZooming;
					Point curContentMousePoint = e.GetPosition(networkControl);
					InitDragZoomRect(origContentMouseDownPoint, curContentMousePoint);
				}

				e.Handled = true;
			}
			else if (mouseHandlingMode == MouseHandlingMode.DragZooming)
			{
				//
				// When in drag zooming mode continously update the position of the rectangle
				// that the user is dragging out.
				//
				Point curContentMousePoint = e.GetPosition(networkControl);
				SetDragZoomRect(origContentMouseDownPoint, curContentMousePoint);

				e.Handled = true;
			}
		}

		/// <summary>
		/// Event raised by rotating the mouse wheel.
		/// </summary>
		private void networkControl_MouseWheel(object sender, MouseWheelEventArgs e)
		{
			e.Handled = true;

			if (e.Delta > 0)
			{
				Point curContentMousePoint = e.GetPosition(networkControl);
				ZoomIn(curContentMousePoint);
			}
			else if (e.Delta < 0)
			{
				Point curContentMousePoint = e.GetPosition(networkControl);
				ZoomOut(curContentMousePoint);
			}
		}

		/// <summary>
		/// Event raised when the user has double clicked in the zoom and pan control.
		/// </summary>
		private void networkControl_MouseDoubleClick(object sender, MouseButtonEventArgs e)
		{
			if ((Keyboard.Modifiers & ModifierKeys.Shift) == 0)
			{
				Point doubleClickPoint = e.GetPosition(networkControl);
				zoomAndPanControl.AnimatedSnapTo(doubleClickPoint);
			}
		}

		/// <summary>
		/// The 'ZoomIn' command (bound to the plus key) was executed.
		/// </summary>
		private void ZoomIn_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			var o = networkControl.SelectedNode;

			ZoomIn(new Point(zoomAndPanControl.ContentZoomFocusX, zoomAndPanControl.ContentZoomFocusY));
		}

		/// <summary>
		/// The 'ZoomOut' command (bound to the minus key) was executed.
		/// </summary>
		private void ZoomOut_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			ZoomOut(new Point(zoomAndPanControl.ContentZoomFocusX, zoomAndPanControl.ContentZoomFocusY));
		}

		/// <summary>
		/// The 'JumpBackToPrevZoom' command was executed.
		/// </summary>
		private void JumpBackToPrevZoom_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			JumpBackToPrevZoom();
		}

		/// <summary>
		/// Determines whether the 'JumpBackToPrevZoom' command can be executed.
		/// </summary>
		private void JumpBackToPrevZoom_CanExecuted(object sender, CanExecuteRoutedEventArgs e)
		{
			e.CanExecute = prevZoomRectSet;
		}

		/// <summary>
		/// The 'Fill' command was executed.
		/// </summary>
		private void FitContent_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			IList nodes = null;

			if (networkControl.SelectedNodes.Count > 0)
			{
				nodes = networkControl.SelectedNodes;
			}
			else
			{
				nodes = ViewModel.Network.Nodes;
				if (nodes.Count == 0)
				{
					return;
				}
			}

			SavePrevZoomRect();

			Rect actualContentRect = DetermineAreaOfNodes(nodes);

			//
			// Inflate the content rect by a fraction of the actual size of the total content area.
			// This puts a nice border around the content we are fitting to the viewport.
			//
			actualContentRect.Inflate(networkControl.ActualWidth / 40, networkControl.ActualHeight / 40);

			zoomAndPanControl.AnimatedZoomTo(actualContentRect);
		}

		/// <summary>
		/// Determine the area covered by the specified list of nodes.
		/// </summary>
		private Rect DetermineAreaOfNodes(IList nodes)
		{
			NodeViewModel firstNode = (NodeViewModel)nodes[0];
			Rect actualContentRect = new Rect(firstNode.X, firstNode.Y, firstNode.Size.Width, firstNode.Size.Height);

			for (int i = 1; i < nodes.Count; ++i)
			{
				NodeViewModel node = (NodeViewModel)nodes[i];
				Rect nodeRect = new Rect(node.X, node.Y, node.Size.Width, node.Size.Height);
				actualContentRect = Rect.Union(actualContentRect, nodeRect);
			}
			return actualContentRect;
		}

		/// <summary>
		/// The 'Fill' command was executed.
		/// </summary>
		private void Fill_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			SavePrevZoomRect();

			zoomAndPanControl.AnimatedScaleToFit();
		}

		/// <summary>
		/// The 'OneHundredPercent' command was executed.
		/// </summary>
		private void OneHundredPercent_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			SavePrevZoomRect();

			zoomAndPanControl.AnimatedZoomTo(1.0);
		}

		/// <summary>
		/// Jump back to the previous zoom level.
		/// </summary>
		private void JumpBackToPrevZoom()
		{
			zoomAndPanControl.AnimatedZoomTo(prevZoomScale, prevZoomRect);

			ClearPrevZoomRect();
		}

		/// <summary>
		/// Zoom the viewport out, centering on the specified point (in content coordinates).
		/// </summary>
		private void ZoomOut(Point contentZoomCenter)
		{
			zoomAndPanControl.ZoomAboutPoint(zoomAndPanControl.ContentScale - 0.1, contentZoomCenter);
		}

		/// <summary>
		/// Zoom the viewport in, centering on the specified point (in content coordinates).
		/// </summary>
		private void ZoomIn(Point contentZoomCenter)
		{
			zoomAndPanControl.ZoomAboutPoint(zoomAndPanControl.ContentScale + 0.1, contentZoomCenter);
		}

		/// <summary>
		/// Initialize the rectangle that the use is dragging out.
		/// </summary>
		private void InitDragZoomRect(Point pt1, Point pt2)
		{
			SetDragZoomRect(pt1, pt2);

			dragZoomCanvas.Visibility = Visibility.Visible;
			dragZoomBorder.Opacity = 0.5;
		}

		/// <summary>
		/// Update the position and size of the rectangle that user is dragging out.
		/// </summary>
		private void SetDragZoomRect(Point pt1, Point pt2)
		{
			double x, y, width, height;

			//
			// Deterine x,y,width and height of the rect inverting the points if necessary.
			// 

			if (pt2.X < pt1.X)
			{
				x = pt2.X;
				width = pt1.X - pt2.X;
			}
			else
			{
				x = pt1.X;
				width = pt2.X - pt1.X;
			}

			if (pt2.Y < pt1.Y)
			{
				y = pt2.Y;
				height = pt1.Y - pt2.Y;
			}
			else
			{
				y = pt1.Y;
				height = pt2.Y - pt1.Y;
			}

			//
			// Update the coordinates of the rectangle that is being dragged out by the user.
			// The we offset and rescale to convert from content coordinates.
			//
			Canvas.SetLeft(dragZoomBorder, x);
			Canvas.SetTop(dragZoomBorder, y);
			dragZoomBorder.Width = width;
			dragZoomBorder.Height = height;
		}

		/// <summary>
		/// When the user has finished dragging out the rectangle the zoom operation is applied.
		/// </summary>
		private void ApplyDragZoomRect()
		{
			//
			// Record the previous zoom level, so that we can jump back to it when the backspace key is pressed.
			//
			SavePrevZoomRect();

			//
			// Retreive the rectangle that the user draggged out and zoom in on it.
			//
			double contentX = Canvas.GetLeft(dragZoomBorder);
			double contentY = Canvas.GetTop(dragZoomBorder);
			double contentWidth = dragZoomBorder.Width;
			double contentHeight = dragZoomBorder.Height;
			zoomAndPanControl.AnimatedZoomTo(new Rect(contentX, contentY, contentWidth, contentHeight));

			FadeOutDragZoomRect();
		}

		//
		// Fade out the drag zoom rectangle.
		//
		private void FadeOutDragZoomRect()
		{
			AnimationHelper.StartAnimation(dragZoomBorder, OpacityProperty, 0.0, 0.1,
				delegate (object sender, EventArgs e)
				{
					dragZoomCanvas.Visibility = Visibility.Collapsed;
				});
		}

		//
		// Record the previous zoom level, so that we can jump back to it when the backspace key is pressed.
		//
		private void SavePrevZoomRect()
		{
			prevZoomRect = new Rect(zoomAndPanControl.ContentOffsetX, zoomAndPanControl.ContentOffsetY, zoomAndPanControl.ContentViewportWidth, zoomAndPanControl.ContentViewportHeight);
			prevZoomScale = zoomAndPanControl.ContentScale;
			prevZoomRectSet = true;
		}

		/// <summary>
		/// Clear the memory of the previous zoom level.
		/// </summary>
		private void ClearPrevZoomRect()
		{
			prevZoomRectSet = false;
		}
	}
}