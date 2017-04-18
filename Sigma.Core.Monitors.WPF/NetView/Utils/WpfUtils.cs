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
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.NetView.Utils
{
	/// <summary>
	///     This class contains helper functions for dealing with WPF.
	/// </summary>
	public static class WpfUtils
	{
		/// <summary>
		///     Search up the element tree to find the Parent window for 'element'.
		///     Returns null if the 'element' is not attached to a window.
		/// </summary>
		public static Window FindParentWindow(FrameworkElement element)
		{
			if (element.Parent == null)
				return null;

			Window window = element.Parent as Window;
			if (window != null)
				return window;

			FrameworkElement parentElement = element.Parent as FrameworkElement;
			if (parentElement != null)
			{
				return FindParentWindow(parentElement);
			}

			return null;
		}

		public static FrameworkElement FindParentWithDataContextAndName<TDataContext>(FrameworkElement childElement,
			string name)
			where TDataContext : class
		{
			FrameworkElement parent = (FrameworkElement) childElement.Parent;
			if (parent != null)
			{
				TDataContext data = parent.DataContext as TDataContext;
				if (data != null)
					if (parent.Name == name)
						return parent;

				parent = FindParentWithDataContextAndName<TDataContext>(parent, name);
				if (parent != null)
					return parent;
			}

			parent = (FrameworkElement) childElement.TemplatedParent;
			if (parent != null)
			{
				TDataContext data = parent.DataContext as TDataContext;
				if (data != null)
					if (parent.Name == name)
						return parent;

				parent = FindParentWithDataContextAndName<TDataContext>(parent, name);
				if (parent != null)
					return parent;
			}

			return null;
		}

		public static FrameworkElement FindParentWithDataContext<TDataContext>(FrameworkElement childElement)
			where TDataContext : class
		{
			if (childElement.Parent != null)
			{
				TDataContext data = ((FrameworkElement) childElement.Parent).DataContext as TDataContext;
				if (data != null)
					return (FrameworkElement) childElement.Parent;

				FrameworkElement parent = FindParentWithDataContext<TDataContext>((FrameworkElement) childElement.Parent);
				if (parent != null)
					return parent;
			}

			if (childElement.TemplatedParent != null)
			{
				TDataContext data = ((FrameworkElement) childElement.TemplatedParent).DataContext as TDataContext;
				if (data != null)
					return (FrameworkElement) childElement.TemplatedParent;

				FrameworkElement parent = FindParentWithDataContext<TDataContext>((FrameworkElement) childElement.TemplatedParent);
				if (parent != null)
					return parent;
			}

			return null;
		}

		public static TParent FindVisualParentWithType<TParent>(FrameworkElement childElement)
			where TParent : class
		{
			FrameworkElement parentElement = (FrameworkElement) VisualTreeHelper.GetParent(childElement);
			if (parentElement != null)
			{
				TParent parent = parentElement as TParent;
				if (parent != null)
					return parent;

				return FindVisualParentWithType<TParent>(parentElement);
			}

			return null;
		}

		public static TParent FindParentWithType<TParent>(FrameworkElement childElement)
			where TParent : class
		{
			if (childElement.Parent != null)
			{
				TParent parent = childElement.Parent as TParent;
				if (parent != null)
					return parent;

				parent = FindParentWithType<TParent>((FrameworkElement) childElement.Parent);
				if (parent != null)
					return parent;
			}

			if (childElement.TemplatedParent != null)
			{
				TParent parent = childElement.TemplatedParent as TParent;
				if (parent != null)
					return parent;

				parent = FindParentWithType<TParent>((FrameworkElement) childElement.TemplatedParent);
				if (parent != null)
					return parent;
			}

			FrameworkElement parentElement = (FrameworkElement) VisualTreeHelper.GetParent(childElement);
			if (parentElement != null)
			{
				TParent parent = parentElement as TParent;
				if (parent != null)
					return parent;

				return FindParentWithType<TParent>(parentElement);
			}

			return null;
		}

		public static TParent FindParentWithTypeAndDataContext<TParent>(FrameworkElement childElement, object dataContext)
			where TParent : FrameworkElement
		{
			if (childElement.Parent != null)
			{
				TParent parent = childElement.Parent as TParent;
				if (parent != null)
					if (parent.DataContext == dataContext)
						return parent;

				parent = FindParentWithTypeAndDataContext<TParent>((FrameworkElement) childElement.Parent, dataContext);
				if (parent != null)
					return parent;
			}

			if (childElement.TemplatedParent != null)
			{
				TParent parent = childElement.TemplatedParent as TParent;
				if (parent != null)
					if (parent.DataContext == dataContext)
						return parent;

				parent = FindParentWithTypeAndDataContext<TParent>((FrameworkElement) childElement.TemplatedParent, dataContext);
				if (parent != null)
					return parent;
			}

			FrameworkElement parentElement = (FrameworkElement) VisualTreeHelper.GetParent(childElement);
			if (parentElement != null)
			{
				TParent parent = parentElement as TParent;
				if (parent != null)
					return parent;

				return FindParentWithType<TParent>(parentElement);
			}

			return null;
		}

		/// <summary>
		///     Hit test against the specified element for a child that has a data context
		///     of the specified type.
		///     Returns 'null' if nothing was 'hit'.
		///     Return the highest level element that matches the hit test.
		/// </summary>
		public static T HitTestHighestForDataContext<T>(FrameworkElement rootElement, Point point)
			where T : class
		{
			FrameworkElement hitFrameworkElement;
			return HitTestHighestForDataContext<T>(rootElement, point, out hitFrameworkElement);
		}

		/// <summary>
		///     Hit test against the specified element for a child that has a data context
		///     of the specified type.
		///     Returns 'null' if nothing was 'hit'.
		///     Return the highest level element that matches the hit test.
		/// </summary>
		public static T HitTestHighestForDataContext<T>(FrameworkElement rootElement,
			Point point, out FrameworkElement hitFrameworkElement)
			where T : class
		{
			hitFrameworkElement = null;

			FrameworkElement hitElement;
			T hitData = HitTestForDataContext<T, FrameworkElement>(rootElement, point, out hitElement);
			if (hitData == null)
				return null;

			hitFrameworkElement = hitElement;

			//
			// Find the highest level parent below root element that still matches the data context.
			while (hitElement != null && !Equals(hitElement, rootElement) &&
					hitElement.DataContext == hitData)
			{
				hitFrameworkElement = hitElement;

				if (hitElement.Parent != null)
				{
					hitElement = hitElement.Parent as FrameworkElement;
					continue;
				}

				if (hitElement.TemplatedParent != null)
				{
					hitElement = hitElement.TemplatedParent as FrameworkElement;
					continue;
				}

				break;
			}

			return hitData;
		}


		/// <summary>
		///     Hit test for a specific data context and name.
		/// </summary>
		public static TDataContext HitTestForDataContextAndName<TDataContext, TEleemnt>(FrameworkElement rootElement,
			Point point, string name, out TEleemnt hitFrameworkElement)
			where TDataContext : class
			where TEleemnt : FrameworkElement
		{
			TDataContext data = null;
			TEleemnt frameworkElement = null;

			VisualTreeHelper.HitTest(
				rootElement,
				// Hit test filter.
				null,
				// Hit test result.
				delegate(HitTestResult result)
				{
					frameworkElement = result.VisualHit as TEleemnt;
					if (frameworkElement != null)
					{
						data = frameworkElement.DataContext as TDataContext;
						if (data != null)
							if (frameworkElement.Name == name)
								return HitTestResultBehavior.Stop;
					}

					return HitTestResultBehavior.Continue;
				},
				new PointHitTestParameters(point));

			hitFrameworkElement = frameworkElement;
			return data;
		}


		/// <summary>
		///     Hit test against the specified element for a child that has a data context
		///     of the specified type.
		///     Returns 'null' if nothing was 'hit'.
		/// </summary>
		public static TDataContext HitTestForDataContext<TDataContext, TElement>(FrameworkElement rootElement,
			Point point, out TElement hitFrameworkElement)
			where TDataContext : class
			where TElement : FrameworkElement
		{
			TDataContext data = null;
			TElement frameworkElement = null;

			VisualTreeHelper.HitTest(
				rootElement,
				// Hit test filter.
				null,
				// Hit test result.
				delegate(HitTestResult result)
				{
					frameworkElement = result.VisualHit as TElement;
					if (frameworkElement != null)
					{
						data = frameworkElement.DataContext as TDataContext;
						return data != null ? HitTestResultBehavior.Stop : HitTestResultBehavior.Continue;
					}

					return HitTestResultBehavior.Continue;
				},
				new PointHitTestParameters(point));

			hitFrameworkElement = frameworkElement;
			return data;
		}

		/// <summary>
		///     Find the ancestor of a particular element based on the type of the ancestor.
		/// </summary>
		public static T FindAncestor<T>(FrameworkElement element) where T : class
		{
			if (element.Parent != null)
			{
				T ancestor = element.Parent as T;
				if (ancestor != null)
					return ancestor;

				FrameworkElement parent = element.Parent as FrameworkElement;
				if (parent != null)
					return FindAncestor<T>(parent);
			}

			if (element.TemplatedParent != null)
			{
				T ancestor = element.TemplatedParent as T;
				if (ancestor != null)
					return ancestor;

				FrameworkElement parent = element.TemplatedParent as FrameworkElement;
				if (parent != null)
					return FindAncestor<T>(parent);
			}

			DependencyObject visualParent = VisualTreeHelper.GetParent(element);
			if (visualParent != null)
			{
				T visualAncestor = visualParent as T;
				if (visualAncestor != null)
					return visualAncestor;

				FrameworkElement visualElement = visualParent as FrameworkElement;
				if (visualElement != null)
					return FindAncestor<T>(visualElement);
			}

			return null;
		}

		/// <summary>
		///     Transform a point to an ancestors coordinate system.
		/// </summary>
		public static Point TransformPointToAncestor<T>(FrameworkElement element, Point point) where T : Visual
		{
			T ancestor = FindAncestor<T>(element);
			if (ancestor == null)
				throw new ApplicationException("Find to find '" + typeof(T).Name + "' for element '" + element.GetType().Name + "'.");

			return TransformPointToAncestor(ancestor, element, point);
		}

		/// <summary>
		///     Transform a point to an ancestors coordinate system.
		/// </summary>
		public static Point TransformPointToAncestor(Visual ancestor, FrameworkElement element, Point point)
		{
			return element.TransformToAncestor(ancestor).Transform(point);
		}

		/// <summary>
		///     Find the framework element with the specified name.
		/// </summary>
		public static TElement FindElementWithName<TElement>(Visual rootElement, string name)
			where TElement : FrameworkElement
		{
			FrameworkElement rootFrameworkElement = rootElement as FrameworkElement;
			rootFrameworkElement?.UpdateLayout();

			int numChildren = VisualTreeHelper.GetChildrenCount(rootElement);
			for (int i = 0; i < numChildren; ++i)
			{
				Visual childElement = (Visual) VisualTreeHelper.GetChild(rootElement, i);

				TElement typedChildElement = childElement as TElement;
				if (typedChildElement != null)
					if (typedChildElement.Name == name)
						return typedChildElement;

				TElement foundElement = FindElementWithName<TElement>(childElement, name);
				if (foundElement != null)
					return foundElement;
			}

			return null;
		}

		/// <summary>
		///     Find the framework element for the specified connector.
		/// </summary>
		public static TElement FindElementWithDataContextAndName<TDataContext, TElement>(Visual rootElement, TDataContext data,
			string name)
			where TDataContext : class
			where TElement : FrameworkElement
		{
			Trace.Assert(rootElement != null);

			FrameworkElement rootFrameworkElement = rootElement as FrameworkElement;
			if (rootFrameworkElement != null)
				rootFrameworkElement.UpdateLayout();

			int numChildren = VisualTreeHelper.GetChildrenCount(rootElement);
			for (int i = 0; i < numChildren; ++i)
			{
				Visual childElement = (Visual) VisualTreeHelper.GetChild(rootElement, i);

				TElement typedChildElement = childElement as TElement;
				if (typedChildElement != null &&
					typedChildElement.DataContext == data)
					if (typedChildElement.Name == name)
						return typedChildElement;

				TElement foundElement = FindElementWithDataContextAndName<TDataContext, TElement>(childElement, data, name);
				if (foundElement != null)
					return foundElement;
			}

			return null;
		}

		/// <summary>
		///     Find the framework element for the specified connector.
		/// </summary>
		public static TElement FindElementWithType<TElement>(Visual rootElement)
			where TElement : FrameworkElement
		{
			if (rootElement == null)
				throw new ArgumentNullException(nameof(rootElement));

			FrameworkElement rootFrameworkElement = rootElement as FrameworkElement;
			rootFrameworkElement?.UpdateLayout();

			//
			// Check each child.
			//
			int numChildren = VisualTreeHelper.GetChildrenCount(rootElement);
			for (int i = 0; i < numChildren; ++i)
			{
				Visual childElement = (Visual) VisualTreeHelper.GetChild(rootElement, i);

				TElement typedChildElement = childElement as TElement;
				if (typedChildElement != null)
					return typedChildElement;
			}

			//
			// Check sub-trees.
			//
			for (int i = 0; i < numChildren; ++i)
			{
				Visual childElement = (Visual) VisualTreeHelper.GetChild(rootElement, i);

				TElement foundElement = FindElementWithType<TElement>(childElement);
				if (foundElement != null)
					return foundElement;
			}

			return null;
		}

		/// <summary>
		///     Find the framework element for the specified connector.
		/// </summary>
		public static TElement FindElementWithDataContext<TDataContext, TElement>(Visual rootElement, TDataContext data)
			where TDataContext : class
			where TElement : FrameworkElement
		{
			if (rootElement == null)
				throw new ArgumentNullException(nameof(rootElement));

			FrameworkElement rootFrameworkElement = rootElement as FrameworkElement;
			rootFrameworkElement?.UpdateLayout();

			int numChildren = VisualTreeHelper.GetChildrenCount(rootElement);
			for (int i = 0; i < numChildren; ++i)
			{
				Visual childElement = (Visual) VisualTreeHelper.GetChild(rootElement, i);

				TElement typedChildElement = childElement as TElement;
				if (typedChildElement != null &&
					typedChildElement.DataContext == data)
					return typedChildElement;

				TElement foundElement = FindElementWithDataContext<TDataContext, TElement>(childElement, data);
				if (foundElement != null)
					return foundElement;
			}

			return null;
		}

		/// <summary>
		///     Walk up the visual tree and find a template for the specified type.
		///     Returns null if none was found.
		/// </summary>
		public static TDataTemplate FindTemplateForType<TDataTemplate>(Type type, FrameworkElement element)
			where TDataTemplate : class
		{
			object resource = element.TryFindResource(new DataTemplateKey(type));
			TDataTemplate dataTemplate = resource as TDataTemplate;
			if (dataTemplate != null)
				return dataTemplate;

			if (type.BaseType != null &&
				type.BaseType != typeof(object))
			{
				dataTemplate = FindTemplateForType<TDataTemplate>(type.BaseType, element);
				if (dataTemplate != null)
					return dataTemplate;
			}

			foreach (Type interfaceType in type.GetInterfaces())
			{
				dataTemplate = FindTemplateForType<TDataTemplate>(interfaceType, element);
				if (dataTemplate != null)
					return dataTemplate;
			}

			return null;
		}

		/// <summary>
		///     Search the visual tree for template and instance it.
		/// </summary>
		public static FrameworkElement CreateVisual(Type type, FrameworkElement element, object dataContext)
		{
			DataTemplate template = FindTemplateForType<DataTemplate>(type, element);
			if (template == null)
				throw new ApplicationException("Failed to find DataTemplate for type " + type.Name);

			FrameworkElement visual = (FrameworkElement) template.LoadContent();
			visual.Resources = element.Resources;
			visual.DataContext = dataContext;
			return visual;
		}

		/// <summary>
		///     Layout, measure and arrange the specified element.
		/// </summary>
		public static void InitaliseElement(FrameworkElement element)
		{
			element.UpdateLayout();
			element.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
			element.Arrange(new Rect(0, 0, element.DesiredSize.Width, element.DesiredSize.Height));
		}


		/// <summary>
		///     Finds a particular type of UI element int he visual tree that has the specified data context.
		/// </summary>
		public static ICollection<T> FindTypedElements<T>(DependencyObject rootElement) where T : DependencyObject
		{
			List<T> foundElements = new List<T>();
			FindTypedElements(rootElement, foundElements);
			return foundElements;
		}

		/// <summary>
		///     Finds a particular type of UI element int he visual tree that has the specified data context.
		/// </summary>
		private static void FindTypedElements<T>(DependencyObject rootElement, List<T> foundElements)
			where T : DependencyObject
		{
			int numChildren = VisualTreeHelper.GetChildrenCount(rootElement);
			for (int i = 0; i < numChildren; ++i)
			{
				DependencyObject childElement = VisualTreeHelper.GetChild(rootElement, i);
				if (childElement is T)
				{
					foundElements.Add((T) childElement);
					continue;
				}

				FindTypedElements(childElement, foundElements);
			}
		}

		/// <summary>
		///     Recursively dump out all elements in the visual tree.
		/// </summary>
		public static void DumpVisualTree(Visual root)
		{
			DumpVisualTree(root, 0);
		}

		/// <summary>
		///     Recursively dump out all elements in the visual tree.
		/// </summary>
		private static void DumpVisualTree(Visual root, int indentLevel)
		{
			string indentStr = new string(' ', indentLevel * 2);
			Trace.Write(indentStr);
			Trace.Write(root.GetType().Name);

			FrameworkElement rootElement = root as FrameworkElement;
			if (rootElement != null)
				if (rootElement.DataContext != null)
				{
					Trace.Write(" [");
					Trace.Write(rootElement.DataContext.GetType().Name);
					Trace.Write("]");
				}

			Trace.WriteLine("");

			int numChildren = VisualTreeHelper.GetChildrenCount(root);
			if (numChildren > 0)
			{
				Trace.Write(indentStr);
				Trace.WriteLine("{");

				for (int i = 0; i < numChildren; ++i)
				{
					Visual child = (Visual) VisualTreeHelper.GetChild(root, i);
					DumpVisualTree(child, indentLevel + 1);
				}

				Trace.Write(indentStr);
				Trace.WriteLine("}");
			}
		}
	}
}