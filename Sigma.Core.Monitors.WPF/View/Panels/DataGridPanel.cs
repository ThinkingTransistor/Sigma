using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public class DataGridPanel : SigmaPanel
	{
		public new DataGrid Content { get; }

		private Dictionary<string, List<object>> dataDictionary;

		private List<FrameworkElementFactory> _imageFactories;

		public DataGridPanel(string title) : base(title)
		{
			dataDictionary = new Dictionary<string, List<object>>();
			_imageFactories = new List<FrameworkElementFactory>();

			Content = new DataGrid
			{
				IsReadOnly = true
			};
		}

		//public DataGridPanel(string title, object header1, Type type1) : this(title)
		//{
		//	AddColumn(header1, type1);
		//}

		//public DataGridPanel(string title, object header1, Type type1, object header2, Type type2) : this(title, header1, type1)
		//{
		//	AddColumn(header2, type2);
		//}

		//public DataGridPanel(string title, object header1, Type type1, object header2, Type type2, object header3, Type type3)
		//	: this(title, header1, type1, header2, type2)
		//{
		//	AddColumn(header3, type3);
		//}

		//public DataGridPanel(string title, object header1, Type type1, object header2, Type type2, object header3, Type type3,
		//	params object[] columns) : this(title, header1, type1, header2, type2, header3, type3)
		//{
		//	if (columns.Length % 2 != 0) throw new ArgumentException(nameof(columns));

		//	for (int i = 0; i < columns.Length; i++)
		//	{
		//		AddColumn(columns[i], (Type) columns[++i]);
		//	}

		//	base.Content = Content;
		//}

		public List<object> AddColumn(object header, string headerIdentifier, Type type)
		{
			if (type == typeof(Image))
			{
				return AddImageColumn(header, headerIdentifier);
			}
			else
			{
				return AddTextColumn(header);
			}
		}


		public List<object> AddTextColumn(object header)
		{
			throw new NotImplementedException();
		}

		public List<object> AddImageColumn(string header)
		{
			return AddImageColumn(header, header);
		}


		public List<object> AddImageColumn(object header, string headerIdentifier)
		{
			if (headerIdentifier == null) throw new ArgumentNullException(nameof(headerIdentifier));

			List<object> dataBinding = AddSource(headerIdentifier);

			DataGridTemplateColumn column = new DataGridTemplateColumn { Header = header };
			FrameworkElementFactory factory = new FrameworkElementFactory(typeof(Image));

			_imageFactories.Add(factory);

			factory.SetValue(Image.SourceProperty, @"C:\Users\Plainer\Desktop\sigma.png");

			DataTemplate cellTemplate = new DataTemplate
			{
				VisualTree = factory
			};

			column.CellTemplate = cellTemplate;
			Content.Columns.Add(column);

			return dataBinding;
		}

		private List<object> AddSource(string identifier)
		{
			List<object> list = new List<object>();

			dataDictionary.Add(identifier, list);

			return list;
		}

	}
}