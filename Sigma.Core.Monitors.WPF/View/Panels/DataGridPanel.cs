using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public class DataGridPanel : SigmaPanel
	{
		public new DataGrid Content { get; }

		public DataGridPanel(string title) : base(title)
		{
			Content = new DataGrid
			{
				IsReadOnly = true
			};

			base.Content = Content;
		}

		public DataGridPanel(string title, object header, Type type, string propertyName) : this(title)
		{
			AddColumn(header, type, propertyName);
		}

		public DataGridPanel(string title, object header1, Type type1, string propertyName1, object header2, Type type2, string propertyName2) : this(title, header1, type1, propertyName1)
		{
			AddColumn(header2, type2, propertyName2);
		}

		public DataGridPanel(string title, object header1, Type type1, string propertyName1, object header2, Type type2, string propertyName2, object header3, Type type3, string propertyName3,
			params object[] columns) : this(title, header1, type1, propertyName1, header2, type2, propertyName2)
		{
			AddColumn(header3, type3, propertyName3);

			if (columns.Length % 3 != 0)
				throw new ArgumentException(nameof(columns));

			for (int i = 0; i < columns.Length; i++)
			{
				AddColumn(columns[i], (Type) columns[++i], (string) columns[++i]);
			}
		}

		//public List<object> AddDataGridTemplateColumn(object header, string headerIdentifier, Type type)
		//{
		//	if (type == typeof(Image))
		//	{
		//		return AddImageColumn(header, headerIdentifier);
		//	}
		//	else
		//	{
		//		return AddTextColumn(header);
		//	}
		//}


		//public List<object> AddTextColumn(object header)
		//{
		//	throw new NotImplementedException();
		//}

		//public List<object> AddImageColumn(string header, )
		//{
		//	return AddImageColumn(header, header);
		//}


		public bool AddColumn(object header, Type type, string propertyName)
		{
			if (type == typeof(string) || type == typeof(Label))
			{
				AddTextColumn(header, propertyName);
			}
			else if (type == typeof(Image))
			{
				AddImageColumn(header, propertyName);
			}
			else
			{
				return false;
			}

			return true;
		}

		private static void GenerateColumn(object header, Type type, out DataGridTemplateColumn column, out FrameworkElementFactory factory)
		{
			column = new DataGridTemplateColumn { Header = header };
			factory = new FrameworkElementFactory(type);
		}

		private static Binding GenerateBinding(string propertyName, BindingMode bindingMode)
		{
			Binding binding = new Binding(propertyName)
			{
				Mode = bindingMode,
			};

			return binding;
		}

		private void AddDataGridTemplateColumn(FrameworkElementFactory factory, DataGridTemplateColumn column)
		{
			DataTemplate cellTemplate = new DataTemplate
			{
				VisualTree = factory
			};

			column.CellTemplate = cellTemplate;
			Content.Columns.Add(column);
		}

		public void AddTextColumn(object header, string propertyName, BindingMode bindingMode = BindingMode.TwoWay)
		{
			DataGridTextColumn column = new DataGridTextColumn
			{
				Header = header,
				Binding = GenerateBinding(propertyName, bindingMode)
			};
			Content.Columns.Add(column);
		}

		public void AddImageColumn(object header, string propertyName, BindingMode bindingMode = BindingMode.TwoWay)
		{
			DataGridTemplateColumn column;
			FrameworkElementFactory factory;

			GenerateColumn(header, typeof(Image), out column, out factory);


			factory.SetValue(Image.SourceProperty, GenerateBinding(propertyName, bindingMode));

			AddDataGridTemplateColumn(factory, column);
		}


	}
}
