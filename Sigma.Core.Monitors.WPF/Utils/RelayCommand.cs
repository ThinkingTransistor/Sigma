using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public class RelayCommand : ICommand
	{
		readonly Action<object> _execute;
		readonly Func<bool> _canExecute;

		public RelayCommand(Action execute) : this(execute, null) { }
		public RelayCommand(Action<object> execute) : this(execute, null) { }
		public RelayCommand(Action execute, Func<bool> canExecute)
		{
			if (execute == null)
				throw new ArgumentNullException(nameof(execute));

			_execute = p => execute();
			_canExecute = canExecute;
		}

		public RelayCommand(Action<object> execute, Func<bool> canExecute)
		{
			if (execute == null) throw new ArgumentNullException(nameof(execute));

			_execute = execute;
			_canExecute = canExecute;
		}

		public bool CanExecute(object parameter)
		{
			return _canExecute?.Invoke() ?? true;
		}

		public event EventHandler CanExecuteChanged
		{
			add { CommandManager.RequerySuggested += value; }
			remove { CommandManager.RequerySuggested -= value; }
		}

		public void Execute(object parameter)
		{
			_execute(parameter);
		}
	}
}
