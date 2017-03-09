using System;
using Sigma.Core.Monitors.WPF.View.Parameterisation;

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	public interface IParameterVisualisationManager
	{
		bool Add(ParameterVisualiserAttribute parameterInfo, Type visualiserClass);
		bool Remove(Type type);
	}
}