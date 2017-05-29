using Sigma.Core.MathAbstract;

namespace Sigma.Core.Monitors.WPF.Panels
{
	public interface IInputPanel 
	{
		bool IsReady { get; }
		INDArray Values { get; }
	}
}