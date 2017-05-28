using Sigma.Core.MathAbstract;

namespace Sigma.Core.Monitors.WPF.Panels
{
	public interface IOutputPanel
	{
		IInputPanel Input { get; set; }
		//TODO: think about beatiful system
		void SetInputReference(INDArray values);
	}
}