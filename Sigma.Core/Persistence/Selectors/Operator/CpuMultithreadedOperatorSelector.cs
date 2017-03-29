/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Training.Operators.Backends.NativeCpu;

namespace Sigma.Core.Persistence.Selectors.Operator
{
    /// <summary>
    /// An operator selector for <see cref="CpuMultithreadedOperator"/>s.
    /// </summary>
    public class CpuMultithreadedOperatorSelector : BaseOperatorSelector<CpuMultithreadedOperator>
    {
        /// <summary>
        /// Create a base operator selector with a certain operator.
        /// </summary>
        /// <param name="operator">The operator.</param>
        public CpuMultithreadedOperatorSelector(CpuMultithreadedOperator @operator) : base(@operator)
        {
        }

        /// <summary>
        /// Create an operator of this operator selectors appropriate type (take necessary constructor arguments from the current <see cref="Result"/> operator).
        /// </summary>
        /// <returns>The operator.</returns>
        protected override CpuMultithreadedOperator CreateOperator()
        {
            return new CpuMultithreadedOperator(Result.Handler, Result.WorkerCount, Result.WorkerPriority);
        }

        /// <summary>
        /// Create an operator selector with a certain operator.
        /// </summary>
        /// <param name="operator">The operator</param>
        /// <returns>An operator selector with the given operator.</returns>
        protected override IOperatorSelector<CpuMultithreadedOperator> CreateSelector(CpuMultithreadedOperator @operator)
        {
            return new CpuMultithreadedOperatorSelector(@operator);
        }
    }
}
