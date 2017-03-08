/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Training.Operators.Backends.NativeCpu;

namespace Sigma.Core.Persistence.Selectors.Operator
{
    /// <summary>
    /// An operator selector for <see cref="CpuSinglethreadedOperator"/>s.
    /// </summary>
    public class CpuSinglethreadedOperatorSelector : BaseOperatorSelector<CpuSinglethreadedOperator>
    {
        /// <summary>
        /// Create a base operator selector with a certain operator.
        /// </summary>
        /// <param name="operator">The operator.</param>
        public CpuSinglethreadedOperatorSelector(CpuSinglethreadedOperator @operator) : base(@operator)
        {
        }

        /// <summary>
        /// Create an operator of this operator selectors appropriate type (take necessary constructor arguments from the current <see cref="Result"/> operator).
        /// </summary>
        /// <returns>The operator.</returns>
        protected override CpuSinglethreadedOperator CreateOperator()
        {
            return new CpuSinglethreadedOperator(Result.Handler, Result.WorkerPriority);
        }

        /// <summary>
        /// Create an operator selector with a certain operator.
        /// </summary>
        /// <param name="operator">The operator</param>
        /// <returns>An operator selector with the given operator.</returns>
        protected override IOperatorSelector<CpuSinglethreadedOperator> CreateSelector(CpuSinglethreadedOperator @operator)
        {
            return new CpuSinglethreadedOperatorSelector(@operator);
        }
    }
}
