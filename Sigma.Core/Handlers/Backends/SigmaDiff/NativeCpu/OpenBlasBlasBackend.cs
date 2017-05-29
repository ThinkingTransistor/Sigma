/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Runtime.InteropServices;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu
{
    /// <summary>
    /// An OpenBLAS partial BLAS backend using external libopenblas native functions.
    /// </summary>
    [Serializable]
    public class OpenBlasBlasBackend : IBlasBackend
    {
        public unsafe int Isamax(int* n, float* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.isamax_(n, x, incx);
        }

        public unsafe void Saxpy(int* n, float* a, float* x, int* incx, float* y, int* incy)
        {
            NativeOpenBlasBlasMethods.saxpy_(n, a, x, incx, y, incy);
        }

        public unsafe void Sscal(int* n, float* alpha, float* x, int* incx)
        {
            NativeOpenBlasBlasMethods.sscal_(n, alpha, x, incx);
        }

        public unsafe void Sdot(int* n, float* x, int* incx, float* y, int* incy)
        {
            NativeOpenBlasBlasMethods.sdot_(n, x, incx, y, incy);
        }

        public unsafe float Sasum(int* n, float* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.sasum_(n, x, incx);
        }

        public unsafe float Snrm2(int* n, float* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.snrm2_(n, x, incx);
        }

        public unsafe int Idamax(int* n, double* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.idamax_(n, x, incx);
        }

        public unsafe void Daxpy(int* n, double* a, double* x, int* incx, double* y, int* incy)
        {
            NativeOpenBlasBlasMethods.daxpy_(n, a, x, incx, y, incy);
        }

        public unsafe void Dscal(int* n, double* alpha, double* x, int* incx)
        {
            NativeOpenBlasBlasMethods.dscal_(n, alpha, x, incx);
        }

        public unsafe void Ddot(int* n, double* x, int* incx, double* y, int* incy)
        {
            NativeOpenBlasBlasMethods.ddot_(n, x, incx, y, incy);
        }

        public unsafe double Dasum(int* n, double* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.dasum_(n, x, incx);
        }

        public unsafe double Dnrm2(int* n, double* x, int* incx)
        {
            return NativeOpenBlasBlasMethods.dnrm2_(n, x, incx);
        }

        public unsafe void Sger(int* m, int* n, float* alpha, float* x, int* incx, float* y, int* incy, float* a, int* lda)
        {
            NativeOpenBlasBlasMethods.sger_(m, n, alpha, x, incx, y, incy, a, lda);
        }

        public unsafe void Dger(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy, double* a, int* lda)
        {
            NativeOpenBlasBlasMethods.dger_(m, n, alpha, x, incx, y, incy, a, lda);
        }

        public unsafe void Sgemm(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda,
            float* b,
            int* ldb, float* beta, float* c, int* ldc)
        {
            NativeOpenBlasBlasMethods.sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public unsafe void Sgemv(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx,
            float* beta,
            float* y, int* incy)
        {
            NativeOpenBlasBlasMethods.sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
        }

        public unsafe void Dgemm(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda,
            double* b,
            int* ldb, double* beta, double* c, int* ldc)
        {
            NativeOpenBlasBlasMethods.dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public unsafe void Dgemv(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx,
            double* beta,
            double* y, int* incy)
        {
            NativeOpenBlasBlasMethods.dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
        }

        public unsafe void Somatcopy(int ordering, int trans, int rows, int cols, float alpha, float* a, int lda, float* b, int ldb)
        {
            NativeOpenBlasBlasMethods.somatcopy_(ordering, trans, rows, cols, alpha, a, lda, b, ldb);
        }

        public unsafe void Domatcopy(int ordering, int trans, int rows, int cols, float alpha, float* a, int lda, float* b, int ldb)
        {
            NativeOpenBlasBlasMethods.domatcopy_(ordering, trans, rows, cols, alpha, a, lda, b, ldb);
        }

        /// <summary>
        /// The external native OpenBLAS BLAS methods.
        /// </summary>
        internal static class NativeOpenBlasBlasMethods
        {
            static NativeOpenBlasBlasMethods()
            {
                PlatformDependentUtils.CheckPlatformDependentLibraries();
            }

            [DllImport(dllName: "libopenblas", EntryPoint = "isamax_")]
            internal static extern unsafe int isamax_(int* n, float* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "saxpy_")]
            internal static extern unsafe void saxpy_(int* n, float* a, float* x, int* incx, float* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "sscal_")]
            internal static extern unsafe void sscal_(int* n, float* alpha, float* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "sdot_")]
            internal static extern unsafe void sdot_(int* n, float* x, int* incx, float* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "sasum_")]
            internal static extern unsafe float sasum_(int* n, float* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "snrm2_")]
            internal static extern unsafe float snrm2_(int* n, float* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "idamax_")]
            internal static extern unsafe int idamax_(int* n, double* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "daxpy_")]
            internal static extern unsafe void daxpy_(int* n, double* a, double* x, int* incx, double* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "dscal_")]
            internal static extern unsafe void dscal_(int* n, double* alpha, double* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "ddot_")]
            internal static extern unsafe void ddot_(int* n, double* x, int* incx, double* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "dasum_")]
            internal static extern unsafe double dasum_(int* n, double* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "dnrm2_")]
            internal static extern unsafe double dnrm2_(int* n, double* x, int* incx);

            [DllImport(dllName: "libopenblas", EntryPoint = "sger_")]
            internal static extern unsafe void sger_(int* m, int* n, float* alpha, float* x, int* incx, float* y, int* incy,
                float* a, int* lda);

            [DllImport(dllName: "libopenblas", EntryPoint = "dger_")]
            internal static extern unsafe void dger_(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy,
                double* a, int* lda);

            [DllImport(dllName: "libopenblas", EntryPoint = "sgemm_")]
            internal static extern unsafe void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a,
                int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);

            [DllImport(dllName: "libopenblas", EntryPoint = "sgemv_")]
            internal static extern unsafe void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x,
                int* incx, float* beta, float* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "dgemm_")]
            internal static extern unsafe void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a,
                int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);

            [DllImport(dllName: "libopenblas", EntryPoint = "dgemv_")]
            internal static extern unsafe void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x,
                int* incx, double* beta, double* y, int* incy);

            [DllImport(dllName: "libopenblas", EntryPoint = "cblas_somatcopy")]
            internal static extern unsafe void somatcopy_(int ordering, int trans, int rows, int cols, float alpha, float* a, int lda, float* b, int ldb);

            [DllImport(dllName: "libopenblas", EntryPoint = "cblas_domatcopy")]
            internal static extern unsafe void domatcopy_(int ordering, int trans, int rows, int cols, float alpha, float* a, int lda, float* b, int ldb);
        }
    }
}