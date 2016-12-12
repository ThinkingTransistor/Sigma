/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Handlers.Backends
{
	/// <summary>
	/// A BLAS backend interface consisting of a subset of the BLAS standard functions.
	/// See http://www.netlib.org/blas/ for details.
	/// </summary>
	public unsafe interface IBlasBackend
	{
		#region Scalar-vector valued BLAS functions

		void Isamax(int* n, float* x, int* incx);
		void Saxpy(int* n, float* a, float* x, int* incx, float* y, int* incy);
		void Sscal(int* n, float* alpha, float* x, int* incx);
		void Sdot(int* n, float* x, int* incx, float* y, int* incy);
		float Sasum(int* n, float* x, int* incx);
		float Snrm2(int* n, float* x, int* incx);

		void Idamax(int* n, double* x, int* incx);
		void Daxpy(int* n, double* a, double* x, int* incx, double* y, int* incy);
		void Dscal(int* n, double* alpha, double* x, int* incx);
		void Ddot(int* n, double* x, int* incx, double* y, int* incy);
		double Dasum(int* n, double* x, int* incx);
		double Dnrm2(int* n, double* x, int* incx);

		#endregion

		#region Vector-matrix valued BLAS functions

		void Sger(int* m, int* n, float* alpha, float* x, int* incx, float* y, int* incy, float* a, int* lda);

		void Dger(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy, double* a, int* lda);

		#endregion

		#region Matrix valued BLAS functions

		void Sgemm(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
		void Sgemv(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);

		void Dgemm(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
		void Dgemv(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);

		#endregion
	}
}
