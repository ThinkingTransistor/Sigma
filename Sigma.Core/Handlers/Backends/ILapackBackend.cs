/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Handlers.Backends
{
	/// <summary>
	/// A LAPACK backend interface consisting of a subset of the LAPACK standard functions.
	/// See http://www.netlib.org/lapack/ for details.
	/// </summary>
	public unsafe interface ILapackBackend
	{
		void Sgesv(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
		void Ssysv_(char* uplo, int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, float* work, int* lwork,
			int* info);
		void Sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);
		void Sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);

		void Dgesv(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
		void Dsysv_(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork,
			int* info);
		void Dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
		void Dgetri_(int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
	}
}
