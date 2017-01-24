#line 1 "C:/Users/Plainer/Documents/GitHub/Sigma/Sigma.Kernels/sigmakernels.cu"
#pragma section("__nv_managed_data__")
#line 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

#line 1
#line 62 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\sal.h"
extern "C" {
#line 2967 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\sal.h"
}
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\ConcurrencySal.h"
extern "C" {
#line 354 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\ConcurrencySal.h"
}
#line 15 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 18
extern "C" {
#line 28 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
typedef char *va_list; 
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
}
#line 122 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
extern "C++" {
#line 124
template< class _Ty> 
#line 125
struct __vcrt_va_list_is_reference { 
#line 127
enum: bool { __the_value}; 
#line 128
}; 
#line 130
template< class _Ty> 
#line 131
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 133
enum: bool { __the_value = '\001'}; 
#line 134
}; 
#line 136
template< class _Ty> 
#line 137
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 139
enum: bool { __the_value = '\001'}; 
#line 140
}; 
#line 142
template < typename _Ty >
        void __vcrt_va_start_verify_argument_type ( ) throw ( )
        {
            static_assert ( ! __vcrt_va_list_is_reference < _Ty > :: __the_value, "va_start argument must not have reference type and must not be parenthesized" );
        }
#line 147
}
#line 157 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vadefs.h"
#pragma pack ( pop )
#line 81 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 194 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#include "crt/host_runtime.h"
#line 195
typedef __int64 ptrdiff_t; 
#line 196
typedef __int64 intptr_t; 
#line 204 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 247 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
extern "C++" {
#line 249
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 253
}
#line 298 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 304
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 305
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 309 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 317 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 12 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 119 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
extern "C++" {
#line 121
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 124
template< class _Ty> 
#line 125
struct _CrtEnableIf< true, _Ty>  { 
#line 127
typedef _Ty _Type; 
#line 128
}; 
#line 129
}
#line 133 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 217 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
__declspec(dllimport) void __cdecl _invalid_parameter(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 226 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
__declspec(dllimport) void __cdecl _invalid_parameter_noinfo(); 
#line 227
__declspec(dllimport) __declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 229
__declspec(noreturn) 
#line 230
__declspec(dllimport) void __cdecl _invoke_watson(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 448 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 449
typedef unsigned short wint_t; 
#line 450
typedef unsigned short wctype_t; 
#line 451
typedef long __time32_t; 
#line 452
typedef __int64 __time64_t; 
#line 459
typedef 
#line 454
struct __crt_locale_data_public { 
#line 456
const unsigned short *_locale_pctype; 
#line 457
int _locale_mb_cur_max; 
#line 458
unsigned _locale_lc_codepage; 
#line 459
} __crt_locale_data_public; 
#line 465
typedef 
#line 461
struct __crt_locale_pointers { 
#line 463
struct __crt_locale_data *locinfo; 
#line 464
struct __crt_multibyte_data *mbcinfo; 
#line 465
} __crt_locale_pointers; 
#line 467
typedef __crt_locale_pointers *_locale_t; 
#line 473
typedef 
#line 469
struct _Mbstatet { 
#line 471
unsigned long _Wchar; 
#line 472
unsigned short _Byte, _State; 
#line 473
} _Mbstatet; 
#line 475
typedef _Mbstatet mbstate_t; 
#line 489 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 499 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 1901 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"
#if 0
#line 61
enum cudaRoundMode { 
#line 63
cudaRoundNearest, 
#line 64
cudaRoundZero, 
#line 65
cudaRoundPosInf, 
#line 66
cudaRoundMinInf
#line 67
}; 
#endif
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\limits.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stddef.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 18
namespace std { 
#line 20
typedef decltype((__nullptr)) nullptr_t; 
#line 21
}
#line 23
using std::nullptr_t;
#line 29 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stddef.h"
__declspec(dllimport) int *__cdecl _errno(); 
#line 32
__declspec(dllimport) errno_t __cdecl _set_errno(int _Value); 
#line 33
__declspec(dllimport) errno_t __cdecl _get_errno(int * _Value); 
#line 46 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stddef.h"
__declspec(dllimport) extern unsigned long __cdecl __threadid(); 
#line 48
__declspec(dllimport) extern uintptr_t __cdecl __threadhandle(); 
#line 52
}__pragma( pack ( pop )) 
#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 156
enum cudaError { 
#line 163
cudaSuccess, 
#line 169
cudaErrorMissingConfiguration, 
#line 175
cudaErrorMemoryAllocation, 
#line 181
cudaErrorInitializationError, 
#line 191
cudaErrorLaunchFailure, 
#line 200
cudaErrorPriorLaunchFailure, 
#line 210
cudaErrorLaunchTimeout, 
#line 219
cudaErrorLaunchOutOfResources, 
#line 225
cudaErrorInvalidDeviceFunction, 
#line 234
cudaErrorInvalidConfiguration, 
#line 240
cudaErrorInvalidDevice, 
#line 246
cudaErrorInvalidValue, 
#line 252
cudaErrorInvalidPitchValue, 
#line 258
cudaErrorInvalidSymbol, 
#line 263
cudaErrorMapBufferObjectFailed, 
#line 268
cudaErrorUnmapBufferObjectFailed, 
#line 274
cudaErrorInvalidHostPointer, 
#line 280
cudaErrorInvalidDevicePointer, 
#line 286
cudaErrorInvalidTexture, 
#line 292
cudaErrorInvalidTextureBinding, 
#line 299
cudaErrorInvalidChannelDescriptor, 
#line 305
cudaErrorInvalidMemcpyDirection, 
#line 315
cudaErrorAddressOfConstant, 
#line 324
cudaErrorTextureFetchFailed, 
#line 333
cudaErrorTextureNotBound, 
#line 342
cudaErrorSynchronizationError, 
#line 348
cudaErrorInvalidFilterSetting, 
#line 354
cudaErrorInvalidNormSetting, 
#line 362
cudaErrorMixedDeviceExecution, 
#line 369
cudaErrorCudartUnloading, 
#line 374
cudaErrorUnknown, 
#line 382
cudaErrorNotYetImplemented, 
#line 391
cudaErrorMemoryValueTooLarge, 
#line 398
cudaErrorInvalidResourceHandle, 
#line 406
cudaErrorNotReady, 
#line 413
cudaErrorInsufficientDriver, 
#line 426
cudaErrorSetOnActiveProcess, 
#line 432
cudaErrorInvalidSurface, 
#line 438
cudaErrorNoDevice, 
#line 444
cudaErrorECCUncorrectable, 
#line 449
cudaErrorSharedObjectSymbolNotFound, 
#line 454
cudaErrorSharedObjectInitFailed, 
#line 460
cudaErrorUnsupportedLimit, 
#line 466
cudaErrorDuplicateVariableName, 
#line 472
cudaErrorDuplicateTextureName, 
#line 478
cudaErrorDuplicateSurfaceName, 
#line 488
cudaErrorDevicesUnavailable, 
#line 493
cudaErrorInvalidKernelImage, 
#line 501
cudaErrorNoKernelImageForDevice, 
#line 514
cudaErrorIncompatibleDriverContext, 
#line 521
cudaErrorPeerAccessAlreadyEnabled, 
#line 528
cudaErrorPeerAccessNotEnabled, 
#line 534
cudaErrorDeviceAlreadyInUse = 54, 
#line 541
cudaErrorProfilerDisabled, 
#line 549
cudaErrorProfilerNotInitialized, 
#line 556
cudaErrorProfilerAlreadyStarted, 
#line 563
cudaErrorProfilerAlreadyStopped, 
#line 571
cudaErrorAssert, 
#line 578
cudaErrorTooManyPeers, 
#line 584
cudaErrorHostMemoryAlreadyRegistered, 
#line 590
cudaErrorHostMemoryNotRegistered, 
#line 595
cudaErrorOperatingSystem, 
#line 601
cudaErrorPeerAccessUnsupported, 
#line 608
cudaErrorLaunchMaxDepthExceeded, 
#line 616
cudaErrorLaunchFileScopedTex, 
#line 624
cudaErrorLaunchFileScopedSurf, 
#line 639
cudaErrorSyncDepthExceeded, 
#line 651
cudaErrorLaunchPendingCountExceeded, 
#line 656
cudaErrorNotPermitted, 
#line 662
cudaErrorNotSupported, 
#line 671
cudaErrorHardwareStackError, 
#line 679
cudaErrorIllegalInstruction, 
#line 688
cudaErrorMisalignedAddress, 
#line 699
cudaErrorInvalidAddressSpace, 
#line 707
cudaErrorInvalidPc, 
#line 715
cudaErrorIllegalAddress, 
#line 721
cudaErrorInvalidPtx, 
#line 726
cudaErrorInvalidGraphicsContext, 
#line 732
cudaErrorNvlinkUncorrectable, 
#line 737
cudaErrorStartupFailure = 127, 
#line 745
cudaErrorApiFailureBase = 10000
#line 746
}; 
#endif
#line 751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 751
enum cudaChannelFormatKind { 
#line 753
cudaChannelFormatKindSigned, 
#line 754
cudaChannelFormatKindUnsigned, 
#line 755
cudaChannelFormatKindFloat, 
#line 756
cudaChannelFormatKindNone
#line 757
}; 
#endif
#line 762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 762
struct cudaChannelFormatDesc { 
#line 764
int x; 
#line 765
int y; 
#line 766
int z; 
#line 767
int w; 
#line 768
cudaChannelFormatKind f; 
#line 769
}; 
#endif
#line 774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 779
typedef const cudaArray *cudaArray_const_t; 
#line 781
struct cudaArray; 
#line 786
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 791
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 793
struct cudaMipmappedArray; 
#line 798
#if 0
#line 798
enum cudaMemoryType { 
#line 800
cudaMemoryTypeHost = 1, 
#line 801
cudaMemoryTypeDevice
#line 802
}; 
#endif
#line 807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 807
enum cudaMemcpyKind { 
#line 809
cudaMemcpyHostToHost, 
#line 810
cudaMemcpyHostToDevice, 
#line 811
cudaMemcpyDeviceToHost, 
#line 812
cudaMemcpyDeviceToDevice, 
#line 813
cudaMemcpyDefault
#line 814
}; 
#endif
#line 821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 821
struct cudaPitchedPtr { 
#line 823
void *ptr; 
#line 824
size_t pitch; 
#line 825
size_t xsize; 
#line 826
size_t ysize; 
#line 827
}; 
#endif
#line 834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 834
struct cudaExtent { 
#line 836
size_t width; 
#line 837
size_t height; 
#line 838
size_t depth; 
#line 839
}; 
#endif
#line 846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 846
struct cudaPos { 
#line 848
size_t x; 
#line 849
size_t y; 
#line 850
size_t z; 
#line 851
}; 
#endif
#line 856 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 856
struct cudaMemcpy3DParms { 
#line 858
cudaArray_t srcArray; 
#line 859
cudaPos srcPos; 
#line 860
cudaPitchedPtr srcPtr; 
#line 862
cudaArray_t dstArray; 
#line 863
cudaPos dstPos; 
#line 864
cudaPitchedPtr dstPtr; 
#line 866
cudaExtent extent; 
#line 867
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 868
}; 
#endif
#line 873 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 873
struct cudaMemcpy3DPeerParms { 
#line 875
cudaArray_t srcArray; 
#line 876
cudaPos srcPos; 
#line 877
cudaPitchedPtr srcPtr; 
#line 878
int srcDevice; 
#line 880
cudaArray_t dstArray; 
#line 881
cudaPos dstPos; 
#line 882
cudaPitchedPtr dstPtr; 
#line 883
int dstDevice; 
#line 885
cudaExtent extent; 
#line 886
}; 
#endif
#line 891 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 896
#if 0
#line 896
enum cudaGraphicsRegisterFlags { 
#line 898
cudaGraphicsRegisterFlagsNone, 
#line 899
cudaGraphicsRegisterFlagsReadOnly, 
#line 900
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 901
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 902
cudaGraphicsRegisterFlagsTextureGather = 8
#line 903
}; 
#endif
#line 908 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 908
enum cudaGraphicsMapFlags { 
#line 910
cudaGraphicsMapFlagsNone, 
#line 911
cudaGraphicsMapFlagsReadOnly, 
#line 912
cudaGraphicsMapFlagsWriteDiscard
#line 913
}; 
#endif
#line 918 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 918
enum cudaGraphicsCubeFace { 
#line 920
cudaGraphicsCubeFacePositiveX, 
#line 921
cudaGraphicsCubeFaceNegativeX, 
#line 922
cudaGraphicsCubeFacePositiveY, 
#line 923
cudaGraphicsCubeFaceNegativeY, 
#line 924
cudaGraphicsCubeFacePositiveZ, 
#line 925
cudaGraphicsCubeFaceNegativeZ
#line 926
}; 
#endif
#line 931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 931
enum cudaResourceType { 
#line 933
cudaResourceTypeArray, 
#line 934
cudaResourceTypeMipmappedArray, 
#line 935
cudaResourceTypeLinear, 
#line 936
cudaResourceTypePitch2D
#line 937
}; 
#endif
#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 942
enum cudaResourceViewFormat { 
#line 944
cudaResViewFormatNone, 
#line 945
cudaResViewFormatUnsignedChar1, 
#line 946
cudaResViewFormatUnsignedChar2, 
#line 947
cudaResViewFormatUnsignedChar4, 
#line 948
cudaResViewFormatSignedChar1, 
#line 949
cudaResViewFormatSignedChar2, 
#line 950
cudaResViewFormatSignedChar4, 
#line 951
cudaResViewFormatUnsignedShort1, 
#line 952
cudaResViewFormatUnsignedShort2, 
#line 953
cudaResViewFormatUnsignedShort4, 
#line 954
cudaResViewFormatSignedShort1, 
#line 955
cudaResViewFormatSignedShort2, 
#line 956
cudaResViewFormatSignedShort4, 
#line 957
cudaResViewFormatUnsignedInt1, 
#line 958
cudaResViewFormatUnsignedInt2, 
#line 959
cudaResViewFormatUnsignedInt4, 
#line 960
cudaResViewFormatSignedInt1, 
#line 961
cudaResViewFormatSignedInt2, 
#line 962
cudaResViewFormatSignedInt4, 
#line 963
cudaResViewFormatHalf1, 
#line 964
cudaResViewFormatHalf2, 
#line 965
cudaResViewFormatHalf4, 
#line 966
cudaResViewFormatFloat1, 
#line 967
cudaResViewFormatFloat2, 
#line 968
cudaResViewFormatFloat4, 
#line 969
cudaResViewFormatUnsignedBlockCompressed1, 
#line 970
cudaResViewFormatUnsignedBlockCompressed2, 
#line 971
cudaResViewFormatUnsignedBlockCompressed3, 
#line 972
cudaResViewFormatUnsignedBlockCompressed4, 
#line 973
cudaResViewFormatSignedBlockCompressed4, 
#line 974
cudaResViewFormatUnsignedBlockCompressed5, 
#line 975
cudaResViewFormatSignedBlockCompressed5, 
#line 976
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 977
cudaResViewFormatSignedBlockCompressed6H, 
#line 978
cudaResViewFormatUnsignedBlockCompressed7
#line 979
}; 
#endif
#line 984 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 984
struct cudaResourceDesc { 
#line 985
cudaResourceType resType; 
#line 987
union { 
#line 988
struct { 
#line 989
cudaArray_t array; 
#line 990
} array; 
#line 991
struct { 
#line 992
cudaMipmappedArray_t mipmap; 
#line 993
} mipmap; 
#line 994
struct { 
#line 995
void *devPtr; 
#line 996
cudaChannelFormatDesc desc; 
#line 997
size_t sizeInBytes; 
#line 998
} linear; 
#line 999
struct { 
#line 1000
void *devPtr; 
#line 1001
cudaChannelFormatDesc desc; 
#line 1002
size_t width; 
#line 1003
size_t height; 
#line 1004
size_t pitchInBytes; 
#line 1005
} pitch2D; 
#line 1006
} res; 
#line 1007
}; 
#endif
#line 1012 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1012
struct cudaResourceViewDesc { 
#line 1014
cudaResourceViewFormat format; 
#line 1015
size_t width; 
#line 1016
size_t height; 
#line 1017
size_t depth; 
#line 1018
unsigned firstMipmapLevel; 
#line 1019
unsigned lastMipmapLevel; 
#line 1020
unsigned firstLayer; 
#line 1021
unsigned lastLayer; 
#line 1022
}; 
#endif
#line 1027 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1027
struct cudaPointerAttributes { 
#line 1033
cudaMemoryType memoryType; 
#line 1044
int device; 
#line 1050
void *devicePointer; 
#line 1056
void *hostPointer; 
#line 1061
int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1062
}; 
#endif
#line 1067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1067
struct cudaFuncAttributes { 
#line 1074
size_t sharedSizeBytes; 
#line 1080
size_t constSizeBytes; 
#line 1085
size_t localSizeBytes; 
#line 1092
int maxThreadsPerBlock; 
#line 1097
int numRegs; 
#line 1104
int ptxVersion; 
#line 1111
int binaryVersion; 
#line 1117
int cacheModeCA; 
#line 1118
}; 
#endif
#line 1123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1123
enum cudaFuncCache { 
#line 1125
cudaFuncCachePreferNone, 
#line 1126
cudaFuncCachePreferShared, 
#line 1127
cudaFuncCachePreferL1, 
#line 1128
cudaFuncCachePreferEqual
#line 1129
}; 
#endif
#line 1135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1135
enum cudaSharedMemConfig { 
#line 1137
cudaSharedMemBankSizeDefault, 
#line 1138
cudaSharedMemBankSizeFourByte, 
#line 1139
cudaSharedMemBankSizeEightByte
#line 1140
}; 
#endif
#line 1145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1145
enum cudaComputeMode { 
#line 1147
cudaComputeModeDefault, 
#line 1148
cudaComputeModeExclusive, 
#line 1149
cudaComputeModeProhibited, 
#line 1150
cudaComputeModeExclusiveProcess
#line 1151
}; 
#endif
#line 1156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1156
enum cudaLimit { 
#line 1158
cudaLimitStackSize, 
#line 1159
cudaLimitPrintfFifoSize, 
#line 1160
cudaLimitMallocHeapSize, 
#line 1161
cudaLimitDevRuntimeSyncDepth, 
#line 1162
cudaLimitDevRuntimePendingLaunchCount
#line 1163
}; 
#endif
#line 1168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1168
enum cudaMemoryAdvise { 
#line 1170
cudaMemAdviseSetReadMostly = 1, 
#line 1171
cudaMemAdviseUnsetReadMostly, 
#line 1172
cudaMemAdviseSetPreferredLocation, 
#line 1173
cudaMemAdviseUnsetPreferredLocation, 
#line 1174
cudaMemAdviseSetAccessedBy, 
#line 1175
cudaMemAdviseUnsetAccessedBy
#line 1176
}; 
#endif
#line 1181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1181
enum cudaMemRangeAttribute { 
#line 1183
cudaMemRangeAttributeReadMostly = 1, 
#line 1184
cudaMemRangeAttributePreferredLocation, 
#line 1185
cudaMemRangeAttributeAccessedBy, 
#line 1186
cudaMemRangeAttributeLastPrefetchLocation
#line 1187
}; 
#endif
#line 1192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1192
enum cudaOutputMode { 
#line 1194
cudaKeyValuePair, 
#line 1195
cudaCSV
#line 1196
}; 
#endif
#line 1201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1201
enum cudaDeviceAttr { 
#line 1203
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1204
cudaDevAttrMaxBlockDimX, 
#line 1205
cudaDevAttrMaxBlockDimY, 
#line 1206
cudaDevAttrMaxBlockDimZ, 
#line 1207
cudaDevAttrMaxGridDimX, 
#line 1208
cudaDevAttrMaxGridDimY, 
#line 1209
cudaDevAttrMaxGridDimZ, 
#line 1210
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1211
cudaDevAttrTotalConstantMemory, 
#line 1212
cudaDevAttrWarpSize, 
#line 1213
cudaDevAttrMaxPitch, 
#line 1214
cudaDevAttrMaxRegistersPerBlock, 
#line 1215
cudaDevAttrClockRate, 
#line 1216
cudaDevAttrTextureAlignment, 
#line 1217
cudaDevAttrGpuOverlap, 
#line 1218
cudaDevAttrMultiProcessorCount, 
#line 1219
cudaDevAttrKernelExecTimeout, 
#line 1220
cudaDevAttrIntegrated, 
#line 1221
cudaDevAttrCanMapHostMemory, 
#line 1222
cudaDevAttrComputeMode, 
#line 1223
cudaDevAttrMaxTexture1DWidth, 
#line 1224
cudaDevAttrMaxTexture2DWidth, 
#line 1225
cudaDevAttrMaxTexture2DHeight, 
#line 1226
cudaDevAttrMaxTexture3DWidth, 
#line 1227
cudaDevAttrMaxTexture3DHeight, 
#line 1228
cudaDevAttrMaxTexture3DDepth, 
#line 1229
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1230
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1231
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1232
cudaDevAttrSurfaceAlignment, 
#line 1233
cudaDevAttrConcurrentKernels, 
#line 1234
cudaDevAttrEccEnabled, 
#line 1235
cudaDevAttrPciBusId, 
#line 1236
cudaDevAttrPciDeviceId, 
#line 1237
cudaDevAttrTccDriver, 
#line 1238
cudaDevAttrMemoryClockRate, 
#line 1239
cudaDevAttrGlobalMemoryBusWidth, 
#line 1240
cudaDevAttrL2CacheSize, 
#line 1241
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1242
cudaDevAttrAsyncEngineCount, 
#line 1243
cudaDevAttrUnifiedAddressing, 
#line 1244
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1245
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1246
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1247
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1248
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1249
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1250
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1251
cudaDevAttrPciDomainId, 
#line 1252
cudaDevAttrTexturePitchAlignment, 
#line 1253
cudaDevAttrMaxTextureCubemapWidth, 
#line 1254
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1255
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1256
cudaDevAttrMaxSurface1DWidth, 
#line 1257
cudaDevAttrMaxSurface2DWidth, 
#line 1258
cudaDevAttrMaxSurface2DHeight, 
#line 1259
cudaDevAttrMaxSurface3DWidth, 
#line 1260
cudaDevAttrMaxSurface3DHeight, 
#line 1261
cudaDevAttrMaxSurface3DDepth, 
#line 1262
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1263
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1264
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1265
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1266
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1267
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1268
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1269
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1270
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1271
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1272
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1273
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1274
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1275
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1276
cudaDevAttrComputeCapabilityMajor, 
#line 1277
cudaDevAttrComputeCapabilityMinor, 
#line 1278
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1279
cudaDevAttrStreamPrioritiesSupported, 
#line 1280
cudaDevAttrGlobalL1CacheSupported, 
#line 1281
cudaDevAttrLocalL1CacheSupported, 
#line 1282
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 1283
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 1284
cudaDevAttrManagedMemory, 
#line 1285
cudaDevAttrIsMultiGpuBoard, 
#line 1286
cudaDevAttrMultiGpuBoardGroupID, 
#line 1287
cudaDevAttrHostNativeAtomicSupported, 
#line 1288
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 1289
cudaDevAttrPageableMemoryAccess, 
#line 1290
cudaDevAttrConcurrentManagedAccess, 
#line 1291
cudaDevAttrComputePreemptionSupported, 
#line 1292
cudaDevAttrCanUseHostPointerForRegisteredMem
#line 1293
}; 
#endif
#line 1299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1299
enum cudaDeviceP2PAttr { 
#line 1300
cudaDevP2PAttrPerformanceRank = 1, 
#line 1301
cudaDevP2PAttrAccessSupported, 
#line 1302
cudaDevP2PAttrNativeAtomicSupported
#line 1303
}; 
#endif
#line 1307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
#line 1307
struct cudaDeviceProp { 
#line 1309
char name[256]; 
#line 1310
size_t totalGlobalMem; 
#line 1311
size_t sharedMemPerBlock; 
#line 1312
int regsPerBlock; 
#line 1313
int warpSize; 
#line 1314
size_t memPitch; 
#line 1315
int maxThreadsPerBlock; 
#line 1316
int maxThreadsDim[3]; 
#line 1317
int maxGridSize[3]; 
#line 1318
int clockRate; 
#line 1319
size_t totalConstMem; 
#line 1320
int major; 
#line 1321
int minor; 
#line 1322
size_t textureAlignment; 
#line 1323
size_t texturePitchAlignment; 
#line 1324
int deviceOverlap; 
#line 1325
int multiProcessorCount; 
#line 1326
int kernelExecTimeoutEnabled; 
#line 1327
int integrated; 
#line 1328
int canMapHostMemory; 
#line 1329
int computeMode; 
#line 1330
int maxTexture1D; 
#line 1331
int maxTexture1DMipmap; 
#line 1332
int maxTexture1DLinear; 
#line 1333
int maxTexture2D[2]; 
#line 1334
int maxTexture2DMipmap[2]; 
#line 1335
int maxTexture2DLinear[3]; 
#line 1336
int maxTexture2DGather[2]; 
#line 1337
int maxTexture3D[3]; 
#line 1338
int maxTexture3DAlt[3]; 
#line 1339
int maxTextureCubemap; 
#line 1340
int maxTexture1DLayered[2]; 
#line 1341
int maxTexture2DLayered[3]; 
#line 1342
int maxTextureCubemapLayered[2]; 
#line 1343
int maxSurface1D; 
#line 1344
int maxSurface2D[2]; 
#line 1345
int maxSurface3D[3]; 
#line 1346
int maxSurface1DLayered[2]; 
#line 1347
int maxSurface2DLayered[3]; 
#line 1348
int maxSurfaceCubemap; 
#line 1349
int maxSurfaceCubemapLayered[2]; 
#line 1350
size_t surfaceAlignment; 
#line 1351
int concurrentKernels; 
#line 1352
int ECCEnabled; 
#line 1353
int pciBusID; 
#line 1354
int pciDeviceID; 
#line 1355
int pciDomainID; 
#line 1356
int tccDriver; 
#line 1357
int asyncEngineCount; 
#line 1358
int unifiedAddressing; 
#line 1359
int memoryClockRate; 
#line 1360
int memoryBusWidth; 
#line 1361
int l2CacheSize; 
#line 1362
int maxThreadsPerMultiProcessor; 
#line 1363
int streamPrioritiesSupported; 
#line 1364
int globalL1CacheSupported; 
#line 1365
int localL1CacheSupported; 
#line 1366
size_t sharedMemPerMultiprocessor; 
#line 1367
int regsPerMultiprocessor; 
#line 1368
int managedMemory; 
#line 1369
int isMultiGpuBoard; 
#line 1370
int multiGpuBoardGroupID; 
#line 1371
int hostNativeAtomicSupported; 
#line 1372
int singleToDoublePrecisionPerfRatio; 
#line 1373
int pageableMemoryAccess; 
#line 1374
int concurrentManagedAccess; 
#line 1375
}; 
#endif
#line 1458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef 
#line 1455
struct cudaIpcEventHandle_st { 
#line 1457
char reserved[64]; 
#line 1458
} cudaIpcEventHandle_t; 
#endif
#line 1466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef 
#line 1463
struct cudaIpcMemHandle_st { 
#line 1465
char reserved[64]; 
#line 1466
} cudaIpcMemHandle_t; 
#endif
#line 1477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 1477
cudaError_t; 
#endif
#line 1482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 1482
cudaStream_t; 
#endif
#line 1487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 1487
cudaEvent_t; 
#endif
#line 1492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 1492
cudaGraphicsResource_t; 
#endif
#line 1497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef struct CUuuid_st 
#line 1497
cudaUUID_t; 
#endif
#line 1502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"
#if 0
typedef cudaOutputMode 
#line 1502
cudaOutputMode_t; 
#endif
#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#if 0
#line 84
enum cudaSurfaceBoundaryMode { 
#line 86
cudaBoundaryModeZero, 
#line 87
cudaBoundaryModeClamp, 
#line 88
cudaBoundaryModeTrap
#line 89
}; 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#if 0
#line 94
enum cudaSurfaceFormatMode { 
#line 96
cudaFormatModeForced, 
#line 97
cudaFormatModeAuto
#line 98
}; 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#if 0
#line 103
struct surfaceReference { 
#line 108
cudaChannelFormatDesc channelDesc; 
#line 109
}; 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 114
cudaSurfaceObject_t; 
#endif
#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
#line 84
enum cudaTextureAddressMode { 
#line 86
cudaAddressModeWrap, 
#line 87
cudaAddressModeClamp, 
#line 88
cudaAddressModeMirror, 
#line 89
cudaAddressModeBorder
#line 90
}; 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
#line 95
enum cudaTextureFilterMode { 
#line 97
cudaFilterModePoint, 
#line 98
cudaFilterModeLinear
#line 99
}; 
#endif
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
#line 104
enum cudaTextureReadMode { 
#line 106
cudaReadModeElementType, 
#line 107
cudaReadModeNormalizedFloat
#line 108
}; 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
#line 113
struct textureReference { 
#line 118
int normalized; 
#line 122
cudaTextureFilterMode filterMode; 
#line 126
cudaTextureAddressMode addressMode[3]; 
#line 130
cudaChannelFormatDesc channelDesc; 
#line 134
int sRGB; 
#line 138
unsigned maxAnisotropy; 
#line 142
cudaTextureFilterMode mipmapFilterMode; 
#line 146
float mipmapLevelBias; 
#line 150
float minMipmapLevelClamp; 
#line 154
float maxMipmapLevelClamp; 
#line 155
int __cudaReserved[15]; 
#line 156
}; 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
#line 161
struct cudaTextureDesc { 
#line 166
cudaTextureAddressMode addressMode[3]; 
#line 170
cudaTextureFilterMode filterMode; 
#line 174
cudaTextureReadMode readMode; 
#line 178
int sRGB; 
#line 182
float borderColor[4]; 
#line 186
int normalizedCoords; 
#line 190
unsigned maxAnisotropy; 
#line 194
cudaTextureFilterMode mipmapFilterMode; 
#line 198
float mipmapLevelBias; 
#line 202
float minMipmapLevelClamp; 
#line 206
float maxMipmapLevelClamp; 
#line 207
}; 
#endif
#line 212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 212
cudaTextureObject_t; 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 98
struct char1 { 
#line 100
signed char x; 
#line 101
}; 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 103
struct uchar1 { 
#line 105
unsigned char x; 
#line 106
}; 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 109
struct __declspec(align(2)) char2 { 
#line 111
signed char x, y; 
#line 112
}; 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 114
struct __declspec(align(2)) uchar2 { 
#line 116
unsigned char x, y; 
#line 117
}; 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 119
struct char3 { 
#line 121
signed char x, y, z; 
#line 122
}; 
#endif
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 124
struct uchar3 { 
#line 126
unsigned char x, y, z; 
#line 127
}; 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 129
struct __declspec(align(4)) char4 { 
#line 131
signed char x, y, z, w; 
#line 132
}; 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 134
struct __declspec(align(4)) uchar4 { 
#line 136
unsigned char x, y, z, w; 
#line 137
}; 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 139
struct short1 { 
#line 141
short x; 
#line 142
}; 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 144
struct ushort1 { 
#line 146
unsigned short x; 
#line 147
}; 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 149
struct __declspec(align(4)) short2 { 
#line 151
short x, y; 
#line 152
}; 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 154
struct __declspec(align(4)) ushort2 { 
#line 156
unsigned short x, y; 
#line 157
}; 
#endif
#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 159
struct short3 { 
#line 161
short x, y, z; 
#line 162
}; 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 164
struct ushort3 { 
#line 166
unsigned short x, y, z; 
#line 167
}; 
#endif
#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 169
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 170
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 172
struct int1 { 
#line 174
int x; 
#line 175
}; 
#endif
#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 177
struct uint1 { 
#line 179
unsigned x; 
#line 180
}; 
#endif
#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 182
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 183
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 185
struct int3 { 
#line 187
int x, y, z; 
#line 188
}; 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 190
struct uint3 { 
#line 192
unsigned x, y, z; 
#line 193
}; 
#endif
#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 195
struct __declspec(align(16)) int4 { 
#line 197
int x, y, z, w; 
#line 198
}; 
#endif
#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 200
struct __declspec(align(16)) uint4 { 
#line 202
unsigned x, y, z, w; 
#line 203
}; 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 205
struct long1 { 
#line 207
long x; 
#line 208
}; 
#endif
#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 210
struct ulong1 { 
#line 212
unsigned long x; 
#line 213
}; 
#endif
#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 216
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 217
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 232
struct long3 { 
#line 234
long x, y, z; 
#line 235
}; 
#endif
#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 237
struct ulong3 { 
#line 239
unsigned long x, y, z; 
#line 240
}; 
#endif
#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 242
struct __declspec(align(16)) long4 { 
#line 244
long x, y, z, w; 
#line 245
}; 
#endif
#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 247
struct __declspec(align(16)) ulong4 { 
#line 249
unsigned long x, y, z, w; 
#line 250
}; 
#endif
#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 252
struct float1 { 
#line 254
float x; 
#line 255
}; 
#endif
#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 274
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 279
struct float3 { 
#line 281
float x, y, z; 
#line 282
}; 
#endif
#line 284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 284
struct __declspec(align(16)) float4 { 
#line 286
float x, y, z, w; 
#line 287
}; 
#endif
#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 289
struct longlong1 { 
#line 291
__int64 x; 
#line 292
}; 
#endif
#line 294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 294
struct ulonglong1 { 
#line 296
unsigned __int64 x; 
#line 297
}; 
#endif
#line 299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 299
struct __declspec(align(16)) longlong2 { 
#line 301
__int64 x, y; 
#line 302
}; 
#endif
#line 304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 304
struct __declspec(align(16)) ulonglong2 { 
#line 306
unsigned __int64 x, y; 
#line 307
}; 
#endif
#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 309
struct longlong3 { 
#line 311
__int64 x, y, z; 
#line 312
}; 
#endif
#line 314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 314
struct ulonglong3 { 
#line 316
unsigned __int64 x, y, z; 
#line 317
}; 
#endif
#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 319
struct __declspec(align(16)) longlong4 { 
#line 321
__int64 x, y, z, w; 
#line 322
}; 
#endif
#line 324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 324
struct __declspec(align(16)) ulonglong4 { 
#line 326
unsigned __int64 x, y, z, w; 
#line 327
}; 
#endif
#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 329
struct double1 { 
#line 331
double x; 
#line 332
}; 
#endif
#line 334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 334
struct __declspec(align(16)) double2 { 
#line 336
double x, y; 
#line 337
}; 
#endif
#line 339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 339
struct double3 { 
#line 341
double x, y, z; 
#line 342
}; 
#endif
#line 344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 344
struct __declspec(align(16)) double4 { 
#line 346
double x, y, z, w; 
#line 347
}; 
#endif
#line 362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef char1 
#line 362
char1; 
#endif
#line 363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 363
uchar1; 
#endif
#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef char2 
#line 364
char2; 
#endif
#line 365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 365
uchar2; 
#endif
#line 366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef char3 
#line 366
char3; 
#endif
#line 367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 367
uchar3; 
#endif
#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef char4 
#line 368
char4; 
#endif
#line 369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 369
uchar4; 
#endif
#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef short1 
#line 370
short1; 
#endif
#line 371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 371
ushort1; 
#endif
#line 372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef short2 
#line 372
short2; 
#endif
#line 373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 373
ushort2; 
#endif
#line 374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef short3 
#line 374
short3; 
#endif
#line 375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 375
ushort3; 
#endif
#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef short4 
#line 376
short4; 
#endif
#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 377
ushort4; 
#endif
#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef int1 
#line 378
int1; 
#endif
#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uint1 
#line 379
uint1; 
#endif
#line 380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef int2 
#line 380
int2; 
#endif
#line 381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uint2 
#line 381
uint2; 
#endif
#line 382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef int3 
#line 382
int3; 
#endif
#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uint3 
#line 383
uint3; 
#endif
#line 384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef int4 
#line 384
int4; 
#endif
#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef uint4 
#line 385
uint4; 
#endif
#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef long1 
#line 386
long1; 
#endif
#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 387
ulong1; 
#endif
#line 388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef long2 
#line 388
long2; 
#endif
#line 389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 389
ulong2; 
#endif
#line 390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef long3 
#line 390
long3; 
#endif
#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 391
ulong3; 
#endif
#line 392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef long4 
#line 392
long4; 
#endif
#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulong4 
#line 393
ulong4; 
#endif
#line 394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef float1 
#line 394
float1; 
#endif
#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef float2 
#line 395
float2; 
#endif
#line 396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef float3 
#line 396
float3; 
#endif
#line 397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef float4 
#line 397
float4; 
#endif
#line 398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 398
longlong1; 
#endif
#line 399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 399
ulonglong1; 
#endif
#line 400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 400
longlong2; 
#endif
#line 401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 401
ulonglong2; 
#endif
#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 402
longlong3; 
#endif
#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 403
ulonglong3; 
#endif
#line 404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef longlong4 
#line 404
longlong4; 
#endif
#line 405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef ulonglong4 
#line 405
ulonglong4; 
#endif
#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef double1 
#line 406
double1; 
#endif
#line 407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef double2 
#line 407
double2; 
#endif
#line 408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef double3 
#line 408
double3; 
#endif
#line 409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef double4 
#line 409
double4; 
#endif
#line 417 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
#line 417
struct dim3 { 
#line 419
unsigned x, y, z; 
#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
}; 
#endif
#line 427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#if 0
typedef dim3 
#line 427
dim3; 
#endif
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\library_types.h"
typedef 
#line 54
enum cudaDataType_t { 
#line 56
CUDA_R_16F = 2, 
#line 57
CUDA_C_16F = 6, 
#line 58
CUDA_R_32F = 0, 
#line 59
CUDA_C_32F = 4, 
#line 60
CUDA_R_64F = 1, 
#line 61
CUDA_C_64F = 5, 
#line 62
CUDA_R_8I = 3, 
#line 63
CUDA_C_8I = 7, 
#line 64
CUDA_R_8U, 
#line 65
CUDA_C_8U, 
#line 66
CUDA_R_32I, 
#line 67
CUDA_C_32I, 
#line 68
CUDA_R_32U, 
#line 69
CUDA_C_32U
#line 70
} cudaDataType; 
#line 78
typedef 
#line 73
enum libraryPropertyType_t { 
#line 75
MAJOR_VERSION, 
#line 76
MINOR_VERSION, 
#line 77
PATCH_LEVEL
#line 78
} libraryPropertyType; 
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 106
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 107
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 108
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 109
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 110
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 111
extern cudaError_t __stdcall cudaGetLastError(); 
#line 112
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 113
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 114
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 115
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 116
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 117
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 118
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 119
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 120
extern cudaError_t __stdcall cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 121
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 122
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 123
extern cudaError_t __stdcall cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 124
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 125
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 126
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 127
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 128
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 129
extern cudaError_t __stdcall cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 130
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 131
extern cudaError_t __stdcall cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 132
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 133
extern cudaError_t __stdcall cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 134
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 135
extern cudaError_t __stdcall cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 136
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 137
extern cudaError_t __stdcall cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 138
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 139
extern cudaError_t __stdcall cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 140
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 161
extern void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size); 
#line 189
extern void *__stdcall cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 190
extern cudaError_t __stdcall cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 191
extern cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 210
extern cudaError_t __stdcall cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 214
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 216
}
#line 218
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 219
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 220
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 221
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
extern "C" {
#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 269
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 344
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 373
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 404
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 439
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 481
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 510
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 552
extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 575
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 602
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 644
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 679
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 717
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 767
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 797
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 837
extern cudaError_t __stdcall cudaThreadExit(); 
#line 861
extern cudaError_t __stdcall cudaThreadSynchronize(); 
#line 908
extern cudaError_t __stdcall cudaThreadSetLimit(cudaLimit limit, size_t value); 
#line 939
extern cudaError_t __stdcall cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
#line 974
extern cudaError_t __stdcall cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 1020
extern cudaError_t __stdcall cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
#line 1074
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1115
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1130
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1145
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1175
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1421
extern cudaError_t __stdcall cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
#line 1593
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 1628
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 1647
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 1681
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 1698
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 1727
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 1789
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 1830
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 1867
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 1896
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 1939
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 1963
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 1984
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2005
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2037
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 2051 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2108
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2128
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2149
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 2220
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 2256
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 2290
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 2321
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 2353
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 2385
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 2410
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 2451
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 2510
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 2560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 2614
extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 2648
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 2671
extern cudaError_t __stdcall cudaSetDoubleForDevice(double * d); 
#line 2694
extern cudaError_t __stdcall cudaSetDoubleForHost(double * d); 
#line 2750
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 2794
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 2845 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0); 
#line 2874
extern cudaError_t __stdcall cudaSetupArgument(const void * arg, size_t size, size_t offset); 
#line 2912
extern cudaError_t __stdcall cudaLaunch(const void * func); 
#line 3030
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 3056
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 3085
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 3124
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 3166
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 3193
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 3213
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 3235
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 3257
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 3316
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 3393
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 3412
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 3454
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 3473
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 3508
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 3643
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 3764
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 3790
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 3890
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 3918
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 4029
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 4052
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 4071
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 4092
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 4130
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 4162
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 4200
extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 4237
extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 4275
extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 4320
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4366
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4412
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4455
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 4493
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 4531
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 4582
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4614
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 4660
extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4705
extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4765
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4820
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4874
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4920
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4966
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 4992
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 5022
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 5065
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 5097
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 5134
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 5184
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 5207
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 5229
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 5296
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
#line 5382
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
#line 5438
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 5474
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 5627
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 5665
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 5704
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 5723
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 5783
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 5815
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 5851
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 5883
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 5912
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 5946
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 5971
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 6011
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 6046
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 6093
extern cudaError_t __stdcall cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = 4294967295U); 
#line 6144
extern cudaError_t __stdcall cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
#line 6172
extern cudaError_t __stdcall cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 6200
extern cudaError_t __stdcall cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
#line 6221
extern cudaError_t __stdcall cudaUnbindTexture(const textureReference * texref); 
#line 6246
extern cudaError_t __stdcall cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
#line 6271
extern cudaError_t __stdcall cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
#line 6311
extern cudaError_t __stdcall cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 6330
extern cudaError_t __stdcall cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
#line 6556
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 6571
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 6587
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 6603
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 6620
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 6659
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 6674
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 6689
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 6716
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 6733
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 6738
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 6964 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_runtime_api.h"
}
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 108
{ 
#line 109
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 110
} 
#line 112
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 113
{ 
#line 114
int e = (((int)sizeof(unsigned short)) * 8); 
#line 116
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 117
} 
#line 119
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 120
{ 
#line 121
int e = (((int)sizeof(unsigned short)) * 8); 
#line 123
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 124
} 
#line 126
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 127
{ 
#line 128
int e = (((int)sizeof(unsigned short)) * 8); 
#line 130
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 131
} 
#line 133
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 134
{ 
#line 135
int e = (((int)sizeof(unsigned short)) * 8); 
#line 137
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 138
} 
#line 140
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 141
{ 
#line 142
int e = (((int)sizeof(char)) * 8); 
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\channel_descriptor.h"
} 
#line 151
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 152
{ 
#line 153
int e = (((int)sizeof(signed char)) * 8); 
#line 155
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 156
} 
#line 158
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 159
{ 
#line 160
int e = (((int)sizeof(unsigned char)) * 8); 
#line 162
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 163
} 
#line 165
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 166
{ 
#line 167
int e = (((int)sizeof(signed char)) * 8); 
#line 169
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 170
} 
#line 172
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 173
{ 
#line 174
int e = (((int)sizeof(unsigned char)) * 8); 
#line 176
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 177
} 
#line 179
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 180
{ 
#line 181
int e = (((int)sizeof(signed char)) * 8); 
#line 183
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 184
} 
#line 186
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 187
{ 
#line 188
int e = (((int)sizeof(unsigned char)) * 8); 
#line 190
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 191
} 
#line 193
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 194
{ 
#line 195
int e = (((int)sizeof(signed char)) * 8); 
#line 197
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 198
} 
#line 200
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 201
{ 
#line 202
int e = (((int)sizeof(unsigned char)) * 8); 
#line 204
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 205
} 
#line 207
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 208
{ 
#line 209
int e = (((int)sizeof(short)) * 8); 
#line 211
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 212
} 
#line 214
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 215
{ 
#line 216
int e = (((int)sizeof(unsigned short)) * 8); 
#line 218
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 219
} 
#line 221
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 222
{ 
#line 223
int e = (((int)sizeof(short)) * 8); 
#line 225
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 226
} 
#line 228
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 229
{ 
#line 230
int e = (((int)sizeof(unsigned short)) * 8); 
#line 232
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 233
} 
#line 235
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 236
{ 
#line 237
int e = (((int)sizeof(short)) * 8); 
#line 239
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 240
} 
#line 242
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 243
{ 
#line 244
int e = (((int)sizeof(unsigned short)) * 8); 
#line 246
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 247
} 
#line 249
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 250
{ 
#line 251
int e = (((int)sizeof(short)) * 8); 
#line 253
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 254
} 
#line 256
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 257
{ 
#line 258
int e = (((int)sizeof(unsigned short)) * 8); 
#line 260
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 261
} 
#line 263
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 264
{ 
#line 265
int e = (((int)sizeof(int)) * 8); 
#line 267
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 268
} 
#line 270
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 271
{ 
#line 272
int e = (((int)sizeof(unsigned)) * 8); 
#line 274
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 275
} 
#line 277
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 278
{ 
#line 279
int e = (((int)sizeof(int)) * 8); 
#line 281
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 282
} 
#line 284
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 285
{ 
#line 286
int e = (((int)sizeof(unsigned)) * 8); 
#line 288
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 289
} 
#line 291
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 292
{ 
#line 293
int e = (((int)sizeof(int)) * 8); 
#line 295
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 296
} 
#line 298
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 299
{ 
#line 300
int e = (((int)sizeof(unsigned)) * 8); 
#line 302
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 303
} 
#line 305
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 306
{ 
#line 307
int e = (((int)sizeof(int)) * 8); 
#line 309
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 310
} 
#line 312
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 313
{ 
#line 314
int e = (((int)sizeof(unsigned)) * 8); 
#line 316
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 317
} 
#line 321
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 322
{ 
#line 323
int e = (((int)sizeof(long)) * 8); 
#line 325
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 326
} 
#line 328
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 329
{ 
#line 330
int e = (((int)sizeof(unsigned long)) * 8); 
#line 332
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 333
} 
#line 335
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 336
{ 
#line 337
int e = (((int)sizeof(long)) * 8); 
#line 339
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 340
} 
#line 342
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 343
{ 
#line 344
int e = (((int)sizeof(unsigned long)) * 8); 
#line 346
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 347
} 
#line 349
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 350
{ 
#line 351
int e = (((int)sizeof(long)) * 8); 
#line 353
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 354
} 
#line 356
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 357
{ 
#line 358
int e = (((int)sizeof(unsigned long)) * 8); 
#line 360
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 361
} 
#line 363
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 364
{ 
#line 365
int e = (((int)sizeof(long)) * 8); 
#line 367
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 368
} 
#line 370
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 371
{ 
#line 372
int e = (((int)sizeof(unsigned long)) * 8); 
#line 374
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 375
} 
#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 380
{ 
#line 381
int e = (((int)sizeof(float)) * 8); 
#line 383
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 384
} 
#line 386
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 387
{ 
#line 388
int e = (((int)sizeof(float)) * 8); 
#line 390
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 391
} 
#line 393
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 394
{ 
#line 395
int e = (((int)sizeof(float)) * 8); 
#line 397
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 398
} 
#line 400
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 401
{ 
#line 402
int e = (((int)sizeof(float)) * 8); 
#line 404
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 405
} 
#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 76
{ 
#line 77
char1 t; (t.x) = x; return t; 
#line 78
} 
#line 80
static __inline uchar1 make_uchar1(unsigned char x) 
#line 81
{ 
#line 82
uchar1 t; (t.x) = x; return t; 
#line 83
} 
#line 85
static __inline char2 make_char2(signed char x, signed char y) 
#line 86
{ 
#line 87
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 88
} 
#line 90
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 91
{ 
#line 92
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 93
} 
#line 95
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 96
{ 
#line 97
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 98
} 
#line 100
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 101
{ 
#line 102
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 103
} 
#line 105
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 106
{ 
#line 107
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 108
} 
#line 110
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 111
{ 
#line 112
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 113
} 
#line 115
static __inline short1 make_short1(short x) 
#line 116
{ 
#line 117
short1 t; (t.x) = x; return t; 
#line 118
} 
#line 120
static __inline ushort1 make_ushort1(unsigned short x) 
#line 121
{ 
#line 122
ushort1 t; (t.x) = x; return t; 
#line 123
} 
#line 125
static __inline short2 make_short2(short x, short y) 
#line 126
{ 
#line 127
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 128
} 
#line 130
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 131
{ 
#line 132
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 133
} 
#line 135
static __inline short3 make_short3(short x, short y, short z) 
#line 136
{ 
#line 137
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 138
} 
#line 140
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 141
{ 
#line 142
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 143
} 
#line 145
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 146
{ 
#line 147
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 148
} 
#line 150
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 151
{ 
#line 152
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 153
} 
#line 155
static __inline int1 make_int1(int x) 
#line 156
{ 
#line 157
int1 t; (t.x) = x; return t; 
#line 158
} 
#line 160
static __inline uint1 make_uint1(unsigned x) 
#line 161
{ 
#line 162
uint1 t; (t.x) = x; return t; 
#line 163
} 
#line 165
static __inline int2 make_int2(int x, int y) 
#line 166
{ 
#line 167
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 168
} 
#line 170
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 171
{ 
#line 172
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 173
} 
#line 175
static __inline int3 make_int3(int x, int y, int z) 
#line 176
{ 
#line 177
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 178
} 
#line 180
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 181
{ 
#line 182
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 183
} 
#line 185
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 186
{ 
#line 187
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 188
} 
#line 190
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 191
{ 
#line 192
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 193
} 
#line 195
static __inline long1 make_long1(long x) 
#line 196
{ 
#line 197
long1 t; (t.x) = x; return t; 
#line 198
} 
#line 200
static __inline ulong1 make_ulong1(unsigned long x) 
#line 201
{ 
#line 202
ulong1 t; (t.x) = x; return t; 
#line 203
} 
#line 205
static __inline long2 make_long2(long x, long y) 
#line 206
{ 
#line 207
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 208
} 
#line 210
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 211
{ 
#line 212
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 213
} 
#line 215
static __inline long3 make_long3(long x, long y, long z) 
#line 216
{ 
#line 217
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 218
} 
#line 220
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 221
{ 
#line 222
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 223
} 
#line 225
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 226
{ 
#line 227
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 228
} 
#line 230
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 231
{ 
#line 232
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 233
} 
#line 235
static __inline float1 make_float1(float x) 
#line 236
{ 
#line 237
float1 t; (t.x) = x; return t; 
#line 238
} 
#line 240
static __inline float2 make_float2(float x, float y) 
#line 241
{ 
#line 242
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 243
} 
#line 245
static __inline float3 make_float3(float x, float y, float z) 
#line 246
{ 
#line 247
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 248
} 
#line 250
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 251
{ 
#line 252
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 253
} 
#line 255
static __inline longlong1 make_longlong1(__int64 x) 
#line 256
{ 
#line 257
longlong1 t; (t.x) = x; return t; 
#line 258
} 
#line 260
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 261
{ 
#line 262
ulonglong1 t; (t.x) = x; return t; 
#line 263
} 
#line 265
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 266
{ 
#line 267
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 268
} 
#line 270
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 271
{ 
#line 272
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 273
} 
#line 275
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 276
{ 
#line 277
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 278
} 
#line 280
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 281
{ 
#line 282
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 283
} 
#line 285
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 286
{ 
#line 287
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 288
} 
#line 290
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 291
{ 
#line 292
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 293
} 
#line 295
static __inline double1 make_double1(double x) 
#line 296
{ 
#line 297
double1 t; (t.x) = x; return t; 
#line 298
} 
#line 300
static __inline double2 make_double2(double x, double y) 
#line 301
{ 
#line 302
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 303
} 
#line 305
static __inline double3 make_double3(double x, double y, double z) 
#line 306
{ 
#line 307
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 308
} 
#line 310
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 311
{ 
#line 312
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 313
} 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\errno.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 17
__declspec(dllimport) extern int *__cdecl _errno(); 
#line 20
__declspec(dllimport) errno_t __cdecl _set_errno(int _Value); 
#line 21
__declspec(dllimport) errno_t __cdecl _get_errno(int * _Value); 
#line 25
__declspec(dllimport) unsigned long *__cdecl __doserrno(); 
#line 28
__declspec(dllimport) errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 29
__declspec(dllimport) errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 129 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime_string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
__declspec(dllimport) const void *__cdecl memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 26
int __cdecl memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 47
__declspec(dllimport) void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 67
__declspec(dllimport) const char *__cdecl strchr(const char * _Str, int _Val); 
#line 73
__declspec(dllimport) const char *__cdecl strrchr(const char * _Str, int _Ch); 
#line 79
__declspec(dllimport) const char *__cdecl strstr(const char * _Str, const char * _SubStr); 
#line 86
__declspec(dllimport) const __wchar_t *__cdecl wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 92
__declspec(dllimport) const __wchar_t *__cdecl wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 99
__declspec(dllimport) const __wchar_t *__cdecl wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 106
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memcpy_s.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 36
_Destination, const rsize_t 
#line 37
_DestinationSize, const void *const 
#line 38
_Source, const rsize_t 
#line 39
_SourceSize) 
#line 41
{ 
#line 42
if (_SourceSize == (0)) 
#line 43
{ 
#line 44
return 0; 
#line 45
}  
#line 47
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 48
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 49
{ 
#line 50
memset(_Destination, 0, _DestinationSize); 
#line 52
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 53
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 56
return 22; 
#line 57
}  
#line 59
memcpy(_Destination, _Source, _SourceSize); 
#line 60
return 0; 
#line 61
} 
#line 64
static __inline errno_t __cdecl memmove_s(void *const 
#line 65
_Destination, const rsize_t 
#line 66
_DestinationSize, const void *const 
#line 67
_Source, const rsize_t 
#line 68
_SourceSize) 
#line 70
{ 
#line 71
if (_SourceSize == (0)) 
#line 72
{ 
#line 73
return 0; 
#line 74
}  
#line 76
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 77
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 78
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 80
memmove(_Destination, _Source, _SourceSize); 
#line 81
return 0; 
#line 82
} 
#line 88 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memcpy_s.h"
}__pragma( pack ( pop )) 
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memory.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 24
__declspec(dllimport) int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 31
__declspec(dllimport) int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 78 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memory.h"
__declspec(dllimport) void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 86
__declspec(dllimport) int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 97 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 98
_Pv, int 
#line 99
_C, size_t 
#line 100
_N) 
#line 102
{ 
#line 103
const void *const _Pvc = _Pv; 
#line 104
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 105
} 
#line 110 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
__declspec(dllimport) errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 35
__declspec(dllimport) errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 42
__declspec(dllimport) errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 50
__declspec(dllimport) errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 58
__declspec(dllimport) __wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 79 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 89 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl wcscat_s(__wchar_t (&_Destination)[_Size], const __wchar_t *_Source) throw() { return wcscat_s(_Destination, _Size, _Source); } }
#line 97 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 106 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 111
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl wcscpy_s(__wchar_t (&_Destination)[_Size], const __wchar_t *_Source) throw() { return wcscpy_s(_Destination, _Size, _Source); } }
#line 117 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 124 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 130
__declspec(dllimport) size_t __cdecl wcslen(const __wchar_t * _String); 
#line 143 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 160
_Source, size_t 
#line 161
_MaxCount) 
#line 163
{ 
#line 164
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 165
} 
#line 169 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl wcsncat_s(__wchar_t (&_Destination)[_Size], const __wchar_t *_Source, ::size_t _Count) throw() { return wcsncat_s(_Destination, _Size, _Source, _Count); } }
#line 176 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 185 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 191
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl wcsncpy_s(__wchar_t (&_Destination)[_Size], const __wchar_t *_Source, ::size_t _Count) throw() { return wcsncpy_s(_Destination, _Size, _Source, _Count); } }
#line 198 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 207 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 213
__declspec(dllimport) size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 219
__declspec(dllimport) __wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 236 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4141 4996)
#line 240
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 241
_String, const __wchar_t *const 
#line 242
_Delimiter) 
#line 244
{ 
#line 245
return wcstok(_String, _Delimiter, 0); 
#line 246
} 
#line 254 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 253
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 254
wcstok(__wchar_t *
#line 255
_String, const __wchar_t *
#line 256
_Delimiter) throw() 
#line 258
{ 
#line 259
return wcstok(_String, _Delimiter, 0); 
#line 260
} 
#line 263 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 271 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 276
__declspec(dllimport) errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 282
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcserror_s(__wchar_t (&_Buffer)[_Size], int _Error) throw() { return _wcserror_s(_Buffer, _Size, _Error); } }
#line 291 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 295
__declspec(dllimport) errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 301
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl __wcserror_s(__wchar_t (&_Buffer)[_Size], const __wchar_t *_ErrorMessage) throw() { return __wcserror_s(_Buffer, _Size, _ErrorMessage); } }
#line 307 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 312
__declspec(dllimport) int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 318
__declspec(dllimport) int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 324
__declspec(dllimport) int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 331
__declspec(dllimport) errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 338
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcsnset_s(__wchar_t (&_Destination)[_Size], __wchar_t _Value, ::size_t _MaxCount) throw() { return _wcsnset_s(_Destination, _Size, _Value, _MaxCount); } }
#line 345 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 353 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 357
__declspec(dllimport) errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 363
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcsset_s(__wchar_t (&_String)[_Size], __wchar_t _Value) throw() { return _wcsset_s(_String, _Size, _Value); } }
#line 369 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 376 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 381
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcslwr_s(__wchar_t (&_String)[_Size]) throw() { return _wcslwr_s(_String, _Size); } }
#line 386 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 392 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 398
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcslwr_s_l(__wchar_t (&_String)[_Size], ::_locale_t _Locale) throw() { return _wcslwr_s_l(_String, _Size, _Locale); } }
#line 404 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 412 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 417
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcsupr_s(__wchar_t (&_String)[_Size]) throw() { return _wcsupr_s(_String, _Size); } }
#line 422 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 428 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 434
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcsupr_s_l(__wchar_t (&_String)[_Size], ::_locale_t _Locale) throw() { return _wcsupr_s_l(_String, _Size, _Locale); } }
#line 440 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 449 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 457
__declspec(dllimport) size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 465
__declspec(dllimport) int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 471
__declspec(dllimport) int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 478
__declspec(dllimport) int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 484
__declspec(dllimport) int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 491
__declspec(dllimport) int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 498
__declspec(dllimport) int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 506
__declspec(dllimport) int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 513
__declspec(dllimport) int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 528
extern "C++" {
#line 532
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 533
{ 
#line 534
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 535
} 
#line 538
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 539
{ 
#line 540
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 541
} 
#line 544
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 545
{ 
#line 546
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 547
} 
#line 551
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 552
{ 
#line 553
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 554
} 
#line 556
}
#line 573 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) __wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 585 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
__declspec(dllimport) int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 591
__declspec(dllimport) int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 599
__declspec(dllimport) __wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 607
__declspec(dllimport) __wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 613
__declspec(dllimport) __wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 620
__declspec(dllimport) __wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 626
__declspec(dllimport) __wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 631
__declspec(dllimport) int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 640 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 27
__declspec(dllimport) errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 34
__declspec(dllimport) errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 41
__declspec(dllimport) errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 47
__declspec(dllimport) errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 55
__declspec(dllimport) errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 63
__declspec(dllimport) char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 71 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 78
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl strcat_s(char (&_Destination)[_Size], const char *_Source) throw() { return strcat_s(_Destination, _Size, _Source); } }
#line 86 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 95 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 101
__declspec(dllimport) int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 107
__declspec(dllimport) int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 113
__declspec(dllimport) int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 119
char *__cdecl strcpy(char * _Dest, const char * _Source); 
#line 124
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl strcpy_s(char (&_Destination)[_Size], const char *_Source) throw() { return strcpy_s(_Destination, _Size, _Source); } }
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) __declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strerror(const char * _ErrorMessage); 
#line 164
__declspec(dllimport) errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 170
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strerror_s(char (&_Buffer)[_Size], const char *_ErrorMessage) throw() { return _strerror_s(_Buffer, _Size, _ErrorMessage); } }
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl strerror(int _ErrorMessage); 
#line 182
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl strerror_s(char (&_Buffer)[_Size], int _ErrorMessage) throw() { return strerror_s(_Buffer, _Size, _ErrorMessage); } }
#line 189 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 195
__declspec(dllimport) int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 201
__declspec(dllimport) int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 208
__declspec(dllimport) int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 215
size_t __cdecl strlen(const char * _Str); 
#line 220
__declspec(dllimport) errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 225
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strlwr_s(char (&_String)[_Size]) throw() { return _strlwr_s(_String, _Size); } }
#line 230 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strlwr(char * _String); 
#line 236 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 242
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strlwr_s_l(char (&_String)[_Size], ::_locale_t _Locale) throw() { return _strlwr_s_l(_String, _Size, _Locale); } }
#line 248 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl strncat(char * _Dest, const char * _Source, size_t _Count); 
#line 261
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl strncat_s(char (&_Destination)[_Size], const char *_Source, ::size_t _Count) throw() { return strncat_s(_Destination, _Size, _Source, _Count); } }
#line 268 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 277 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 284
__declspec(dllimport) int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 291
__declspec(dllimport) int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 299
__declspec(dllimport) int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 306
__declspec(dllimport) int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 314
__declspec(dllimport) int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 321
__declspec(dllimport) int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 328
__declspec(dllimport) size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 333
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl strncpy_s(char (&_Destination)[_Size], const char *_Source, ::size_t _Count) throw() { return strncpy_s(_Destination, _Size, _Source, _Count); } }
#line 340 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 357 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 373 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 374
_String, size_t 
#line 375
_MaxCount) 
#line 377
{ 
#line 378
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 379
} 
#line 383 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strnset(char * _Dest, int _Val, size_t _Count); 
#line 390
__declspec(dllimport) errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 397
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strnset_s(char (&_Destination)[_Size], int _Value, ::size_t _Count) throw() { return _strnset_s(_Destination, _Size, _Value, _Count); } }
#line 404 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 413 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 418
__declspec(dllimport) char *__cdecl _strrev(char * _Str); 
#line 423
__declspec(dllimport) errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 429
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strset_s(char (&_Destination)[_Size], int _Value) throw() { return _strset_s(_Destination, _Size, _Value); } }
#line 435 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 441 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Dest, int _Value); 
#line 447
__declspec(dllimport) size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 453
__declspec(dllimport) char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 459
__declspec(dllimport) errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 464
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strupr_s(char (&_String)[_Size]) throw() { return _strupr_s(_String, _Size); } }
#line 469 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strupr(char * _String); 
#line 475 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 481
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strupr_s_l(char (&_String)[_Size], ::_locale_t _Locale) throw() { return _strupr_s_l(_String, _Size, _Locale); } }
#line 487 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 496 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 504
__declspec(dllimport) size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 514
extern "C++" {
#line 517
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 518
{ 
#line 519
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 520
} 
#line 523
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 524
{ 
#line 525
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 526
} 
#line 529
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 530
{ 
#line 531
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 532
} 
#line 535
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 536
{ 
#line 537
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 538
} 
#line 539
}
#line 552 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) char *__cdecl strdup(const char * _String); 
#line 562 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
__declspec(dllimport) int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 568
__declspec(dllimport) int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 574
__declspec(dllimport) char *__cdecl strlwr(char * _String); 
#line 579
__declspec(dllimport) int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 586
__declspec(dllimport) char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 593
__declspec(dllimport) char *__cdecl strrev(char * _String); 
#line 598
char *__cdecl strset(char * _String, int _Value); 
#line 603
__declspec(dllimport) char *__cdecl strupr(char * _String); 
#line 611 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 22
struct tm { 
#line 24
int tm_sec; 
#line 25
int tm_min; 
#line 26
int tm_hour; 
#line 27
int tm_mday; 
#line 28
int tm_mon; 
#line 29
int tm_year; 
#line 30
int tm_wday; 
#line 31
int tm_yday; 
#line 32
int tm_isdst; 
#line 33
}; 
#line 44
__declspec(dllimport) __wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 50
__declspec(dllimport) errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 56
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wasctime_s(__wchar_t (&_Buffer)[_Size], const ::tm *_Time) throw() { return _wasctime_s(_Buffer, _Size, _Time); } }
#line 65 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 74
__declspec(dllimport) size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 84
__declspec(dllimport) __wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 89
__declspec(dllimport) errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 95
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wctime32_s(__wchar_t (&_Buffer)[_Size], const ::__time32_t *_Time) throw() { return _wctime32_s(_Buffer, _Size, _Time); } }
#line 104 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) __wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 109
__declspec(dllimport) errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 114
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wctime64_s(__wchar_t (&_Buffer)[_Size], const ::__time64_t *_Time) throw() { return _wctime64_s(_Buffer, _Size, _Time); } }
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 126
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wstrdate_s(__wchar_t (&_Buffer)[_Size]) throw() { return _wstrdate_s(_Buffer, _Size); } }
#line 131 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) __wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 142
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wstrtime_s(__wchar_t (&_Buffer)[_Size]) throw() { return _wstrtime_s(_Buffer, _Size); } }
#line 147 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__declspec(dllimport) __wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 160 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4996)
#line 185
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 186
_Time) 
#line 187
{ 
#line 188
return _wctime64(_Time); 
#line 189
} 
#line 192
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 193
_Buffer, const size_t 
#line 194
_SizeInWords, const time_t *const 
#line 195
_Time) 
#line 197
{ 
#line 198
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 199
} 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
}
#line 203 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
#pragma warning(pop)
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef long clock_t; 
#line 25
struct _timespec32 { 
#line 27
__time32_t tv_sec; 
#line 28
long tv_nsec; 
#line 29
}; 
#line 31
struct _timespec64 { 
#line 33
__time64_t tv_sec; 
#line 34
long tv_nsec; 
#line 35
}; 
#line 38
struct timespec { 
#line 40
time_t tv_sec; 
#line 41
long tv_nsec; 
#line 42
}; 
#line 61 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) int *__cdecl __daylight(); 
#line 67
__declspec(dllimport) long *__cdecl __dstbias(); 
#line 73
__declspec(dllimport) long *__cdecl __timezone(); 
#line 79
__declspec(dllimport) char **__cdecl __tzname(); 
#line 84
__declspec(dllimport) errno_t __cdecl _get_daylight(int * _Daylight); 
#line 89
__declspec(dllimport) errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 94
__declspec(dllimport) errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 99
__declspec(dllimport) errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 116
__declspec(dllimport) char *__cdecl asctime(const tm * _Tm); 
#line 123
__declspec(dllimport) errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl asctime_s(char (&_Buffer)[_Size], const ::tm *_Time) throw() { return asctime_s(_Buffer, _Size, _Time); } }
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) clock_t __cdecl clock(); 
#line 142
__declspec(dllimport) char *__cdecl _ctime32(const __time32_t * _Time); 
#line 147
__declspec(dllimport) errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 153
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ctime32_s(char (&_Buffer)[_Size], const ::__time32_t *_Time) throw() { return _ctime32_s(_Buffer, _Size, _Time); } }
#line 162 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) char *__cdecl _ctime64(const __time64_t * _Time); 
#line 167
__declspec(dllimport) errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 173
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ctime64_s(char (&_Buffer)[_Size], const ::__time64_t *_Time) throw() { return _ctime64_s(_Buffer, _Size, _Time); } }
#line 180 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 186
__declspec(dllimport) double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 193
__declspec(dllimport) tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 198
__declspec(dllimport) errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 205
__declspec(dllimport) tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 210
__declspec(dllimport) errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 217
__declspec(dllimport) tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 222
__declspec(dllimport) errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 229
__declspec(dllimport) tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 234
__declspec(dllimport) errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 240
__declspec(dllimport) __time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 245
__declspec(dllimport) __time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 250
__declspec(dllimport) __time32_t __cdecl _mktime32(tm * _Tm); 
#line 255
__declspec(dllimport) __time64_t __cdecl _mktime64(tm * _Tm); 
#line 261
__declspec(dllimport) size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 270
__declspec(dllimport) size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 279
__declspec(dllimport) errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 284
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strdate_s(char (&_Buffer)[_Size]) throw() { return _strdate_s(_Buffer, _Size); } }
#line 289 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) char *__cdecl _strdate(char * _Buffer); 
#line 295 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 300
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _strtime_s(char (&_Buffer)[_Size]) throw() { return _strtime_s(_Buffer, _Size); } }
#line 305 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) char *__cdecl _strtime(char * _Buffer); 
#line 310 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) __time32_t __cdecl _time32(__time32_t * _Time); 
#line 314
__declspec(dllimport) __time64_t __cdecl _time64(__time64_t * _Time); 
#line 320
__declspec(dllimport) int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 327
__declspec(dllimport) int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 341
__declspec(dllimport) void __cdecl _tzset(); 
#line 344
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) 
#line 345
__declspec(dllimport) unsigned __cdecl _getsystime(tm * _Tm); 
#line 349
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) 
#line 350
__declspec(dllimport) unsigned __cdecl _setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 475 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 476
_Time) 
#line 478
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _ctime64(_Time); 
#pragma warning(pop)
} 
#line 486
static __inline double __cdecl difftime(const time_t 
#line 487
_Time1, const time_t 
#line 488
_Time2) 
#line 490
{ 
#line 491
return _difftime64(_Time1, _Time2); 
#line 492
} 
#line 495
static __inline tm *__cdecl gmtime(const time_t *const 
#line 496
_Time) 
#line 497
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _gmtime64(_Time); 
#pragma warning(pop)
} 
#line 505
static __inline tm *__cdecl localtime(const time_t *const 
#line 506
_Time) 
#line 508
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _localtime64(_Time); 
#pragma warning(pop)
} 
#line 516
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 517
_Tm) 
#line 519
{ 
#line 520
return _mkgmtime64(_Tm); 
#line 521
} 
#line 524
static __inline time_t __cdecl mktime(tm *const 
#line 525
_Tm) 
#line 527
{ 
#line 528
return _mktime64(_Tm); 
#line 529
} 
#line 531
static __inline time_t __cdecl time(time_t *const 
#line 532
_Time) 
#line 534
{ 
#line 535
return _time64(_Time); 
#line 536
} 
#line 539
static __inline int __cdecl timespec_get(timespec *const 
#line 540
_Ts, const int 
#line 541
_Base) 
#line 543
{ 
#line 544
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 545
} 
#line 549
static __inline errno_t __cdecl ctime_s(char *const 
#line 550
_Buffer, const size_t 
#line 551
_SizeInBytes, const time_t *const 
#line 552
_Time) 
#line 554
{ 
#line 555
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 556
} 
#line 559
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 560
_Tm, const time_t *const 
#line 561
_Time) 
#line 563
{ 
#line 564
return _gmtime64_s(_Tm, _Time); 
#line 565
} 
#line 568
static __inline errno_t __cdecl localtime_s(tm *const 
#line 569
_Tm, const time_t *const 
#line 570
_Time) 
#line 572
{ 
#line 573
return _localtime64_s(_Tm, _Time); 
#line 574
} 
#line 593 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
__declspec(dllimport) void __cdecl tzset(); 
#line 600 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\common_functions.h"
extern "C" {
#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\common_functions.h"
__declspec(dllimport) extern clock_t __cdecl clock(); 
#line 72
extern void *__cdecl memset(void *, int, size_t); 
#line 73
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 75
}
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern "C" {
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __cdecl abs(int); 
#line 165
extern long __cdecl labs(long); 
#line 166
extern __int64 llabs(__int64); 
#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 257
extern __inline float fabsf(float x); 
#line 261
extern inline int min(int, int); 
#line 263
extern inline unsigned umin(unsigned, unsigned); 
#line 264
extern inline __int64 llmin(__int64, __int64); 
#line 265
extern inline unsigned __int64 ullmin(unsigned __int64, unsigned __int64); 
#line 288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl fminf(float x, float y); 
#line 308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl fmin(double x, double y); 
#line 313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline int max(int, int); 
#line 315
extern inline unsigned umax(unsigned, unsigned); 
#line 316
extern inline __int64 llmax(__int64, __int64); 
#line 317
extern inline unsigned __int64 ullmax(unsigned __int64, unsigned __int64); 
#line 340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl fmaxf(float x, float y); 
#line 360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl fmax(double, double); 
#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double __cdecl sin(double x); 
#line 435
extern double __cdecl cos(double x); 
#line 454
extern void sincos(double x, double * sptr, double * cptr); 
#line 470
extern void sincosf(float x, float * sptr, float * cptr); 
#line 515
extern double __cdecl tan(double x); 
#line 584
extern double __cdecl sqrt(double x); 
#line 656
extern double rsqrt(double x); 
#line 726
extern float rsqrtf(float x); 
#line 784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl log2(double x); 
#line 809 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl exp2(double x); 
#line 834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl exp2f(float x); 
#line 859 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double exp10(double x); 
#line 882
extern float exp10f(float x); 
#line 930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl expm1(double x); 
#line 975 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl expm1f(float x); 
#line 1030 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl log2f(float x); 
#line 1082 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double __cdecl log10(double x); 
#line 1153
extern double __cdecl log(double x); 
#line 1249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl log1p(double x); 
#line 1346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl log1pf(float x); 
#line 1419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl floor(double x); 
#line 1458
extern double __cdecl exp(double x); 
#line 1489
extern double __cdecl cosh(double x); 
#line 1519
extern double __cdecl sinh(double x); 
#line 1549
extern double __cdecl tanh(double x); 
#line 1586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl acosh(double x); 
#line 1624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl acoshf(float x); 
#line 1640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl asinh(double x); 
#line 1656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl asinhf(float x); 
#line 1710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl atanh(double x); 
#line 1764 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl atanhf(float x); 
#line 1821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl ldexp(double x, int exp); 
#line 1877
extern __inline float ldexpf(float x, int exp); 
#line 1931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl logb(double x); 
#line 1986 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl logbf(float x); 
#line 2016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern int __cdecl ilogb(double x); 
#line 2046 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern int __cdecl ilogbf(float x); 
#line 2122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl scalbn(double x, int n); 
#line 2198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl scalbnf(float x, int n); 
#line 2274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl scalbln(double x, long n); 
#line 2350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl scalblnf(float x, long n); 
#line 2426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl frexp(double x, int * nptr); 
#line 2501
extern __inline float frexpf(float x, int * nptr); 
#line 2517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl round(double x); 
#line 2534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl roundf(float x); 
#line 2552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern long __cdecl lround(double x); 
#line 2570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern long __cdecl lroundf(float x); 
#line 2588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern __int64 __cdecl llround(double x); 
#line 2606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern __int64 __cdecl llroundf(float x); 
#line 2658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl rintf(float x); 
#line 2674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern long __cdecl lrint(double x); 
#line 2690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern long __cdecl lrintf(float x); 
#line 2706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern __int64 __cdecl llrint(double x); 
#line 2722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern __int64 __cdecl llrintf(float x); 
#line 2775 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl nearbyint(double x); 
#line 2828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl nearbyintf(float x); 
#line 2888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl ceil(double x); 
#line 2902 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl trunc(double x); 
#line 2917 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl truncf(float x); 
#line 2943 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl fdim(double x, double y); 
#line 2969 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl fdimf(float x, float y); 
#line 3003 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 3034
extern double __cdecl atan(double x); 
#line 3057
extern double __cdecl acos(double x); 
#line 3089
extern double __cdecl asin(double x); 
#line 3132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl hypot(double x, double y); 
#line 3187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double rhypot(double x, double y); 
#line 3231
static __inline float __cdecl hypotf(float x, float y); 
#line 3285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern float rhypotf(float x, float y); 
#line 3332
__declspec(dllimport) extern double __cdecl norm3d(double a, double b, double c); 
#line 3383
extern double rnorm3d(double a, double b, double c); 
#line 3432
__declspec(dllimport) extern double __cdecl norm4d(double a, double b, double c, double d); 
#line 3488
extern double rnorm4d(double a, double b, double c, double d); 
#line 3533
extern double norm(int dim, const double * t); 
#line 3584
extern double rnorm(int dim, const double * t); 
#line 3636
extern float rnormf(int dim, const float * a); 
#line 3680
extern float normf(int dim, const float * a); 
#line 3725
extern float norm3df(float a, float b, float c); 
#line 3776
extern float rnorm3df(float a, float b, float c); 
#line 3825
extern float norm4df(float a, float b, float c, float d); 
#line 3881
extern float rnorm4df(float a, float b, float c, float d); 
#line 3967 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl cbrt(double x); 
#line 4053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl cbrtf(float x); 
#line 4106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double rcbrt(double x); 
#line 4156
extern float rcbrtf(float x); 
#line 4216
extern double sinpi(double x); 
#line 4276
extern float sinpif(float x); 
#line 4328
extern double cospi(double x); 
#line 4380
extern float cospif(float x); 
#line 4410
extern void sincospi(double x, double * sptr, double * cptr); 
#line 4440
extern void sincospif(float x, float * sptr, float * cptr); 
#line 4752
extern double __cdecl pow(double x, double y); 
#line 4808
__declspec(dllimport) extern double __cdecl modf(double x, double * iptr); 
#line 4867
extern double __cdecl fmod(double x, double y); 
#line 4955 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl remainder(double x, double y); 
#line 5045 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl remainderf(float x, float y); 
#line 5099 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl remquo(double x, double y, int * quo); 
#line 5153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl remquof(float x, float y, int * quo); 
#line 5192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl j0(double x); 
#line 5234
extern float j0f(float x); 
#line 5295
__declspec(dllimport) extern double __cdecl j1(double x); 
#line 5356
extern float j1f(float x); 
#line 5399
__declspec(dllimport) extern double __cdecl jn(int n, double x); 
#line 5442
extern float jnf(int n, float x); 
#line 5494
__declspec(dllimport) extern double __cdecl y0(double x); 
#line 5546
extern float y0f(float x); 
#line 5598
__declspec(dllimport) extern double __cdecl y1(double x); 
#line 5650
extern float y1f(float x); 
#line 5703
__declspec(dllimport) extern double __cdecl yn(int n, double x); 
#line 5756
extern float ynf(int n, float x); 
#line 5783
__declspec(dllimport) extern double __cdecl cyl_bessel_i0(double x); 
#line 5809
extern float cyl_bessel_i0f(float x); 
#line 5836
__declspec(dllimport) extern double __cdecl cyl_bessel_i1(double x); 
#line 5862
extern float cyl_bessel_i1f(float x); 
#line 5947 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl erf(double x); 
#line 6029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl erff(float x); 
#line 6091 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double erfinv(double y); 
#line 6148
extern float erfinvf(float y); 
#line 6189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl erfc(double x); 
#line 6227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl erfcf(float x); 
#line 6355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl lgamma(double x); 
#line 6416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern double erfcinv(double y); 
#line 6472
extern float erfcinvf(float y); 
#line 6530
extern double normcdfinv(double y); 
#line 6588
extern float normcdfinvf(float y); 
#line 6631
extern double normcdf(double y); 
#line 6674
extern float normcdff(float y); 
#line 6749
extern double erfcx(double x); 
#line 6824
extern float erfcxf(float x); 
#line 6960 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl lgammaf(float x); 
#line 7069 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl tgamma(double x); 
#line 7178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl tgammaf(float x); 
#line 7191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl copysign(double x, double y); 
#line 7204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl copysignf(float x, float y); 
#line 7241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl nextafter(double x, double y); 
#line 7278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl nextafterf(float x, float y); 
#line 7294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl nan(const char * tagp); 
#line 7310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl nanf(const char * tagp); 
#line 7315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __isinff(float); 
#line 7316
extern int __isnanf(float); 
#line 7326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __finite(double); 
#line 7327
extern int __finitef(float); 
#line 7328
extern int __signbit(double); 
#line 7329
extern int __isnan(double); 
#line 7330
extern int __isinf(double); 
#line 7333 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __signbitf(float); 
#line 7494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern double __cdecl fma(double x, double y, double z); 
#line 7652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl fmaf(float x, float y, float z); 
#line 7661 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __signbitl(long double); 
#line 7667 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern int __finitel(long double); 
#line 7668
extern int __isinfl(long double); 
#line 7669
extern int __isnanl(long double); 
#line 7673 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
__declspec(dllimport) extern float __cdecl acosf(float); 
#line 7674
__declspec(dllimport) extern float __cdecl asinf(float); 
#line 7675
__declspec(dllimport) extern float __cdecl atanf(float); 
#line 7676
__declspec(dllimport) extern float __cdecl atan2f(float, float); 
#line 7677
__declspec(dllimport) extern float __cdecl cosf(float); 
#line 7678
__declspec(dllimport) extern float __cdecl sinf(float); 
#line 7679
__declspec(dllimport) extern float __cdecl tanf(float); 
#line 7680
__declspec(dllimport) extern float __cdecl coshf(float); 
#line 7681
__declspec(dllimport) extern float __cdecl sinhf(float); 
#line 7682
__declspec(dllimport) extern float __cdecl tanhf(float); 
#line 7683
__declspec(dllimport) extern float __cdecl expf(float); 
#line 7684
__declspec(dllimport) extern float __cdecl logf(float); 
#line 7685
__declspec(dllimport) extern float __cdecl log10f(float); 
#line 7686
__declspec(dllimport) extern float __cdecl modff(float, float *); 
#line 7687
__declspec(dllimport) extern float __cdecl powf(float, float); 
#line 7688
__declspec(dllimport) extern float __cdecl sqrtf(float); 
#line 7689
__declspec(dllimport) extern float __cdecl ceilf(float); 
#line 7690
__declspec(dllimport) extern float __cdecl floorf(float); 
#line 7691
__declspec(dllimport) extern float __cdecl fmodf(float, float); 
#line 8823 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
}
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 17
#pragma warning(push)
#pragma warning(disable:4738)
#pragma warning(disable:4820)
#line 26 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
struct _exception { 
#line 28
int type; 
#line 29
char *name; 
#line 30
double arg1; 
#line 31
double arg2; 
#line 32
double retval; 
#line 33
}; 
#line 40
struct _complex { 
#line 42
double x, y; 
#line 43
}; 
#line 52 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
typedef float float_t; 
#line 53
typedef double double_t; 
#line 68
extern const double _HUGE; 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
void __cdecl _fperrraise(int _Except); 
#line 161
__declspec(dllimport) short __cdecl _dclass(double _X); 
#line 162
__declspec(dllimport) short __cdecl _ldclass(long double _X); 
#line 163
__declspec(dllimport) short __cdecl _fdclass(float _X); 
#line 165
__declspec(dllimport) int __cdecl _dsign(double _X); 
#line 166
__declspec(dllimport) int __cdecl _ldsign(long double _X); 
#line 167
__declspec(dllimport) int __cdecl _fdsign(float _X); 
#line 169
__declspec(dllimport) int __cdecl _dpcomp(double _X, double _Y); 
#line 170
__declspec(dllimport) int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 171
__declspec(dllimport) int __cdecl _fdpcomp(float _X, float _Y); 
#line 173
__declspec(dllimport) short __cdecl _dtest(double * _Px); 
#line 174
__declspec(dllimport) short __cdecl _ldtest(long double * _Px); 
#line 175
__declspec(dllimport) short __cdecl _fdtest(float * _Px); 
#line 177
__declspec(dllimport) short __cdecl _d_int(double * _Px, short _Xexp); 
#line 178
__declspec(dllimport) short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 179
__declspec(dllimport) short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 181
__declspec(dllimport) short __cdecl _dscale(double * _Px, long _Lexp); 
#line 182
__declspec(dllimport) short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 183
__declspec(dllimport) short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 185
__declspec(dllimport) short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 186
__declspec(dllimport) short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 187
__declspec(dllimport) short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 189
__declspec(dllimport) short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 190
__declspec(dllimport) short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 191
__declspec(dllimport) short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 193
__declspec(dllimport) short __cdecl _dnorm(unsigned short * _Ps); 
#line 194
__declspec(dllimport) short __cdecl _fdnorm(unsigned short * _Ps); 
#line 196
__declspec(dllimport) double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 197
__declspec(dllimport) long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 198
__declspec(dllimport) float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 200
__declspec(dllimport) double __cdecl _dlog(double _X, int _Baseflag); 
#line 201
__declspec(dllimport) long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 202
__declspec(dllimport) float __cdecl _fdlog(float _X, int _Baseflag); 
#line 204
__declspec(dllimport) double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 205
__declspec(dllimport) long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 206
__declspec(dllimport) float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 213
typedef 
#line 210
union { 
#line 211
unsigned short _Sh[4]; 
#line 212
double _Val; 
#line 213
} _double_val; 
#line 220
typedef 
#line 217
union { 
#line 218
unsigned short _Sh[2]; 
#line 219
float _Val; 
#line 220
} _float_val; 
#line 227
typedef 
#line 224
union { 
#line 225
unsigned short _Sh[4]; 
#line 226
long double _Val; 
#line 227
} _ldouble_val; 
#line 235
typedef 
#line 230
union { 
#line 231
unsigned short _Word[4]; 
#line 232
float _Float; 
#line 233
double _Double; 
#line 234
long double _Long_double; 
#line 235
} _float_const; 
#line 237
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 238
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 239
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 241
extern const _float_const _Eps_C, _Rteps_C; 
#line 242
extern const _float_const _FEps_C, _FRteps_C; 
#line 243
extern const _float_const _LEps_C, _LRteps_C; 
#line 245
extern const double _Zero_C, _Xbig_C; 
#line 246
extern const float _FZero_C, _FXbig_C; 
#line 247
extern const long double _LZero_C, _LXbig_C; 
#line 276
extern "C++" {
#line 278
inline int fpclassify(float _X) throw() 
#line 279
{ 
#line 280
return _fdtest(&_X); 
#line 281
} 
#line 283
inline int fpclassify(double _X) throw() 
#line 284
{ 
#line 285
return _dtest(&_X); 
#line 286
} 
#line 288
inline int fpclassify(long double _X) throw() 
#line 289
{ 
#line 290
return _ldtest(&_X); 
#line 291
} 
#line 293
inline bool signbit(float _X) throw() 
#line 294
{ 
#line 295
return _fdsign(_X) != 0; 
#line 296
} 
#line 298
inline bool signbit(double _X) throw() 
#line 299
{ 
#line 300
return _dsign(_X) != 0; 
#line 301
} 
#line 303
inline bool signbit(long double _X) throw() 
#line 304
{ 
#line 305
return _ldsign(_X) != 0; 
#line 306
} 
#line 308
inline int _fpcomp(float _X, float _Y) throw() 
#line 309
{ 
#line 310
return _fdpcomp(_X, _Y); 
#line 311
} 
#line 313
inline int _fpcomp(double _X, double _Y) throw() 
#line 314
{ 
#line 315
return _dpcomp(_X, _Y); 
#line 316
} 
#line 318
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 319
{ 
#line 320
return _ldpcomp(_X, _Y); 
#line 321
} 
#line 323
template< class _Trc, class _Tre> struct _Combined_type { 
#line 325
typedef float _Type; 
#line 326
}; 
#line 328
template<> struct _Combined_type< float, double>  { 
#line 330
typedef double _Type; 
#line 331
}; 
#line 333
template<> struct _Combined_type< float, long double>  { 
#line 335
typedef long double _Type; 
#line 336
}; 
#line 338
template< class _Ty, class _T2> struct _Real_widened { 
#line 340
typedef long double _Type; 
#line 341
}; 
#line 343
template<> struct _Real_widened< float, float>  { 
#line 345
typedef float _Type; 
#line 346
}; 
#line 348
template<> struct _Real_widened< float, double>  { 
#line 350
typedef double _Type; 
#line 351
}; 
#line 353
template<> struct _Real_widened< double, float>  { 
#line 355
typedef double _Type; 
#line 356
}; 
#line 358
template<> struct _Real_widened< double, double>  { 
#line 360
typedef double _Type; 
#line 361
}; 
#line 363
template< class _Ty> struct _Real_type { 
#line 365
typedef double _Type; 
#line 366
}; 
#line 368
template<> struct _Real_type< float>  { 
#line 370
typedef float _Type; 
#line 371
}; 
#line 373
template<> struct _Real_type< long double>  { 
#line 375
typedef long double _Type; 
#line 376
}; 
#line 378
template< class _T1, class _T2> inline int 
#line 379
_fpcomp(_T1 _X, _T2 _Y) throw() 
#line 380
{ 
#line 384
typedef typename _Combined_type< float, typename _Real_widened< typename _Real_type< _T1> ::_Type, typename _Real_type< _T2> ::_Type> ::_Type> ::_Type _Tw; 
#line 385
return _fpcomp((_Tw)_X, (_Tw)_Y); 
#line 386
} 
#line 388
template< class _Ty> inline bool 
#line 389
isfinite(_Ty _X) throw() 
#line 390
{ 
#line 391
return fpclassify(_X) <= 0; 
#line 392
} 
#line 394
template< class _Ty> inline bool 
#line 395
isinf(_Ty _X) throw() 
#line 396
{ 
#line 397
return fpclassify(_X) == 1; 
#line 398
} 
#line 400
template< class _Ty> inline bool 
#line 401
isnan(_Ty _X) throw() 
#line 402
{ 
#line 403
return fpclassify(_X) == 2; 
#line 404
} 
#line 406
template< class _Ty> inline bool 
#line 407
isnormal(_Ty _X) throw() 
#line 408
{ 
#line 409
return fpclassify(_X) == (-1); 
#line 410
} 
#line 412
template< class _Ty1, class _Ty2> inline bool 
#line 413
isgreater(_Ty1 _X, _Ty2 _Y) throw() 
#line 414
{ 
#line 415
return (_fpcomp(_X, _Y) & 4) != 0; 
#line 416
} 
#line 418
template< class _Ty1, class _Ty2> inline bool 
#line 419
isgreaterequal(_Ty1 _X, _Ty2 _Y) throw() 
#line 420
{ 
#line 421
return (_fpcomp(_X, _Y) & (2 | 4)) != 0; 
#line 422
} 
#line 424
template< class _Ty1, class _Ty2> inline bool 
#line 425
isless(_Ty1 _X, _Ty2 _Y) throw() 
#line 426
{ 
#line 427
return (_fpcomp(_X, _Y) & 1) != 0; 
#line 428
} 
#line 430
template< class _Ty1, class _Ty2> inline bool 
#line 431
islessequal(_Ty1 _X, _Ty2 _Y) throw() 
#line 432
{ 
#line 433
return (_fpcomp(_X, _Y) & (1 | 2)) != 0; 
#line 434
} 
#line 436
template< class _Ty1, class _Ty2> inline bool 
#line 437
islessgreater(_Ty1 _X, _Ty2 _Y) throw() 
#line 438
{ 
#line 439
return (_fpcomp(_X, _Y) & (1 | 4)) != 0; 
#line 440
} 
#line 442
template< class _Ty1, class _Ty2> inline bool 
#line 443
isunordered(_Ty1 _X, _Ty2 _Y) throw() 
#line 444
{ 
#line 445
return _fpcomp(_X, _Y) == 0; 
#line 446
} 
#line 447
}
#line 454 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
int __cdecl abs(int _X); 
#line 455
long __cdecl labs(long _X); 
#line 456
__int64 __cdecl llabs(__int64 _X); 
#line 458
double __cdecl acos(double _X); 
#line 459
double __cdecl asin(double _X); 
#line 460
double __cdecl atan(double _X); 
#line 461
double __cdecl atan2(double _Y, double _X); 
#line 463
double __cdecl cos(double _X); 
#line 464
double __cdecl cosh(double _X); 
#line 465
double __cdecl exp(double _X); 
#line 466
double __cdecl fabs(double _X); 
#line 467
double __cdecl fmod(double _X, double _Y); 
#line 468
double __cdecl log(double _X); 
#line 469
double __cdecl log10(double _X); 
#line 470
double __cdecl pow(double _X, double _Y); 
#line 471
double __cdecl sin(double _X); 
#line 472
double __cdecl sinh(double _X); 
#line 473
double __cdecl sqrt(double _X); 
#line 474
double __cdecl tan(double _X); 
#line 475
double __cdecl tanh(double _X); 
#line 477
__declspec(dllimport) double __cdecl acosh(double _X); 
#line 478
__declspec(dllimport) double __cdecl asinh(double _X); 
#line 479
__declspec(dllimport) double __cdecl atanh(double _X); 
#line 480
__declspec(dllimport) double __cdecl atof(const char * _String); 
#line 481
__declspec(dllimport) double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 482
__declspec(dllimport) double __cdecl _cabs(_complex _Complex_value); 
#line 483
__declspec(dllimport) double __cdecl cbrt(double _X); 
#line 484
__declspec(dllimport) double __cdecl ceil(double _X); 
#line 485
__declspec(dllimport) double __cdecl _chgsign(double _X); 
#line 486
__declspec(dllimport) double __cdecl copysign(double _Number, double _Sign); 
#line 487
__declspec(dllimport) double __cdecl _copysign(double _Number, double _Sign); 
#line 488
__declspec(dllimport) double __cdecl erf(double _X); 
#line 489
__declspec(dllimport) double __cdecl erfc(double _X); 
#line 490
__declspec(dllimport) double __cdecl exp2(double _X); 
#line 491
__declspec(dllimport) double __cdecl expm1(double _X); 
#line 492
__declspec(dllimport) double __cdecl fdim(double _X, double _Y); 
#line 493
__declspec(dllimport) double __cdecl floor(double _X); 
#line 494
__declspec(dllimport) double __cdecl fma(double _X, double _Y, double _Z); 
#line 495
__declspec(dllimport) double __cdecl fmax(double _X, double _Y); 
#line 496
__declspec(dllimport) double __cdecl fmin(double _X, double _Y); 
#line 497
__declspec(dllimport) double __cdecl frexp(double _X, int * _Y); 
#line 498
__declspec(dllimport) double __cdecl hypot(double _X, double _Y); 
#line 499
__declspec(dllimport) double __cdecl _hypot(double _X, double _Y); 
#line 500
__declspec(dllimport) int __cdecl ilogb(double _X); 
#line 501
__declspec(dllimport) double __cdecl ldexp(double _X, int _Y); 
#line 502
__declspec(dllimport) double __cdecl lgamma(double _X); 
#line 503
__declspec(dllimport) __int64 __cdecl llrint(double _X); 
#line 504
__declspec(dllimport) __int64 __cdecl llround(double _X); 
#line 505
__declspec(dllimport) double __cdecl log1p(double _X); 
#line 506
__declspec(dllimport) double __cdecl log2(double _X); 
#line 507
__declspec(dllimport) double __cdecl logb(double _X); 
#line 508
__declspec(dllimport) long __cdecl lrint(double _X); 
#line 509
__declspec(dllimport) long __cdecl lround(double _X); 
#line 511
int __cdecl _matherr(_exception * _Except); 
#line 513
__declspec(dllimport) double __cdecl modf(double _X, double * _Y); 
#line 514
__declspec(dllimport) double __cdecl nan(const char *); 
#line 515
__declspec(dllimport) double __cdecl nearbyint(double _X); 
#line 516
__declspec(dllimport) double __cdecl nextafter(double _X, double _Y); 
#line 517
__declspec(dllimport) double __cdecl nexttoward(double _X, long double _Y); 
#line 518
__declspec(dllimport) double __cdecl remainder(double _X, double _Y); 
#line 519
__declspec(dllimport) double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 520
__declspec(dllimport) double __cdecl rint(double _X); 
#line 521
__declspec(dllimport) double __cdecl round(double _X); 
#line 522
__declspec(dllimport) double __cdecl scalbln(double _X, long _Y); 
#line 523
__declspec(dllimport) double __cdecl scalbn(double _X, int _Y); 
#line 524
__declspec(dllimport) double __cdecl tgamma(double _X); 
#line 525
__declspec(dllimport) double __cdecl trunc(double _X); 
#line 526
__declspec(dllimport) double __cdecl _j0(double _X); 
#line 527
__declspec(dllimport) double __cdecl _j1(double _X); 
#line 528
__declspec(dllimport) double __cdecl _jn(int _X, double _Y); 
#line 529
__declspec(dllimport) double __cdecl _y0(double _X); 
#line 530
__declspec(dllimport) double __cdecl _y1(double _X); 
#line 531
__declspec(dllimport) double __cdecl _yn(int _X, double _Y); 
#line 533
__declspec(dllimport) float __cdecl acoshf(float _X); 
#line 534
__declspec(dllimport) float __cdecl asinhf(float _X); 
#line 535
__declspec(dllimport) float __cdecl atanhf(float _X); 
#line 536
__declspec(dllimport) float __cdecl cbrtf(float _X); 
#line 537
__declspec(dllimport) float __cdecl _chgsignf(float _X); 
#line 538
__declspec(dllimport) float __cdecl copysignf(float _Number, float _Sign); 
#line 539
__declspec(dllimport) float __cdecl _copysignf(float _Number, float _Sign); 
#line 540
__declspec(dllimport) float __cdecl erff(float _X); 
#line 541
__declspec(dllimport) float __cdecl erfcf(float _X); 
#line 542
__declspec(dllimport) float __cdecl expm1f(float _X); 
#line 543
__declspec(dllimport) float __cdecl exp2f(float _X); 
#line 544
__declspec(dllimport) float __cdecl fdimf(float _X, float _Y); 
#line 545
__declspec(dllimport) float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 546
__declspec(dllimport) float __cdecl fmaxf(float _X, float _Y); 
#line 547
__declspec(dllimport) float __cdecl fminf(float _X, float _Y); 
#line 548
__declspec(dllimport) float __cdecl _hypotf(float _X, float _Y); 
#line 549
__declspec(dllimport) int __cdecl ilogbf(float _X); 
#line 550
__declspec(dllimport) float __cdecl lgammaf(float _X); 
#line 551
__declspec(dllimport) __int64 __cdecl llrintf(float _X); 
#line 552
__declspec(dllimport) __int64 __cdecl llroundf(float _X); 
#line 553
__declspec(dllimport) float __cdecl log1pf(float _X); 
#line 554
__declspec(dllimport) float __cdecl log2f(float _X); 
#line 555
__declspec(dllimport) float __cdecl logbf(float _X); 
#line 556
__declspec(dllimport) long __cdecl lrintf(float _X); 
#line 557
__declspec(dllimport) long __cdecl lroundf(float _X); 
#line 558
__declspec(dllimport) float __cdecl nanf(const char *); 
#line 559
__declspec(dllimport) float __cdecl nearbyintf(float _X); 
#line 560
__declspec(dllimport) float __cdecl nextafterf(float _X, float _Y); 
#line 561
__declspec(dllimport) float __cdecl nexttowardf(float _X, long double _Y); 
#line 562
__declspec(dllimport) float __cdecl remainderf(float _X, float _Y); 
#line 563
__declspec(dllimport) float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 564
__declspec(dllimport) float __cdecl rintf(float _X); 
#line 565
__declspec(dllimport) float __cdecl roundf(float _X); 
#line 566
__declspec(dllimport) float __cdecl scalblnf(float _X, long _Y); 
#line 567
__declspec(dllimport) float __cdecl scalbnf(float _X, int _Y); 
#line 568
__declspec(dllimport) float __cdecl tgammaf(float _X); 
#line 569
__declspec(dllimport) float __cdecl truncf(float _X); 
#line 577 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__declspec(dllimport) float __cdecl _logbf(float _X); 
#line 578
__declspec(dllimport) float __cdecl _nextafterf(float _X, float _Y); 
#line 579
__declspec(dllimport) int __cdecl _finitef(float _X); 
#line 580
__declspec(dllimport) int __cdecl _isnanf(float _X); 
#line 581
__declspec(dllimport) int __cdecl _fpclassf(float _X); 
#line 583
__declspec(dllimport) int __cdecl _set_FMA3_enable(int _Flag); 
#line 584
__declspec(dllimport) int __cdecl _get_FMA3_enable(); 
#line 597 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__declspec(dllimport) float __cdecl acosf(float _X); 
#line 598
__declspec(dllimport) float __cdecl asinf(float _X); 
#line 599
__declspec(dllimport) float __cdecl atan2f(float _Y, float _X); 
#line 600
__declspec(dllimport) float __cdecl atanf(float _X); 
#line 601
__declspec(dllimport) float __cdecl ceilf(float _X); 
#line 602
__declspec(dllimport) float __cdecl cosf(float _X); 
#line 603
__declspec(dllimport) float __cdecl coshf(float _X); 
#line 604
__declspec(dllimport) float __cdecl expf(float _X); 
#line 656 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__inline float __cdecl fabsf(float _X) 
#line 657
{ 
#line 658
return (float)fabs(_X); 
#line 659
} 
#line 665 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__declspec(dllimport) float __cdecl floorf(float _X); 
#line 666
__declspec(dllimport) float __cdecl fmodf(float _X, float _Y); 
#line 682 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 683
{ 
#line 684
return (float)frexp(_X, _Y); 
#line 685
} 
#line 687
__inline float __cdecl hypotf(float _X, float _Y) 
#line 688
{ 
#line 689
return _hypotf(_X, _Y); 
#line 690
} 
#line 692
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 693
{ 
#line 694
return (float)ldexp(_X, _Y); 
#line 695
} 
#line 699
__declspec(dllimport) float __cdecl log10f(float _X); 
#line 700
__declspec(dllimport) float __cdecl logf(float _X); 
#line 701
__declspec(dllimport) float __cdecl modff(float _X, float * _Y); 
#line 702
__declspec(dllimport) float __cdecl powf(float _X, float _Y); 
#line 703
__declspec(dllimport) float __cdecl sinf(float _X); 
#line 704
__declspec(dllimport) float __cdecl sinhf(float _X); 
#line 705
__declspec(dllimport) float __cdecl sqrtf(float _X); 
#line 706
__declspec(dllimport) float __cdecl tanf(float _X); 
#line 707
__declspec(dllimport) float __cdecl tanhf(float _X); 
#line 761 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__declspec(dllimport) long double __cdecl acoshl(long double _X); 
#line 763
__inline long double __cdecl acosl(long double _X) 
#line 764
{ 
#line 765
return acos((double)_X); 
#line 766
} 
#line 768
__declspec(dllimport) long double __cdecl asinhl(long double _X); 
#line 770
__inline long double __cdecl asinl(long double _X) 
#line 771
{ 
#line 772
return asin((double)_X); 
#line 773
} 
#line 775
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 776
{ 
#line 777
return atan2((double)_Y, (double)_X); 
#line 778
} 
#line 780
__declspec(dllimport) long double __cdecl atanhl(long double _X); 
#line 782
__inline long double __cdecl atanl(long double _X) 
#line 783
{ 
#line 784
return atan((double)_X); 
#line 785
} 
#line 787
__declspec(dllimport) long double __cdecl cbrtl(long double _X); 
#line 789
__inline long double __cdecl ceill(long double _X) 
#line 790
{ 
#line 791
return ceil((double)_X); 
#line 792
} 
#line 794
__inline long double __cdecl _chgsignl(long double _X) 
#line 795
{ 
#line 796
return _chgsign((double)_X); 
#line 797
} 
#line 799
__declspec(dllimport) long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 801
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 802
{ 
#line 803
return _copysign((double)_Number, (double)_Sign); 
#line 804
} 
#line 806
__inline long double __cdecl coshl(long double _X) 
#line 807
{ 
#line 808
return cosh((double)_X); 
#line 809
} 
#line 811
__inline long double __cdecl cosl(long double _X) 
#line 812
{ 
#line 813
return cos((double)_X); 
#line 814
} 
#line 816
__declspec(dllimport) long double __cdecl erfl(long double _X); 
#line 817
__declspec(dllimport) long double __cdecl erfcl(long double _X); 
#line 819
__inline long double __cdecl expl(long double _X) 
#line 820
{ 
#line 821
return exp((double)_X); 
#line 822
} 
#line 824
__declspec(dllimport) long double __cdecl exp2l(long double _X); 
#line 825
__declspec(dllimport) long double __cdecl expm1l(long double _X); 
#line 827
__inline long double __cdecl fabsl(long double _X) 
#line 828
{ 
#line 829
return fabs((double)_X); 
#line 830
} 
#line 832
__declspec(dllimport) long double __cdecl fdiml(long double _X, long double _Y); 
#line 834
__inline long double __cdecl floorl(long double _X) 
#line 835
{ 
#line 836
return floor((double)_X); 
#line 837
} 
#line 839
__declspec(dllimport) long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 840
__declspec(dllimport) long double __cdecl fmaxl(long double _X, long double _Y); 
#line 841
__declspec(dllimport) long double __cdecl fminl(long double _X, long double _Y); 
#line 843
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 844
{ 
#line 845
return fmod((double)_X, (double)_Y); 
#line 846
} 
#line 848
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 849
{ 
#line 850
return frexp((double)_X, _Y); 
#line 851
} 
#line 853
__declspec(dllimport) int __cdecl ilogbl(long double _X); 
#line 855
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 856
{ 
#line 857
return _hypot((double)_X, (double)_Y); 
#line 858
} 
#line 860
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 861
{ 
#line 862
return _hypot((double)_X, (double)_Y); 
#line 863
} 
#line 865
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 866
{ 
#line 867
return ldexp((double)_X, _Y); 
#line 868
} 
#line 870
__declspec(dllimport) long double __cdecl lgammal(long double _X); 
#line 871
__declspec(dllimport) __int64 __cdecl llrintl(long double _X); 
#line 872
__declspec(dllimport) __int64 __cdecl llroundl(long double _X); 
#line 874
__inline long double __cdecl logl(long double _X) 
#line 875
{ 
#line 876
return log((double)_X); 
#line 877
} 
#line 879
__inline long double __cdecl log10l(long double _X) 
#line 880
{ 
#line 881
return log10((double)_X); 
#line 882
} 
#line 884
__declspec(dllimport) long double __cdecl log1pl(long double _X); 
#line 885
__declspec(dllimport) long double __cdecl log2l(long double _X); 
#line 886
__declspec(dllimport) long double __cdecl logbl(long double _X); 
#line 887
__declspec(dllimport) long __cdecl lrintl(long double _X); 
#line 888
__declspec(dllimport) long __cdecl lroundl(long double _X); 
#line 890
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 891
{ 
#line 892
double _F, _I; 
#line 893
_F = modf((double)_X, &_I); 
#line 894
(*_Y) = _I; 
#line 895
return _F; 
#line 896
} 
#line 898
__declspec(dllimport) long double __cdecl nanl(const char *); 
#line 899
__declspec(dllimport) long double __cdecl nearbyintl(long double _X); 
#line 900
__declspec(dllimport) long double __cdecl nextafterl(long double _X, long double _Y); 
#line 901
__declspec(dllimport) long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 903
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 904
{ 
#line 905
return pow((double)_X, (double)_Y); 
#line 906
} 
#line 908
__declspec(dllimport) long double __cdecl remainderl(long double _X, long double _Y); 
#line 909
__declspec(dllimport) long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 910
__declspec(dllimport) long double __cdecl rintl(long double _X); 
#line 911
__declspec(dllimport) long double __cdecl roundl(long double _X); 
#line 912
__declspec(dllimport) long double __cdecl scalblnl(long double _X, long _Y); 
#line 913
__declspec(dllimport) long double __cdecl scalbnl(long double _X, int _Y); 
#line 915
__inline long double __cdecl sinhl(long double _X) 
#line 916
{ 
#line 917
return sinh((double)_X); 
#line 918
} 
#line 920
__inline long double __cdecl sinl(long double _X) 
#line 921
{ 
#line 922
return sin((double)_X); 
#line 923
} 
#line 925
__inline long double __cdecl sqrtl(long double _X) 
#line 926
{ 
#line 927
return sqrt((double)_X); 
#line 928
} 
#line 930
__inline long double __cdecl tanhl(long double _X) 
#line 931
{ 
#line 932
return tanh((double)_X); 
#line 933
} 
#line 935
__inline long double __cdecl tanl(long double _X) 
#line 936
{ 
#line 937
return tan((double)_X); 
#line 938
} 
#line 940
__declspec(dllimport) long double __cdecl tgammal(long double _X); 
#line 941
__declspec(dllimport) long double __cdecl truncl(long double _X); 
#line 963 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
extern double HUGE; 
#line 968 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__declspec(dllimport) double __cdecl j0(double _X); 
#line 969
__declspec(dllimport) double __cdecl j1(double _X); 
#line 970
__declspec(dllimport) double __cdecl jn(int _X, double _Y); 
#line 971
__declspec(dllimport) double __cdecl y0(double _X); 
#line 972
__declspec(dllimport) double __cdecl y1(double _X); 
#line 973
__declspec(dllimport) double __cdecl yn(int _X, double _Y); 
#line 983 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
}
#line 979 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
#pragma warning(pop)
#line 983 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\math.h"
__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_malloc.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 54 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_malloc.h"
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 55
_calloc_base(size_t _Count, size_t _Size); 
#line 61
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 62
calloc(size_t _Count, size_t _Size); 
#line 68
__declspec(dllimport) int __cdecl _callnewh(size_t _Size); 
#line 73
__declspec(dllimport) __declspec(allocator) void *__cdecl 
#line 74
_expand(void * _Block, size_t _Size); 
#line 79
__declspec(dllimport) void __cdecl 
#line 80
_free_base(void * _Block); 
#line 84
__declspec(dllimport) void __cdecl 
#line 85
free(void * _Block); 
#line 90
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 91
_malloc_base(size_t _Size); 
#line 96
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 97
malloc(size_t _Size); 
#line 102
__declspec(dllimport) size_t __cdecl 
#line 103
_msize(void * _Block); 
#line 108
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 109
_realloc_base(void * _Block, size_t _Size); 
#line 115
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 116
realloc(void * _Block, size_t _Size); 
#line 122
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 123
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 129
__declspec(dllimport) void __cdecl 
#line 130
_aligned_free(void * _Block); 
#line 135
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 136
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 142
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 143
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 150
__declspec(dllimport) size_t __cdecl 
#line 151
_aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 158
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 159
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 167
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 168
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 177
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 178
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 185
__declspec(dllimport) __declspec(allocator) __declspec(restrict) void *__cdecl 
#line 186
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 214 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_search.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 23
__declspec(dllimport) void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context); 
#line 32
__declspec(dllimport) void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context); 
#line 45 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_search.h"
__declspec(dllimport) void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 53
__declspec(dllimport) void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 61
__declspec(dllimport) void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context); 
#line 71
__declspec(dllimport) void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 80
__declspec(dllimport) void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context); 
#line 90
__declspec(dllimport) void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 185 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_search.h"
__declspec(dllimport) void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 194
__declspec(dllimport) void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, int (__cdecl * _PtFuncCompare)(const void *, const void *)); 
#line 206 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 48
__declspec(dllimport) errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 55
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _itow_s(int _Value, __wchar_t (&_Buffer)[_Size], int _Radix) throw() { return _itow_s(_Value, _Buffer, _Size, _Radix); } }
#line 62 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) __wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 71 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 78
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ltow_s(long _Value, __wchar_t (&_Buffer)[_Size], int _Radix) throw() { return _ltow_s(_Value, _Buffer, _Size, _Radix); } }
#line 85 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) __wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 100
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t (&_Buffer)[_Size], int _Radix) throw() { return _ultow_s(_Value, _Buffer, _Size, _Radix); } }
#line 107 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) __wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 115 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 121
__declspec(dllimport) double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 128
__declspec(dllimport) long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 135
__declspec(dllimport) long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 143
__declspec(dllimport) __int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 150
__declspec(dllimport) __int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 158
__declspec(dllimport) unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 165
__declspec(dllimport) unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 173
__declspec(dllimport) unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 180
__declspec(dllimport) unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 188
__declspec(dllimport) long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 194
__declspec(dllimport) long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 201
__declspec(dllimport) float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 207
__declspec(dllimport) float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 214
__declspec(dllimport) double __cdecl _wtof(const __wchar_t * _String); 
#line 219
__declspec(dllimport) double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 225
__declspec(dllimport) int __cdecl _wtoi(const __wchar_t * _String); 
#line 230
__declspec(dllimport) int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 236
__declspec(dllimport) long __cdecl _wtol(const __wchar_t * _String); 
#line 241
__declspec(dllimport) long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 247
__declspec(dllimport) __int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 252
__declspec(dllimport) __int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 258
__declspec(dllimport) errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 266
__declspec(dllimport) __wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 273
__declspec(dllimport) errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 281
__declspec(dllimport) __wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 288
__declspec(dllimport) __int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 293
__declspec(dllimport) __int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 299
__declspec(dllimport) __int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 306
__declspec(dllimport) __int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 314
__declspec(dllimport) unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 321
__declspec(dllimport) unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 333
__declspec(dllimport) __declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 342
__declspec(dllimport) errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 351
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wmakepath_s(__wchar_t (&_Buffer)[_Size], const __wchar_t *_Drive, const __wchar_t *_Dir, const __wchar_t *_Filename, const __wchar_t *_Ext) throw() { return _wmakepath_s(_Buffer, _Size, _Drive, _Dir, _Filename, _Ext); } }
#line 360 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 369 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) void __cdecl _wperror(const __wchar_t * _ErrMsg); 
#line 374
__declspec(dllimport) void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 382
__declspec(dllimport) errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 394
extern "C++" {template< size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize> inline ::errno_t __cdecl _wsplitpath_s(const __wchar_t *_Path, __wchar_t (&_Drive)[_DriveSize], __wchar_t (&_Dir)[_DirSize], __wchar_t (&_Name)[_NameSize], __wchar_t (&_Ext)[_ExtSize]) throw() { return _wsplitpath_s(_Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize); } }
#line 407 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 416
__declspec(dllimport) __wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 422
__declspec(dllimport) errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 429
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wgetenv_s(::size_t *_RequiredCount, __wchar_t (&_Buffer)[_Size], const __wchar_t *_VarName) throw() { return _wgetenv_s(_RequiredCount, _Buffer, _Size, _VarName); } }
#line 438 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 443
__declspec(dllimport) errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 448
__declspec(dllimport) errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 455
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wsearchenv_s(const __wchar_t *_Filename, const __wchar_t *_VarName, __wchar_t (&_ResultPath)[_Size]) throw() { return _wsearchenv_s(_Filename, _VarName, _ResultPath, _Size); } }
#line 462 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 469 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
__declspec(dllimport) int __cdecl _wsystem(const __wchar_t * _Command); 
#line 477 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 17 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 33 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 50
__declspec(dllimport) __declspec(noreturn) void __cdecl exit(int _Code); 
#line 51
__declspec(dllimport) __declspec(noreturn) void __cdecl _exit(int _Code); 
#line 52
__declspec(dllimport) __declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 53
__declspec(dllimport) __declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 54
__declspec(dllimport) __declspec(noreturn) void __cdecl abort(); 
#line 60
__declspec(dllimport) unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 68
typedef int (__cdecl *_onexit_t)(void); 
#line 139 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 140
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 143 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 154
typedef void (__cdecl *_purecall_handler)(void); 
#line 157
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 166
__declspec(dllimport) _purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 170
__declspec(dllimport) _purecall_handler __cdecl _get_purecall_handler(); 
#line 173
__declspec(dllimport) _invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 177
__declspec(dllimport) _invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 179
__declspec(dllimport) _invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 183
__declspec(dllimport) _invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 207 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int __cdecl _set_error_mode(int _Mode); 
#line 211
__declspec(dllimport) int *__cdecl _errno(); 
#line 214
__declspec(dllimport) errno_t __cdecl _set_errno(int _Value); 
#line 215
__declspec(dllimport) errno_t __cdecl _get_errno(int * _Value); 
#line 217
__declspec(dllimport) unsigned long *__cdecl __doserrno(); 
#line 220
__declspec(dllimport) errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 221
__declspec(dllimport) errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 224
__declspec(dllimport) char **__cdecl __sys_errlist(); 
#line 227
__declspec(dllimport) int *__cdecl __sys_nerr(); 
#line 230
__declspec(dllimport) void __cdecl perror(const char * _ErrMsg); 
#line 235
__declspec(dllimport) char **__cdecl __p__pgmptr(); 
#line 236
__declspec(dllimport) __wchar_t **__cdecl __p__wpgmptr(); 
#line 237
__declspec(dllimport) int *__cdecl __p__fmode(); 
#line 250 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 253
__declspec(dllimport) errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 255
__declspec(dllimport) errno_t __cdecl _set_fmode(int _Mode); 
#line 257
__declspec(dllimport) errno_t __cdecl _get_fmode(int * _PMode); 
#line 270
typedef 
#line 266
struct _div_t { 
#line 268
int quot; 
#line 269
int rem; 
#line 270
} div_t; 
#line 276
typedef 
#line 272
struct _ldiv_t { 
#line 274
long quot; 
#line 275
long rem; 
#line 276
} ldiv_t; 
#line 282
typedef 
#line 278
struct _lldiv_t { 
#line 280
__int64 quot; 
#line 281
__int64 rem; 
#line 282
} lldiv_t; 
#line 284
int __cdecl abs(int _Number); 
#line 285
long __cdecl labs(long _Number); 
#line 286
__int64 __cdecl llabs(__int64 _Number); 
#line 287
__int64 __cdecl _abs64(__int64 _Number); 
#line 289
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 290
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 291
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 293
__declspec(dllimport) div_t __cdecl div(int _Numerator, int _Denominator); 
#line 294
__declspec(dllimport) ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 295
__declspec(dllimport) lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 299
#pragma warning (push)
#pragma warning (disable:6540)
#line 302
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 308
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 313
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 318
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 324
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 329
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 334
#pragma warning (pop)
#line 341
__declspec(dllimport) void __cdecl srand(unsigned _Seed); 
#line 343
__declspec(dllimport) int __cdecl rand(); 
#line 352 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
extern "C++" {
#line 354
inline long abs(const long _X) throw() 
#line 355
{ 
#line 356
return labs(_X); 
#line 357
} 
#line 359
inline __int64 abs(const __int64 _X) throw() 
#line 360
{ 
#line 361
return llabs(_X); 
#line 362
} 
#line 364
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 365
{ 
#line 366
return ldiv(_A1, _A2); 
#line 367
} 
#line 369
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 370
{ 
#line 371
return lldiv(_A1, _A2); 
#line 372
} 
#line 373
}
#line 385 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 389
typedef 
#line 387
struct { 
#line 388
unsigned char ld[10]; 
#line 389
} _LDOUBLE; 
#pragma pack ( pop )
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
typedef 
#line 407
struct { 
#line 408
double x; 
#line 409
} _CRT_DOUBLE; 
#line 414
typedef 
#line 412
struct { 
#line 413
float f; 
#line 414
} _CRT_FLOAT; 
#line 423
typedef 
#line 421
struct { 
#line 422
long double x; 
#line 423
} _LONGDOUBLE; 
#line 427
#pragma pack ( push, 4 )
#line 431
typedef 
#line 429
struct { 
#line 430
unsigned char ld12[12]; 
#line 431
} _LDBL12; 
#pragma pack ( pop )
#line 441
__declspec(dllimport) double __cdecl atof(const char * _String); 
#line 442
__declspec(dllimport) int __cdecl atoi(const char * _String); 
#line 443
__declspec(dllimport) long __cdecl atol(const char * _String); 
#line 444
__declspec(dllimport) __int64 __cdecl atoll(const char * _String); 
#line 445
__declspec(dllimport) __int64 __cdecl _atoi64(const char * _String); 
#line 447
__declspec(dllimport) double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 448
__declspec(dllimport) int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 449
__declspec(dllimport) long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 450
__declspec(dllimport) __int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 451
__declspec(dllimport) __int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 453
__declspec(dllimport) int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 454
__declspec(dllimport) int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 455
__declspec(dllimport) int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 458
__declspec(dllimport) int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 465
__declspec(dllimport) int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 473
__declspec(dllimport) int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 480
__declspec(dllimport) float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 486
__declspec(dllimport) float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 493
__declspec(dllimport) double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 499
__declspec(dllimport) double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 506
__declspec(dllimport) long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 512
__declspec(dllimport) long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 519
__declspec(dllimport) long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 526
__declspec(dllimport) long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 534
__declspec(dllimport) __int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 541
__declspec(dllimport) __int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 549
__declspec(dllimport) unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 556
__declspec(dllimport) unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 564
__declspec(dllimport) unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 571
__declspec(dllimport) unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 579
__declspec(dllimport) __int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 586
__declspec(dllimport) __int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 594
__declspec(dllimport) unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 601
__declspec(dllimport) unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 617
__declspec(dllimport) errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 624
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _itoa_s(int _Value, char (&_Buffer)[_Size], int _Radix) throw() { return _itoa_s(_Value, _Buffer, _Size, _Radix); } }
#line 632 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 648
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ltoa_s(long _Value, char (&_Buffer)[_Size], int _Radix) throw() { return _ltoa_s(_Value, _Buffer, _Size, _Radix); } }
#line 655 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 664 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 671
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ultoa_s(unsigned long _Value, char (&_Buffer)[_Size], int _Radix) throw() { return _ultoa_s(_Value, _Buffer, _Size, _Radix); } }
#line 678 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 696
__declspec(dllimport) char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 704
__declspec(dllimport) errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 712
__declspec(dllimport) char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 732
__declspec(dllimport) errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 741
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _ecvt_s(char (&_Buffer)[_Size], double _Value, int _DigitCount, int *_PtDec, int *_PtSign) throw() { return _ecvt_s(_Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign); } }
#line 751 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 760
__declspec(dllimport) errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 769
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _fcvt_s(char (&_Buffer)[_Size], double _Value, int _FractionalDigitCount, int *_PtDec, int *_PtSign) throw() { return _fcvt_s(_Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign); } }
#line 781 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 789
__declspec(dllimport) errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 796
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _gcvt_s(char (&_Buffer)[_Size], double _Value, int _DigitCount) throw() { return _gcvt_s(_Buffer, _Size, _Value, _DigitCount); } }
#line 805 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 834 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int __cdecl ___mb_cur_max_func(); 
#line 837
__declspec(dllimport) int __cdecl ___mb_cur_max_l_func(_locale_t); 
#line 843 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 849
__declspec(dllimport) int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 857
__declspec(dllimport) size_t __cdecl _mbstrlen(const char * _String); 
#line 863
__declspec(dllimport) size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 870
__declspec(dllimport) size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 877
__declspec(dllimport) size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 884
__declspec(dllimport) int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 891
__declspec(dllimport) int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 899
__declspec(dllimport) errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 907
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl mbstowcs_s(::size_t *_PtNumOfCharConverted, __wchar_t (&_Dest)[_Size], const char *_Source, ::size_t _MaxCount) throw() { return mbstowcs_s(_PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount); } }
#line 915 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 923 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 932
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _mbstowcs_s_l(::size_t *_PtNumOfCharConverted, __wchar_t (&_Dest)[_Size], const char *_Source, ::size_t _MaxCount, ::_locale_t _Locale) throw() { return _mbstowcs_s_l(_PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale); } }
#line 941 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 954 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 960
__declspec(dllimport) int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 969
__declspec(dllimport) errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 979 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 987
__declspec(dllimport) errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 995
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl wcstombs_s(::size_t *_PtNumOfCharConverted, char (&_Dest)[_Size], const __wchar_t *_Source, ::size_t _MaxCount) throw() { return wcstombs_s(_PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount); } }
#line 1003 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1011 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1020
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wcstombs_s_l(::size_t *_PtNumOfCharConverted, char (&_Dest)[_Size], const __wchar_t *_Source, ::size_t _MaxCount, ::_locale_t _Locale) throw() { return _wcstombs_s_l(_PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale); } }
#line 1029 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1059 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) __declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1068
__declspec(dllimport) errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1077
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _makepath_s(char (&_Buffer)[_Size], const char *_Drive, const char *_Dir, const char *_Filename, const char *_Ext) throw() { return _makepath_s(_Buffer, _Size, _Drive, _Dir, _Filename, _Ext); } }
#line 1086 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1096 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1105
__declspec(dllimport) errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1117
extern "C++" {template< size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize> inline ::errno_t __cdecl _splitpath_s(const char *_Dest, char (&_Drive)[_DriveSize], char (&_Dir)[_DirSize], char (&_Name)[_NameSize], char (&_Ext)[_ExtSize]) throw() { return _splitpath_s(_Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize); } }
#line 1132
__declspec(dllimport) errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1144 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int *__cdecl __p___argc(); 
#line 1145
__declspec(dllimport) char ***__cdecl __p___argv(); 
#line 1146
__declspec(dllimport) __wchar_t ***__cdecl __p___wargv(); 
#line 1158 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char ***__cdecl __p__environ(); 
#line 1159
__declspec(dllimport) __wchar_t ***__cdecl __p__wenviron(); 
#line 1183 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) char *__cdecl getenv(const char * _VarName); 
#line 1187
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl getenv_s(::size_t *_RequiredCount, char (&_Buffer)[_Size], const char *_VarName) throw() { return getenv_s(_RequiredCount, _Buffer, _Size, _VarName); } }
#line 1200 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1210 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) int __cdecl system(const char * _Command); 
#line 1216
#pragma warning (push)
#pragma warning (disable:6540)
#line 1220
__declspec(dllimport) int __cdecl _putenv(const char * _EnvString); 
#line 1225
__declspec(dllimport) errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1230
#pragma warning (pop)
#line 1232
__declspec(dllimport) errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1239
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _searchenv_s(const char *_Filename, const char *_VarName, char (&_Buffer)[_Size]) throw() { return _searchenv_s(_Filename, _VarName, _Buffer, _Size); } }
#line 1246 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(dllimport) void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1254 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) 
#line 1255
__declspec(dllimport) void __cdecl _seterrormode(int _Mode); 
#line 1259
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) 
#line 1260
__declspec(dllimport) void __cdecl _beep(unsigned _Frequency, unsigned _Duration); 
#line 1265
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) 
#line 1266
__declspec(dllimport) void __cdecl _sleep(unsigned long _Duration); 
#line 1289 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1293
__declspec(dllimport) char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1301
__declspec(dllimport) char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1309
__declspec(dllimport) char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1316
__declspec(dllimport) char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1323
__declspec(dllimport) char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1331
__declspec(dllimport) void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1338
__declspec(dllimport) char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1349
__declspec(dllimport) int __cdecl putenv(const char * _EnvString); 
#line 1355 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 1357
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1363 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
}
#line 253 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
#pragma detect_mismatch("_MSC_VER", "1900")
#line 257 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
#pragma detect_mismatch("_ITERATOR_DEBUG_LEVEL", "2")
#line 268 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
#pragma detect_mismatch("RuntimeLibrary", "MDd_DynamicDebug")
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\use_ansi.h"
#pragma comment(lib, "msvcprtd")
#line 1363 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdlib.h"
__pragma( pack ( pop )) 
#line 10 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
#pragma pack ( push, 8 )
#line 592 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
typedef __int64 _Longlong; 
#line 593
typedef unsigned __int64 _ULonglong; 
#line 611
namespace std { 
#line 612
enum _Uninitialized { 
#line 614
_Noinit
#line 615
}; 
#line 619
#pragma warning(push)
#pragma warning(disable:4412)
class __declspec(dllimport) _Lockit { 
#line 644 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
public: __thiscall _Lockit(); 
#line 645
explicit __thiscall _Lockit(int); 
#line 646
__thiscall ~_Lockit() noexcept; 
#line 649 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
static void __cdecl _Lockit_ctor(int); 
#line 650
static void __cdecl _Lockit_dtor(int); 
#line 653
private: static void __cdecl _Lockit_ctor(_Lockit *); 
#line 654
static void __cdecl _Lockit_ctor(_Lockit *, int); 
#line 655
static void __cdecl _Lockit_dtor(_Lockit *); 
#line 658
public: _Lockit(const _Lockit &) = delete;
#line 659
_Lockit &operator=(const _Lockit &) = delete;
#line 662
private: int _Locktype; 
#line 677 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
}; 
#line 764 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
class __declspec(dllimport) _Init_locks { 
#line 780 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
public: __thiscall _Init_locks(); 
#line 781
__thiscall ~_Init_locks() noexcept; 
#line 785 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
private: static void __cdecl _Init_locks_ctor(_Init_locks *); 
#line 786
static void __cdecl _Init_locks_dtor(_Init_locks *); 
#line 797 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
}; 
#line 799
#pragma warning(pop)
}
#line 809 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\yvals.h"
__declspec(dllimport) void __cdecl _Atexit(void (__cdecl *)(void)); 
#line 811
typedef unsigned long _Uint32t; 
#line 817
#pragma pack ( pop )
#line 17 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
inline double abs(double _Xx) noexcept 
#line 18
{ 
#line 19
return ::fabs(_Xx); 
#line 20
} 
#line 22
inline double pow(double _Xx, int _Yx) noexcept 
#line 23
{ 
#line 24
if (_Yx == 2) { 
#line 25
return _Xx * _Xx; }  
#line 27
return ::pow(_Xx, static_cast< double>(_Yx)); 
#line 28
} 
#line 30
inline float abs(float _Xx) noexcept 
#line 31
{ 
#line 32
return ::fabsf(_Xx); 
#line 33
} 
#line 35
inline float acos(float _Xx) noexcept 
#line 36
{ 
#line 37
return ::acosf(_Xx); 
#line 38
} 
#line 40
inline float acosh(float _Xx) noexcept 
#line 41
{ 
#line 42
return ::acoshf(_Xx); 
#line 43
} 
#line 45
inline float asin(float _Xx) noexcept 
#line 46
{ 
#line 47
return ::asinf(_Xx); 
#line 48
} 
#line 50
inline float asinh(float _Xx) noexcept 
#line 51
{ 
#line 52
return ::asinhf(_Xx); 
#line 53
} 
#line 55
inline float atan(float _Xx) noexcept 
#line 56
{ 
#line 57
return ::atanf(_Xx); 
#line 58
} 
#line 60
inline float atanh(float _Xx) noexcept 
#line 61
{ 
#line 62
return ::atanhf(_Xx); 
#line 63
} 
#line 65
inline float atan2(float _Yx, float _Xx) noexcept 
#line 66
{ 
#line 67
return ::atan2f(_Yx, _Xx); 
#line 68
} 
#line 70
inline float cbrt(float _Xx) noexcept 
#line 71
{ 
#line 72
return ::cbrtf(_Xx); 
#line 73
} 
#line 75
inline float ceil(float _Xx) noexcept 
#line 76
{ 
#line 77
return ::ceilf(_Xx); 
#line 78
} 
#line 80
inline float copysign(float _Number, float 
#line 81
_Sign) noexcept 
#line 82
{ 
#line 83
return ::copysignf(_Number, _Sign); 
#line 84
} 
#line 86
inline float cos(float _Xx) noexcept 
#line 87
{ 
#line 88
return ::cosf(_Xx); 
#line 89
} 
#line 91
inline float cosh(float _Xx) noexcept 
#line 92
{ 
#line 93
return ::coshf(_Xx); 
#line 94
} 
#line 96
inline float erf(float _Xx) noexcept 
#line 97
{ 
#line 98
return ::erff(_Xx); 
#line 99
} 
#line 101
inline float erfc(float _Xx) noexcept 
#line 102
{ 
#line 103
return ::erfcf(_Xx); 
#line 104
} 
#line 106
inline float exp(float _Xx) noexcept 
#line 107
{ 
#line 108
return ::expf(_Xx); 
#line 109
} 
#line 111
inline float exp2(float _Xx) noexcept 
#line 112
{ 
#line 113
return ::exp2f(_Xx); 
#line 114
} 
#line 116
inline float expm1(float _Xx) noexcept 
#line 117
{ 
#line 118
return ::expm1f(_Xx); 
#line 119
} 
#line 121
inline float fabs(float _Xx) noexcept 
#line 122
{ 
#line 123
return ::fabsf(_Xx); 
#line 124
} 
#line 126
inline float fdim(float _Xx, float _Yx) noexcept 
#line 127
{ 
#line 128
return ::fdimf(_Xx, _Yx); 
#line 129
} 
#line 131
inline float floor(float _Xx) noexcept 
#line 132
{ 
#line 133
return ::floorf(_Xx); 
#line 134
} 
#line 136
inline float fma(float _Xx, float _Yx, float 
#line 137
_Zx) noexcept 
#line 138
{ 
#line 139
return ::fmaf(_Xx, _Yx, _Zx); 
#line 140
} 
#line 142
inline float fmax(float _Xx, float _Yx) noexcept 
#line 143
{ 
#line 144
return ::fmaxf(_Xx, _Yx); 
#line 145
} 
#line 147
inline float fmin(float _Xx, float _Yx) noexcept 
#line 148
{ 
#line 149
return ::fminf(_Xx, _Yx); 
#line 150
} 
#line 152
inline float fmod(float _Xx, float _Yx) noexcept 
#line 153
{ 
#line 154
return ::fmodf(_Xx, _Yx); 
#line 155
} 
#line 157
inline float frexp(float _Xx, int *_Yx) noexcept 
#line 158
{ 
#line 159
return ::frexpf(_Xx, _Yx); 
#line 160
} 
#line 162
inline float hypot(float _Xx, float _Yx) noexcept 
#line 163
{ 
#line 164
return ::hypotf(_Xx, _Yx); 
#line 165
} 
#line 167
inline int ilogb(float _Xx) noexcept 
#line 168
{ 
#line 169
return ::ilogbf(_Xx); 
#line 170
} 
#line 172
inline float ldexp(float _Xx, int _Yx) noexcept 
#line 173
{ 
#line 174
return ::ldexpf(_Xx, _Yx); 
#line 175
} 
#line 177
inline float lgamma(float _Xx) noexcept 
#line 178
{ 
#line 179
return ::lgammaf(_Xx); 
#line 180
} 
#line 182
inline __int64 llrint(float _Xx) noexcept 
#line 183
{ 
#line 184
return ::llrintf(_Xx); 
#line 185
} 
#line 187
inline __int64 llround(float _Xx) noexcept 
#line 188
{ 
#line 189
return ::llroundf(_Xx); 
#line 190
} 
#line 192
inline float log(float _Xx) noexcept 
#line 193
{ 
#line 194
return ::logf(_Xx); 
#line 195
} 
#line 197
inline float log10(float _Xx) noexcept 
#line 198
{ 
#line 199
return ::log10f(_Xx); 
#line 200
} 
#line 202
inline float log1p(float _Xx) noexcept 
#line 203
{ 
#line 204
return ::log1pf(_Xx); 
#line 205
} 
#line 207
inline float log2(float _Xx) noexcept 
#line 208
{ 
#line 209
return ::log2f(_Xx); 
#line 210
} 
#line 212
inline float logb(float _Xx) noexcept 
#line 213
{ 
#line 214
return ::logbf(_Xx); 
#line 215
} 
#line 217
inline long lrint(float _Xx) noexcept 
#line 218
{ 
#line 219
return ::lrintf(_Xx); 
#line 220
} 
#line 222
inline long lround(float _Xx) noexcept 
#line 223
{ 
#line 224
return ::lroundf(_Xx); 
#line 225
} 
#line 227
inline float modf(float _Xx, float *_Yx) noexcept 
#line 228
{ 
#line 229
return ::modff(_Xx, _Yx); 
#line 230
} 
#line 232
inline float nearbyint(float _Xx) noexcept 
#line 233
{ 
#line 234
return ::nearbyintf(_Xx); 
#line 235
} 
#line 237
inline float nextafter(float _Xx, float _Yx) noexcept 
#line 238
{ 
#line 239
return ::nextafterf(_Xx, _Yx); 
#line 240
} 
#line 242
inline float nexttoward(float _Xx, long double 
#line 243
_Yx) noexcept 
#line 244
{ 
#line 245
return ::nexttowardf(_Xx, _Yx); 
#line 246
} 
#line 248
inline float pow(float _Xx, float 
#line 249
_Yx) noexcept 
#line 250
{ 
#line 251
return ::powf(_Xx, _Yx); 
#line 252
} 
#line 254
inline float pow(float _Xx, int _Yx) noexcept 
#line 255
{ 
#line 256
if (_Yx == 2) { 
#line 257
return _Xx * _Xx; }  
#line 259
return ::powf(_Xx, static_cast< float>(_Yx)); 
#line 260
} 
#line 262
inline float remainder(float _Xx, float _Yx) noexcept 
#line 263
{ 
#line 264
return ::remainderf(_Xx, _Yx); 
#line 265
} 
#line 267
inline float remquo(float _Xx, float _Yx, int *
#line 268
_Zx) noexcept 
#line 269
{ 
#line 270
return ::remquof(_Xx, _Yx, _Zx); 
#line 271
} 
#line 273
inline float rint(float _Xx) noexcept 
#line 274
{ 
#line 275
return ::rintf(_Xx); 
#line 276
} 
#line 278
inline float round(float _Xx) noexcept 
#line 279
{ 
#line 280
return ::roundf(_Xx); 
#line 281
} 
#line 283
inline float scalbln(float _Xx, long _Yx) noexcept 
#line 284
{ 
#line 285
return ::scalblnf(_Xx, _Yx); 
#line 286
} 
#line 288
inline float scalbn(float _Xx, int _Yx) noexcept 
#line 289
{ 
#line 290
return ::scalbnf(_Xx, _Yx); 
#line 291
} 
#line 293
inline float sin(float _Xx) noexcept 
#line 294
{ 
#line 295
return ::sinf(_Xx); 
#line 296
} 
#line 298
inline float sinh(float _Xx) noexcept 
#line 299
{ 
#line 300
return ::sinhf(_Xx); 
#line 301
} 
#line 303
inline float sqrt(float _Xx) noexcept 
#line 304
{ 
#line 305
return ::sqrtf(_Xx); 
#line 306
} 
#line 308
inline float tan(float _Xx) noexcept 
#line 309
{ 
#line 310
return ::tanf(_Xx); 
#line 311
} 
#line 313
inline float tanh(float _Xx) noexcept 
#line 314
{ 
#line 315
return ::tanhf(_Xx); 
#line 316
} 
#line 318
inline float tgamma(float _Xx) noexcept 
#line 319
{ 
#line 320
return ::tgammaf(_Xx); 
#line 321
} 
#line 323
inline float trunc(float _Xx) noexcept 
#line 324
{ 
#line 325
return ::truncf(_Xx); 
#line 326
} 
#line 328
inline long double abs(long double _Xx) noexcept 
#line 329
{ 
#line 330
return ::fabsl(_Xx); 
#line 331
} 
#line 333
inline long double acos(long double _Xx) noexcept 
#line 334
{ 
#line 335
return ::acosl(_Xx); 
#line 336
} 
#line 338
inline long double acosh(long double _Xx) noexcept 
#line 339
{ 
#line 340
return ::acoshl(_Xx); 
#line 341
} 
#line 343
inline long double asin(long double _Xx) noexcept 
#line 344
{ 
#line 345
return ::asinl(_Xx); 
#line 346
} 
#line 348
inline long double asinh(long double _Xx) noexcept 
#line 349
{ 
#line 350
return ::asinhl(_Xx); 
#line 351
} 
#line 353
inline long double atan(long double _Xx) noexcept 
#line 354
{ 
#line 355
return ::atanl(_Xx); 
#line 356
} 
#line 358
inline long double atanh(long double _Xx) noexcept 
#line 359
{ 
#line 360
return ::atanhl(_Xx); 
#line 361
} 
#line 363
inline long double atan2(long double _Yx, long double 
#line 364
_Xx) noexcept 
#line 365
{ 
#line 366
return ::atan2l(_Yx, _Xx); 
#line 367
} 
#line 369
inline long double cbrt(long double _Xx) noexcept 
#line 370
{ 
#line 371
return ::cbrtl(_Xx); 
#line 372
} 
#line 374
inline long double ceil(long double _Xx) noexcept 
#line 375
{ 
#line 376
return ::ceill(_Xx); 
#line 377
} 
#line 379
inline long double copysign(long double _Number, long double 
#line 380
_Sign) noexcept 
#line 381
{ 
#line 382
return ::copysignl(_Number, _Sign); 
#line 383
} 
#line 385
inline long double cos(long double _Xx) noexcept 
#line 386
{ 
#line 387
return ::cosl(_Xx); 
#line 388
} 
#line 390
inline long double cosh(long double _Xx) noexcept 
#line 391
{ 
#line 392
return ::coshl(_Xx); 
#line 393
} 
#line 395
inline long double erf(long double _Xx) noexcept 
#line 396
{ 
#line 397
return ::erfl(_Xx); 
#line 398
} 
#line 400
inline long double erfc(long double _Xx) noexcept 
#line 401
{ 
#line 402
return ::erfcl(_Xx); 
#line 403
} 
#line 405
inline long double exp(long double _Xx) noexcept 
#line 406
{ 
#line 407
return ::expl(_Xx); 
#line 408
} 
#line 410
inline long double exp2(long double _Xx) noexcept 
#line 411
{ 
#line 412
return ::exp2l(_Xx); 
#line 413
} 
#line 415
inline long double expm1(long double _Xx) noexcept 
#line 416
{ 
#line 417
return ::expm1l(_Xx); 
#line 418
} 
#line 420
inline long double fabs(long double _Xx) noexcept 
#line 421
{ 
#line 422
return ::fabsl(_Xx); 
#line 423
} 
#line 425
inline long double fdim(long double _Xx, long double 
#line 426
_Yx) noexcept 
#line 427
{ 
#line 428
return ::fdiml(_Xx, _Yx); 
#line 429
} 
#line 431
inline long double floor(long double _Xx) noexcept 
#line 432
{ 
#line 433
return ::floorl(_Xx); 
#line 434
} 
#line 436
inline long double fma(long double _Xx, long double 
#line 437
_Yx, long double _Zx) noexcept 
#line 438
{ 
#line 439
return ::fmal(_Xx, _Yx, _Zx); 
#line 440
} 
#line 442
inline long double fmax(long double _Xx, long double 
#line 443
_Yx) noexcept 
#line 444
{ 
#line 445
return ::fmaxl(_Xx, _Yx); 
#line 446
} 
#line 448
inline long double fmin(long double _Xx, long double 
#line 449
_Yx) noexcept 
#line 450
{ 
#line 451
return ::fminl(_Xx, _Yx); 
#line 452
} 
#line 454
inline long double fmod(long double _Xx, long double 
#line 455
_Yx) noexcept 
#line 456
{ 
#line 457
return ::fmodl(_Xx, _Yx); 
#line 458
} 
#line 460
inline long double frexp(long double _Xx, int *
#line 461
_Yx) noexcept 
#line 462
{ 
#line 463
return ::frexpl(_Xx, _Yx); 
#line 464
} 
#line 466
inline long double hypot(long double _Xx, long double 
#line 467
_Yx) noexcept 
#line 468
{ 
#line 469
return ::hypotl(_Xx, _Yx); 
#line 470
} 
#line 472
inline int ilogb(long double _Xx) noexcept 
#line 473
{ 
#line 474
return ::ilogbl(_Xx); 
#line 475
} 
#line 477
inline long double ldexp(long double _Xx, int 
#line 478
_Yx) noexcept 
#line 479
{ 
#line 480
return ::ldexpl(_Xx, _Yx); 
#line 481
} 
#line 483
inline long double lgamma(long double _Xx) noexcept 
#line 484
{ 
#line 485
return ::lgammal(_Xx); 
#line 486
} 
#line 488
inline __int64 llrint(long double _Xx) noexcept 
#line 489
{ 
#line 490
return ::llrintl(_Xx); 
#line 491
} 
#line 493
inline __int64 llround(long double _Xx) noexcept 
#line 494
{ 
#line 495
return ::llroundl(_Xx); 
#line 496
} 
#line 498
inline long double log(long double _Xx) noexcept 
#line 499
{ 
#line 500
return ::logl(_Xx); 
#line 501
} 
#line 503
inline long double log10(long double _Xx) noexcept 
#line 504
{ 
#line 505
return ::log10l(_Xx); 
#line 506
} 
#line 508
inline long double log1p(long double _Xx) noexcept 
#line 509
{ 
#line 510
return ::log1pl(_Xx); 
#line 511
} 
#line 513
inline long double log2(long double _Xx) noexcept 
#line 514
{ 
#line 515
return ::log2l(_Xx); 
#line 516
} 
#line 518
inline long double logb(long double _Xx) noexcept 
#line 519
{ 
#line 520
return ::logbl(_Xx); 
#line 521
} 
#line 523
inline long lrint(long double _Xx) noexcept 
#line 524
{ 
#line 525
return ::lrintl(_Xx); 
#line 526
} 
#line 528
inline long lround(long double _Xx) noexcept 
#line 529
{ 
#line 530
return ::lroundl(_Xx); 
#line 531
} 
#line 533
inline long double modf(long double _Xx, long double *
#line 534
_Yx) noexcept 
#line 535
{ 
#line 536
return ::modfl(_Xx, _Yx); 
#line 537
} 
#line 539
inline long double nearbyint(long double _Xx) noexcept 
#line 540
{ 
#line 541
return ::nearbyintl(_Xx); 
#line 542
} 
#line 544
inline long double nextafter(long double _Xx, long double 
#line 545
_Yx) noexcept 
#line 546
{ 
#line 547
return ::nextafterl(_Xx, _Yx); 
#line 548
} 
#line 550
inline long double nexttoward(long double _Xx, long double 
#line 551
_Yx) noexcept 
#line 552
{ 
#line 553
return ::nexttowardl(_Xx, _Yx); 
#line 554
} 
#line 556
inline long double pow(long double _Xx, long double 
#line 557
_Yx) noexcept 
#line 558
{ 
#line 559
return ::powl(_Xx, _Yx); 
#line 560
} 
#line 562
inline long double pow(long double _Xx, int 
#line 563
_Yx) noexcept 
#line 564
{ 
#line 565
if (_Yx == 2) { 
#line 566
return _Xx * _Xx; }  
#line 568
return ::powl(_Xx, static_cast< long double>(_Yx)); 
#line 569
} 
#line 571
inline long double remainder(long double _Xx, long double 
#line 572
_Yx) noexcept 
#line 573
{ 
#line 574
return ::remainderl(_Xx, _Yx); 
#line 575
} 
#line 577
inline long double remquo(long double _Xx, long double 
#line 578
_Yx, int *_Zx) noexcept 
#line 579
{ 
#line 580
return ::remquol(_Xx, _Yx, _Zx); 
#line 581
} 
#line 583
inline long double rint(long double _Xx) noexcept 
#line 584
{ 
#line 585
return ::rintl(_Xx); 
#line 586
} 
#line 588
inline long double round(long double _Xx) noexcept 
#line 589
{ 
#line 590
return ::roundl(_Xx); 
#line 591
} 
#line 593
inline long double scalbln(long double _Xx, long 
#line 594
_Yx) noexcept 
#line 595
{ 
#line 596
return ::scalblnl(_Xx, _Yx); 
#line 597
} 
#line 599
inline long double scalbn(long double _Xx, int 
#line 600
_Yx) noexcept 
#line 601
{ 
#line 602
return ::scalbnl(_Xx, _Yx); 
#line 603
} 
#line 605
inline long double sin(long double _Xx) noexcept 
#line 606
{ 
#line 607
return ::sinl(_Xx); 
#line 608
} 
#line 610
inline long double sinh(long double _Xx) noexcept 
#line 611
{ 
#line 612
return ::sinhl(_Xx); 
#line 613
} 
#line 615
inline long double sqrt(long double _Xx) noexcept 
#line 616
{ 
#line 617
return ::sqrtl(_Xx); 
#line 618
} 
#line 620
inline long double tan(long double _Xx) noexcept 
#line 621
{ 
#line 622
return ::tanl(_Xx); 
#line 623
} 
#line 625
inline long double tanh(long double _Xx) noexcept 
#line 626
{ 
#line 627
return ::tanhl(_Xx); 
#line 628
} 
#line 630
inline long double tgamma(long double _Xx) noexcept 
#line 631
{ 
#line 632
return ::tgammal(_Xx); 
#line 633
} 
#line 635
inline long double trunc(long double _Xx) noexcept 
#line 636
{ 
#line 637
return ::truncl(_Xx); 
#line 638
} 
#line 17 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cstdlib"
namespace std { 
#line 18
using ::size_t;using ::div_t;using ::ldiv_t;
#line 19
using ::abort;using ::abs;using ::atexit;
#line 20
using ::atof;using ::atoi;using ::atol;
#line 21
using ::bsearch;using ::calloc;using ::div;
#line 22
using ::exit;using ::free;
#line 23
using ::labs;using ::ldiv;using ::malloc;
#line 24
using ::mblen;using ::mbstowcs;using ::mbtowc;
#line 25
using ::qsort;using ::rand;using ::realloc;
#line 26
using ::srand;using ::strtod;using ::strtol;
#line 27
using ::strtoul;
#line 28
using ::wcstombs;using ::wctomb;
#line 30
using ::lldiv_t;
#line 33
using ::getenv;
#line 34
using ::system;
#line 37 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cstdlib"
using ::atoll;using ::llabs;using ::lldiv;
#line 38
using ::strtof;using ::strtold;
#line 39
using ::strtoll;using ::strtoull;
#line 41
using ::_Exit;using ::at_quick_exit;using ::quick_exit;
#line 42
}
#line 9 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
#pragma warning(push,3)
#line 8
#pragma pack ( push, 8 )
#line 13
namespace std { 
#line 15
struct _Nil { 
#line 17
}; 
#line 20
template < class _Ty,
 _Ty _Val >
 struct integral_constant
 {
 static constexpr _Ty value = _Val;

 typedef _Ty value_type;
 typedef integral_constant < _Ty, _Val > type;

 constexpr operator value_type ( ) const noexcept
  {
  return ( value );
  }

 constexpr value_type operator ( ) ( ) const noexcept
  {
  return ( value );
  }
 };
#line 40
typedef integral_constant< bool, true>  true_type; 
#line 41
typedef integral_constant< bool, false>  false_type; 
#line 44
template< bool _Val> using bool_constant = integral_constant< bool, _Val> ; 
#line 48
template< bool _Val> 
#line 49
struct _Cat_base : public integral_constant< bool, _Val>  { 
#line 52
}; 
#line 55
template< bool _Test, class 
#line 56
_Ty = void> 
#line 57
struct enable_if { 
#line 59
}; 
#line 61
template< class _Ty> 
#line 62
struct enable_if< true, _Ty>  { 
#line 64
typedef _Ty type; 
#line 65
}; 
#line 68
template< bool _Test, class 
#line 69
_Ty1, class 
#line 70
_Ty2> 
#line 71
struct conditional { 
#line 73
typedef _Ty2 type; 
#line 74
}; 
#line 76
template< class _Ty1, class 
#line 77
_Ty2> 
#line 78
struct conditional< true, _Ty1, _Ty2>  { 
#line 80
typedef _Ty1 type; 
#line 81
}; 
#line 84
template< class _Ty1, class 
#line 85
_Ty2> 
#line 86
struct is_same : public false_type { 
#line 89
}; 
#line 91
template< class _Ty1> 
#line 92
struct is_same< _Ty1, _Ty1>  : public true_type { 
#line 95
}; 
#line 98
template< class _Ty, class 
#line 99
_Uty> constexpr bool 
#line 100
is_same_v = (is_same< _Ty, _Uty> ::value); 
#line 104 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
template< class _Ty> 
#line 105
struct remove_const { 
#line 107
typedef _Ty type; 
#line 108
}; 
#line 110
template< class _Ty> 
#line 111
struct remove_const< const _Ty>  { 
#line 113
typedef _Ty type; 
#line 114
}; 
#line 117
template< class _Ty> 
#line 118
struct remove_volatile { 
#line 120
typedef _Ty type; 
#line 121
}; 
#line 123
template< class _Ty> 
#line 124
struct remove_volatile< volatile _Ty>  { 
#line 126
typedef _Ty type; 
#line 127
}; 
#line 130
template< class _Ty> 
#line 131
struct remove_cv { 
#line 134
typedef typename remove_const< typename remove_volatile< _Ty> ::type> ::type type; 
#line 135
}; 
#line 138
template< class _Ty> 
#line 139
struct _Is_integral : public false_type { 
#line 142
}; 
#line 145
template<> struct _Is_integral< bool>  : public true_type { 
#line 148
}; 
#line 151
template<> struct _Is_integral< char>  : public true_type { 
#line 154
}; 
#line 157
template<> struct _Is_integral< unsigned char>  : public true_type { 
#line 160
}; 
#line 163
template<> struct _Is_integral< signed char>  : public true_type { 
#line 166
}; 
#line 170
template<> struct _Is_integral< __wchar_t>  : public true_type { 
#line 173
}; 
#line 177 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
template<> struct _Is_integral< unsigned short>  : public true_type { 
#line 180
}; 
#line 183
template<> struct _Is_integral< signed short>  : public true_type { 
#line 186
}; 
#line 189
template<> struct _Is_integral< unsigned>  : public true_type { 
#line 192
}; 
#line 195
template<> struct _Is_integral< signed int>  : public true_type { 
#line 198
}; 
#line 201
template<> struct _Is_integral< unsigned long>  : public true_type { 
#line 204
}; 
#line 207
template<> struct _Is_integral< signed long>  : public true_type { 
#line 210
}; 
#line 213
template<> struct _Is_integral< char16_t>  : public true_type { 
#line 216
}; 
#line 219
template<> struct _Is_integral< char32_t>  : public true_type { 
#line 222
}; 
#line 225
template<> struct _Is_integral< __int64>  : public true_type { 
#line 228
}; 
#line 231
template<> struct _Is_integral< unsigned __int64>  : public true_type { 
#line 234
}; 
#line 237
template< class _Ty> 
#line 238
struct is_integral : public _Is_integral< typename remove_cv< _Ty> ::type>  { 
#line 241
}; 
#line 244
template< class _Ty> constexpr bool 
#line 245
is_integral_v = (is_integral< _Ty> ::value); 
#line 249 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
template< class _Ty> 
#line 250
struct _Is_floating_point : public false_type { 
#line 253
}; 
#line 256
template<> struct _Is_floating_point< float>  : public true_type { 
#line 259
}; 
#line 262
template<> struct _Is_floating_point< double>  : public true_type { 
#line 265
}; 
#line 268
template<> struct _Is_floating_point< long double>  : public true_type { 
#line 271
}; 
#line 274
template< class _Ty> 
#line 275
struct is_floating_point : public _Is_floating_point< typename remove_cv< _Ty> ::type>  { 
#line 278
}; 
#line 281
template< class _Ty> constexpr bool 
#line 282
is_floating_point_v = (is_floating_point< _Ty> ::value); 
#line 286 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
template< class _Ty> 
#line 287
struct is_arithmetic : public _Cat_base< is_integral< _Ty> ::value || is_floating_point< _Ty> ::value>  { 
#line 291
}; 
#line 294
template< class _Ty> constexpr bool 
#line 295
is_arithmetic_v = (is_arithmetic< _Ty> ::value); 
#line 299 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
template< class _Ty> 
#line 300
struct remove_reference { 
#line 302
typedef _Ty type; 
#line 303
}; 
#line 305
template< class _Ty> 
#line 306
struct remove_reference< _Ty &>  { 
#line 308
typedef _Ty type; 
#line 309
}; 
#line 311
template< class _Ty> 
#line 312
struct remove_reference< _Ty &&>  { 
#line 314
typedef _Ty type; 
#line 315
}; 
#line 318
struct _Wrap_int { 
#line 320
_Wrap_int(int) 
#line 321
{ 
#line 322
} 
#line 323
}; 
#line 325
template< class _Ty> 
#line 326
struct _Identity { 
#line 328
typedef _Ty type; 
#line 329
}; 
#line 344
}
#line 346
#pragma warning(pop)
#line 11 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtgmath.h"
#pragma warning(push,3)
#line 347 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtr1common"
#pragma pack ( pop )
#line 10 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\xtgmath.h"
#pragma pack ( push, 8 )
#line 15
namespace std { 
#line 16
template< class _Ty> 
#line 17
struct _Promote_to_float { 
#line 20
typedef typename conditional< is_integral< _Ty> ::value, double, _Ty> ::type type; 
#line 21
}; 
#line 23
template< class _Ty1, class 
#line 24
_Ty2> 
#line 25
struct _Common_float_type { 
#line 27
typedef typename _Promote_to_float< _Ty1> ::type _Ty1f; 
#line 28
typedef typename _Promote_to_float< _Ty2> ::type _Ty2f; 
#line 33
typedef typename conditional< is_same< typename _Promote_to_float< _Ty1> ::type, long double> ::value || is_same< typename _Promote_to_float< _Ty2> ::type, long double> ::value, long double, typename conditional< is_same< typename _Promote_to_float< _Ty1> ::type, double> ::value || is_same< typename _Promote_to_float< _Ty2> ::type, double> ::value, double, float> ::type> ::type type; 
#line 34
}; 
#line 35
}
#line 76
template < class _Ty1,
 class _Ty2 > inline
 typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value
  && :: std :: is_arithmetic < _Ty2 > :: value,
  typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type
 pow ( const _Ty1 _Left, const _Ty2 _Right )
 {
 typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type;
 return ( :: pow ( type ( _Left ), type ( _Right ) ) );
 }
#line 88
extern "C" double __cdecl acos(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type acos ( _Ty _Left ) { return ( :: acos ( ( double ) _Left ) ); }
#line 89
extern "C" double __cdecl asin(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type asin ( _Ty _Left ) { return ( :: asin ( ( double ) _Left ) ); }
#line 90
extern "C" double __cdecl atan(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type atan ( _Ty _Left ) { return ( :: atan ( ( double ) _Left ) ); }
#line 91
extern "C" double __cdecl atan2(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type atan2 ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: atan2 ( ( type ) _Left, ( type ) _Right ) ); }
#line 92
extern "C" __declspec(dllimport) double __cdecl ceil(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type ceil ( _Ty _Left ) { return ( :: ceil ( ( double ) _Left ) ); }
#line 93
extern "C" double __cdecl cos(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type cos ( _Ty _Left ) { return ( :: cos ( ( double ) _Left ) ); }
#line 94
extern "C" double __cdecl cosh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type cosh ( _Ty _Left ) { return ( :: cosh ( ( double ) _Left ) ); }
#line 95
extern "C" double __cdecl exp(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type exp ( _Ty _Left ) { return ( :: exp ( ( double ) _Left ) ); }
#line 97
extern "C" double __cdecl fabs(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type fabs ( _Ty _Left ) { return ( :: fabs ( ( double ) _Left ) ); }
#line 99
extern "C" __declspec(dllimport) double __cdecl floor(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type floor ( _Ty _Left ) { return ( :: floor ( ( double ) _Left ) ); }
#line 100
extern "C" double __cdecl fmod(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type fmod ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: fmod ( ( type ) _Left, ( type ) _Right ) ); }
#line 101
extern "C" __declspec(dllimport) double __cdecl frexp(double, int *); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type frexp ( _Ty _Left, int * _Arg2 ) { return ( :: frexp ( ( double ) _Left, _Arg2 ) ); }
#line 102
extern "C" __declspec(dllimport) double __cdecl ldexp(double, int); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type ldexp ( _Ty _Left, int _Arg2 ) { return ( :: ldexp ( ( double ) _Left, _Arg2 ) ); }
#line 103
extern "C" double __cdecl log(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type log ( _Ty _Left ) { return ( :: log ( ( double ) _Left ) ); }
#line 104
extern "C" double __cdecl log10(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type log10 ( _Ty _Left ) { return ( :: log10 ( ( double ) _Left ) ); }
#line 107
extern "C" double __cdecl sin(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type sin ( _Ty _Left ) { return ( :: sin ( ( double ) _Left ) ); }
#line 108
extern "C" double __cdecl sinh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type sinh ( _Ty _Left ) { return ( :: sinh ( ( double ) _Left ) ); }
#line 109
extern "C" double __cdecl sqrt(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type sqrt ( _Ty _Left ) { return ( :: sqrt ( ( double ) _Left ) ); }
#line 110
extern "C" double __cdecl tan(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type tan ( _Ty _Left ) { return ( :: tan ( ( double ) _Left ) ); }
#line 111
extern "C" double __cdecl tanh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type tanh ( _Ty _Left ) { return ( :: tanh ( ( double ) _Left ) ); }
#line 125
inline float _Fma(float _Left, float _Middle, float _Right) 
#line 126
{ 
#line 127
return ::fmaf(_Left, _Middle, _Right); 
#line 128
} 
#line 130
inline double _Fma(double _Left, double _Middle, double _Right) 
#line 131
{ 
#line 132
return ::fma(_Left, _Middle, _Right); 
#line 133
} 
#line 135
inline long double _Fma(long double _Left, long double _Middle, long double 
#line 136
_Right) 
#line 137
{ 
#line 138
return ::fmal(_Left, _Middle, _Right); 
#line 139
} 
#line 141
template < class _Ty1,
 class _Ty2,
 class _Ty3 > inline
 typename :: std :: _Common_float_type < _Ty1,
  typename :: std :: _Common_float_type < _Ty2, _Ty3 > :: type > :: type
 fma ( _Ty1 _Left, _Ty2 _Middle, _Ty3 _Right )
 {
 typedef typename :: std :: _Common_float_type < _Ty1,
  typename :: std :: _Common_float_type < _Ty2, _Ty3 > :: type > :: type type;
 return ( _Fma ( ( type ) _Left, ( type ) _Middle, ( type ) _Right ) );
 }
#line 155
inline float _Remquo(float _Left, float _Right, int *_Pquo) 
#line 156
{ 
#line 157
return ::remquof(_Left, _Right, _Pquo); 
#line 158
} 
#line 160
inline double _Remquo(double _Left, double _Right, int *_Pquo) 
#line 161
{ 
#line 162
return ::remquo(_Left, _Right, _Pquo); 
#line 163
} 
#line 165
inline long double _Remquo(long double _Left, long double _Right, int *_Pquo) 
#line 166
{ 
#line 167
return ::remquol(_Left, _Right, _Pquo); 
#line 168
} 
#line 170
template < class _Ty1,
 class _Ty2 > inline
 typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type
 remquo ( _Ty1 _Left, _Ty2 _Right, int * _Pquo )
 {
 typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type;
 return ( _Remquo ( ( type ) _Left, ( type ) _Right, _Pquo ) );
 }
#line 179
extern "C" __declspec(dllimport) double __cdecl acosh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type acosh ( _Ty _Left ) { return ( :: acosh ( ( double ) _Left ) ); }
#line 180
extern "C" __declspec(dllimport) double __cdecl asinh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type asinh ( _Ty _Left ) { return ( :: asinh ( ( double ) _Left ) ); }
#line 181
extern "C" __declspec(dllimport) double __cdecl atanh(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type atanh ( _Ty _Left ) { return ( :: atanh ( ( double ) _Left ) ); }
#line 182
extern "C" __declspec(dllimport) double __cdecl cbrt(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type cbrt ( _Ty _Left ) { return ( :: cbrt ( ( double ) _Left ) ); }
#line 183
extern "C" __declspec(dllimport) double __cdecl copysign(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type copysign ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: copysign ( ( type ) _Left, ( type ) _Right ) ); }
#line 184
extern "C" __declspec(dllimport) double __cdecl erf(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type erf ( _Ty _Left ) { return ( :: erf ( ( double ) _Left ) ); }
#line 185
extern "C" __declspec(dllimport) double __cdecl erfc(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type erfc ( _Ty _Left ) { return ( :: erfc ( ( double ) _Left ) ); }
#line 186
extern "C" __declspec(dllimport) double __cdecl expm1(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type expm1 ( _Ty _Left ) { return ( :: expm1 ( ( double ) _Left ) ); }
#line 187
extern "C" __declspec(dllimport) double __cdecl exp2(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type exp2 ( _Ty _Left ) { return ( :: exp2 ( ( double ) _Left ) ); }
#line 188
extern "C" __declspec(dllimport) double __cdecl fdim(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type fdim ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: fdim ( ( type ) _Left, ( type ) _Right ) ); }
#line 190
extern "C" __declspec(dllimport) double __cdecl fmax(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type fmax ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: fmax ( ( type ) _Left, ( type ) _Right ) ); }
#line 191
extern "C" __declspec(dllimport) double __cdecl fmin(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type fmin ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: fmin ( ( type ) _Left, ( type ) _Right ) ); }
#line 192
extern "C" __declspec(dllimport) double __cdecl hypot(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type hypot ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: hypot ( ( type ) _Left, ( type ) _Right ) ); }
#line 193
extern "C" __declspec(dllimport) int __cdecl ilogb(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, int > :: type ilogb ( _Ty _Left ) { return ( :: ilogb ( ( double ) _Left ) ); }
#line 194
extern "C" __declspec(dllimport) double __cdecl lgamma(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type lgamma ( _Ty _Left ) { return ( :: lgamma ( ( double ) _Left ) ); }
#line 195
extern "C" __declspec(dllimport) __int64 __cdecl llrint(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, long long > :: type llrint ( _Ty _Left ) { return ( :: llrint ( ( double ) _Left ) ); }
#line 196
extern "C" __declspec(dllimport) __int64 __cdecl llround(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, long long > :: type llround ( _Ty _Left ) { return ( :: llround ( ( double ) _Left ) ); }
#line 197
extern "C" __declspec(dllimport) double __cdecl log1p(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type log1p ( _Ty _Left ) { return ( :: log1p ( ( double ) _Left ) ); }
#line 198
extern "C" __declspec(dllimport) double __cdecl log2(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type log2 ( _Ty _Left ) { return ( :: log2 ( ( double ) _Left ) ); }
#line 199
extern "C" __declspec(dllimport) double __cdecl logb(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type logb ( _Ty _Left ) { return ( :: logb ( ( double ) _Left ) ); }
#line 200
extern "C" __declspec(dllimport) long __cdecl lrint(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, long > :: type lrint ( _Ty _Left ) { return ( :: lrint ( ( double ) _Left ) ); }
#line 201
extern "C" __declspec(dllimport) long __cdecl lround(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, long > :: type lround ( _Ty _Left ) { return ( :: lround ( ( double ) _Left ) ); }
#line 202
extern "C" __declspec(dllimport) double __cdecl nearbyint(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type nearbyint ( _Ty _Left ) { return ( :: nearbyint ( ( double ) _Left ) ); }
#line 203
extern "C" __declspec(dllimport) double __cdecl nextafter(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type nextafter ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: nextafter ( ( type ) _Left, ( type ) _Right ) ); }
#line 204
extern "C" __declspec(dllimport) double __cdecl nexttoward(double, long double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type nexttoward ( _Ty _Left, long double _Arg2 ) { return ( :: nexttoward ( ( double ) _Left, _Arg2 ) ); }
#line 205
extern "C" __declspec(dllimport) double __cdecl remainder(double, double); template < class _Ty1, class _Ty2 > inline typename :: std :: enable_if < :: std :: is_arithmetic < _Ty1 > :: value && :: std :: is_arithmetic < _Ty2 > :: value, typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type > :: type remainder ( _Ty1 _Left, _Ty2 _Right ) { typedef typename :: std :: _Common_float_type < _Ty1, _Ty2 > :: type type; return ( :: remainder ( ( type ) _Left, ( type ) _Right ) ); }
#line 207
extern "C" __declspec(dllimport) double __cdecl rint(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type rint ( _Ty _Left ) { return ( :: rint ( ( double ) _Left ) ); }
#line 208
extern "C" __declspec(dllimport) double __cdecl round(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type round ( _Ty _Left ) { return ( :: round ( ( double ) _Left ) ); }
#line 209
extern "C" __declspec(dllimport) double __cdecl scalbln(double, long); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type scalbln ( _Ty _Left, long _Arg2 ) { return ( :: scalbln ( ( double ) _Left, _Arg2 ) ); }
#line 210
extern "C" __declspec(dllimport) double __cdecl scalbn(double, int); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type scalbn ( _Ty _Left, int _Arg2 ) { return ( :: scalbn ( ( double ) _Left, _Arg2 ) ); }
#line 211
extern "C" __declspec(dllimport) double __cdecl tgamma(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type tgamma ( _Ty _Left ) { return ( :: tgamma ( ( double ) _Left ) ); }
#line 212
extern "C" __declspec(dllimport) double __cdecl trunc(double); template < class _Ty > inline typename :: std :: enable_if < :: std :: is_integral < _Ty > :: value, double > :: type trunc ( _Ty _Left ) { return ( :: trunc ( ( double ) _Left ) ); }
#line 215
#pragma warning(pop)
#pragma pack ( pop )
#line 651 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
namespace std { 
#line 652
using ::abs;using ::acos;using ::asin;
#line 653
using ::atan;using ::atan2;using ::ceil;
#line 654
using ::cos;using ::cosh;using ::exp;
#line 655
using ::fabs;using ::floor;using ::fmod;
#line 656
using ::frexp;using ::ldexp;using ::log;
#line 657
using ::log10;using ::modf;using ::pow;
#line 658
using ::sin;using ::sinh;using ::sqrt;
#line 659
using ::tan;using ::tanh;
#line 661
using ::acosf;using ::asinf;
#line 662
using ::atanf;using ::atan2f;using ::ceilf;
#line 663
using ::cosf;using ::coshf;using ::expf;
#line 664
using ::fabsf;using ::floorf;using ::fmodf;
#line 665
using ::frexpf;using ::ldexpf;using ::logf;
#line 666
using ::log10f;using ::modff;using ::powf;
#line 667
using ::sinf;using ::sinhf;using ::sqrtf;
#line 668
using ::tanf;using ::tanhf;
#line 670
using ::acosl;using ::asinl;
#line 671
using ::atanl;using ::atan2l;using ::ceill;
#line 672
using ::cosl;using ::coshl;using ::expl;
#line 673
using ::fabsl;using ::floorl;using ::fmodl;
#line 674
using ::frexpl;using ::ldexpl;using ::logl;
#line 675
using ::log10l;using ::modfl;using ::powl;
#line 676
using ::sinl;using ::sinhl;using ::sqrtl;
#line 677
using ::tanl;using ::tanhl;
#line 679
using ::float_t;using ::double_t;
#line 681
using ::acosh;using ::asinh;using ::atanh;
#line 682
using ::cbrt;using ::erf;using ::erfc;
#line 683
using ::expm1;using ::exp2;
#line 684
using ::hypot;using ::ilogb;using ::lgamma;
#line 685
using ::log1p;using ::log2;using ::logb;
#line 686
using ::llrint;using ::lrint;using ::nearbyint;
#line 687
using ::rint;using ::llround;using ::lround;
#line 688
using ::fdim;using ::fma;using ::fmax;using ::fmin;
#line 689
using ::round;using ::trunc;
#line 690
using ::remainder;using ::remquo;
#line 691
using ::copysign;using ::nan;using ::nextafter;
#line 692
using ::scalbn;using ::scalbln;
#line 693
using ::nexttoward;using ::tgamma;
#line 695
using ::acoshf;using ::asinhf;using ::atanhf;
#line 696
using ::cbrtf;using ::erff;using ::erfcf;
#line 697
using ::expm1f;using ::exp2f;
#line 698
using ::hypotf;using ::ilogbf;using ::lgammaf;
#line 699
using ::log1pf;using ::log2f;using ::logbf;
#line 700
using ::llrintf;using ::lrintf;using ::nearbyintf;
#line 701
using ::rintf;using ::llroundf;using ::lroundf;
#line 702
using ::fdimf;using ::fmaf;using ::fmaxf;using ::fminf;
#line 703
using ::roundf;using ::truncf;
#line 704
using ::remainderf;using ::remquof;
#line 705
using ::copysignf;using ::nanf;
#line 706
using ::nextafterf;using ::scalbnf;using ::scalblnf;
#line 707
using ::nexttowardf;using ::tgammaf;
#line 709
using ::acoshl;using ::asinhl;using ::atanhl;
#line 710
using ::cbrtl;using ::erfl;using ::erfcl;
#line 711
using ::expm1l;using ::exp2l;
#line 712
using ::hypotl;using ::ilogbl;using ::lgammal;
#line 713
using ::log1pl;using ::log2l;using ::logbl;
#line 714
using ::llrintl;using ::lrintl;using ::nearbyintl;
#line 715
using ::rintl;using ::llroundl;using ::lroundl;
#line 716
using ::fdiml;using ::fmal;using ::fmaxl;using ::fminl;
#line 717
using ::roundl;using ::truncl;
#line 718
using ::remainderl;using ::remquol;
#line 719
using ::copysignl;using ::nanl;
#line 720
using ::nextafterl;using ::scalbnl;using ::scalblnl;
#line 721
using ::nexttowardl;using ::tgammal;
#line 723
using ::fpclassify;using ::signbit;
#line 724
using ::isfinite;using ::isinf;
#line 725
using ::isnan;using ::isnormal;
#line 726
using ::isgreater;using ::isgreaterequal;
#line 727
using ::isless;using ::islessequal;
#line 728
using ::islessgreater;using ::isunordered;
#line 729
}
#line 9034 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern "C" __declspec(dllimport) double __cdecl _hypot(double x, double y); 
#line 9035
extern "C" __declspec(dllimport) float __cdecl _hypotf(float x, float y); 
#line 9045 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline bool signbit(long double) throw(); 
#line 9046
extern "C" __declspec(dllimport) int _ldsign(long double); 
#line 9089 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline bool signbit(double) throw(); 
#line 9090
extern "C" __declspec(dllimport) int _dsign(double); 
#line 9134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline bool signbit(float) throw(); 
#line 9135
extern "C" __declspec(dllimport) int _fdsign(float); 
#line 9143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isinf(long double a); 
#line 9178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isinf(double a); 
#line 9216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isinf(float a); 
#line 9223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isnan(long double a); 
#line 9256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isnan(double a); 
#line 9292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isnan(float a); 
#line 9299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isfinite(long double a); 
#line 9336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isfinite(double a); 
#line 9372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static __inline bool isfinite(float a); 
#line 9380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
template< class T> extern T _Pow_int(T, int) throw(); 
#line 9381
extern inline __int64 abs(__int64) throw(); 
#line 9452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline long __cdecl abs(long) throw(); 
#line 9456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline float __cdecl abs(float) throw(); 
#line 9457
extern inline double __cdecl abs(double) throw(); 
#line 9458
extern inline float __cdecl fabs(float) throw(); 
#line 9459
extern inline float __cdecl ceil(float) throw(); 
#line 9460
extern inline float __cdecl floor(float) throw(); 
#line 9461
extern inline float __cdecl sqrt(float) throw(); 
#line 9462
extern inline float __cdecl pow(float, float) throw(); 
#line 9487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline float __cdecl pow(float, int) throw(); 
#line 9488
extern inline double __cdecl pow(double, int) throw(); 
#line 9491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline float __cdecl log(float) throw(); 
#line 9492
extern inline float __cdecl log10(float) throw(); 
#line 9493
extern inline float __cdecl fmod(float, float) throw(); 
#line 9494
extern inline float __cdecl modf(float, float *) throw(); 
#line 9495
extern inline float __cdecl exp(float) throw(); 
#line 9496
extern inline float __cdecl frexp(float, int *) throw(); 
#line 9497
extern inline float __cdecl ldexp(float, int) throw(); 
#line 9498
extern inline float __cdecl asin(float) throw(); 
#line 9499
extern inline float __cdecl sin(float) throw(); 
#line 9500
extern inline float __cdecl sinh(float) throw(); 
#line 9501
extern inline float __cdecl acos(float) throw(); 
#line 9502
extern inline float __cdecl cos(float) throw(); 
#line 9503
extern inline float __cdecl cosh(float) throw(); 
#line 9504
extern inline float __cdecl atan(float) throw(); 
#line 9505
extern inline float __cdecl atan2(float, float) throw(); 
#line 9506
extern inline float __cdecl tan(float) throw(); 
#line 9507
extern inline float __cdecl tanh(float) throw(); 
#line 9681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
extern inline float __cdecl logb(float) throw(); 
#line 9682
extern inline int __cdecl ilogb(float) throw(); 
#line 9683
extern float __cdecl scalbn(float, float) throw(); 
#line 9684
extern inline float __cdecl scalbln(float, long) throw(); 
#line 9685
extern inline float __cdecl exp2(float) throw(); 
#line 9686
extern inline float __cdecl expm1(float) throw(); 
#line 9687
extern inline float __cdecl log2(float) throw(); 
#line 9688
extern inline float __cdecl log1p(float) throw(); 
#line 9689
extern inline float __cdecl acosh(float) throw(); 
#line 9690
extern inline float __cdecl asinh(float) throw(); 
#line 9691
extern inline float __cdecl atanh(float) throw(); 
#line 9692
extern inline float __cdecl hypot(float, float) throw(); 
#line 9693
extern float __cdecl norm3d(float, float, float) throw(); 
#line 9694
extern float __cdecl norm4d(float, float, float, float) throw(); 
#line 9695
extern inline float __cdecl cbrt(float) throw(); 
#line 9696
extern inline float __cdecl erf(float) throw(); 
#line 9697
extern inline float __cdecl erfc(float) throw(); 
#line 9698
extern inline float __cdecl lgamma(float) throw(); 
#line 9699
extern inline float __cdecl tgamma(float) throw(); 
#line 9700
extern inline float __cdecl copysign(float, float) throw(); 
#line 9701
extern inline float __cdecl nextafter(float, float) throw(); 
#line 9702
extern inline float __cdecl remainder(float, float) throw(); 
#line 9703
extern inline float __cdecl remquo(float, float, int *) throw(); 
#line 9704
extern inline float __cdecl round(float) throw(); 
#line 9705
extern inline long __cdecl lround(float) throw(); 
#line 9706
extern inline __int64 __cdecl llround(float) throw(); 
#line 9707
extern inline float __cdecl trunc(float) throw(); 
#line 9708
extern inline float __cdecl rint(float) throw(); 
#line 9709
extern inline long __cdecl lrint(float) throw(); 
#line 9710
extern inline __int64 __cdecl llrint(float) throw(); 
#line 9711
extern inline float __cdecl nearbyint(float) throw(); 
#line 9712
extern inline float __cdecl fdim(float, float) throw(); 
#line 9713
extern inline float __cdecl fma(float, float, float) throw(); 
#line 9714
extern inline float __cdecl fmax(float, float) throw(); 
#line 9715
extern inline float __cdecl fmin(float, float) throw(); 
#line 9718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
static inline float exp10(float a); 
#line 9720
static inline float rsqrt(float a); 
#line 9722
static inline float rcbrt(float a); 
#line 9724
static inline float sinpi(float a); 
#line 9726
static inline float cospi(float a); 
#line 9728
static inline void sincospi(float a, float * sptr, float * cptr); 
#line 9730
static inline void sincos(float a, float * sptr, float * cptr); 
#line 9732
static inline float j0(float a); 
#line 9734
static inline float j1(float a); 
#line 9736
static inline float jn(int n, float a); 
#line 9738
static inline float y0(float a); 
#line 9740
static inline float y1(float a); 
#line 9742
static inline float yn(int n, float a); 
#line 9744
static inline float cyl_bessel_i0(float a); 
#line 9746
static inline float cyl_bessel_i1(float a); 
#line 9748
static inline float erfinv(float a); 
#line 9750
static inline float erfcinv(float a); 
#line 9752
static inline float normcdfinv(float a); 
#line 9754
static inline float normcdf(float a); 
#line 9756
static inline float erfcx(float a); 
#line 9758
static inline double copysign(double a, float b); 
#line 9760
static inline float copysign(float a, double b); 
#line 9762
static inline unsigned min(unsigned a, unsigned b); 
#line 9764
static inline unsigned min(int a, unsigned b); 
#line 9766
static inline unsigned min(unsigned a, int b); 
#line 9768
static inline long min(long a, long b); 
#line 9770
static inline unsigned long min(unsigned long a, unsigned long b); 
#line 9772
static inline unsigned long min(long a, unsigned long b); 
#line 9774
static inline unsigned long min(unsigned long a, long b); 
#line 9776
static inline __int64 min(__int64 a, __int64 b); 
#line 9778
static inline unsigned __int64 min(unsigned __int64 a, unsigned __int64 b); 
#line 9780
static inline unsigned __int64 min(__int64 a, unsigned __int64 b); 
#line 9782
static inline unsigned __int64 min(unsigned __int64 a, __int64 b); 
#line 9784
static inline float min(float a, float b); 
#line 9786
static inline double min(double a, double b); 
#line 9788
static inline double min(float a, double b); 
#line 9790
static inline double min(double a, float b); 
#line 9792
static inline unsigned max(unsigned a, unsigned b); 
#line 9794
static inline unsigned max(int a, unsigned b); 
#line 9796
static inline unsigned max(unsigned a, int b); 
#line 9798
static inline long max(long a, long b); 
#line 9800
static inline unsigned long max(unsigned long a, unsigned long b); 
#line 9802
static inline unsigned long max(long a, unsigned long b); 
#line 9804
static inline unsigned long max(unsigned long a, long b); 
#line 9806
static inline __int64 max(__int64 a, __int64 b); 
#line 9808
static inline unsigned __int64 max(unsigned __int64 a, unsigned __int64 b); 
#line 9810
static inline unsigned __int64 max(__int64 a, unsigned __int64 b); 
#line 9812
static inline unsigned __int64 max(unsigned __int64 a, __int64 b); 
#line 9814
static inline float max(float a, float b); 
#line 9816
static inline double max(double a, double b); 
#line 9818
static inline double max(float a, double b); 
#line 9820
static inline double max(double a, float b); 
#line 354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isinf(long double a) 
#line 355
{ 
#line 359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isinf< long double> (a); 
#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isinf(double a) 
#line 371
{ 
#line 375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isinf< double> (a); 
#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isinf(float a) 
#line 387
{ 
#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isinf< float> (a); 
#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isnan(long double a) 
#line 403
{ 
#line 407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isnan< long double> (a); 
#line 409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isnan(double a) 
#line 419
{ 
#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isnan< double> (a); 
#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isnan(float a) 
#line 435
{ 
#line 439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isnan< float> (a); 
#line 441 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isfinite(long double a) 
#line 451
{ 
#line 455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isfinite< long double> (a); 
#line 457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isfinite(double a) 
#line 467
{ 
#line 471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isfinite< double> (a); 
#line 473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static __inline bool isfinite(float a) 
#line 483
{ 
#line 487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return isfinite< float> (a); 
#line 489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
} 
#line 681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static inline float exp10(float a) 
#line 682
{ 
#line 683
return exp10f(a); 
#line 684
} 
#line 686
static inline float rsqrt(float a) 
#line 687
{ 
#line 688
return rsqrtf(a); 
#line 689
} 
#line 691
static inline float rcbrt(float a) 
#line 692
{ 
#line 693
return rcbrtf(a); 
#line 694
} 
#line 696
static inline float sinpi(float a) 
#line 697
{ 
#line 698
return sinpif(a); 
#line 699
} 
#line 701
static inline float cospi(float a) 
#line 702
{ 
#line 703
return cospif(a); 
#line 704
} 
#line 706
static inline void sincospi(float a, float *sptr, float *cptr) 
#line 707
{ 
#line 708
sincospif(a, sptr, cptr); 
#line 709
} 
#line 711
static inline void sincos(float a, float *sptr, float *cptr) 
#line 712
{ 
#line 713
sincosf(a, sptr, cptr); 
#line 714
} 
#line 716
static inline float j0(float a) 
#line 717
{ 
#line 718
return j0f(a); 
#line 719
} 
#line 721
static inline float j1(float a) 
#line 722
{ 
#line 723
return j1f(a); 
#line 724
} 
#line 726
static inline float jn(int n, float a) 
#line 727
{ 
#line 728
return jnf(n, a); 
#line 729
} 
#line 731
static inline float y0(float a) 
#line 732
{ 
#line 733
return y0f(a); 
#line 734
} 
#line 736
static inline float y1(float a) 
#line 737
{ 
#line 738
return y1f(a); 
#line 739
} 
#line 741
static inline float yn(int n, float a) 
#line 742
{ 
#line 743
return ynf(n, a); 
#line 744
} 
#line 746
static inline float cyl_bessel_i0(float a) 
#line 747
{ 
#line 748
return cyl_bessel_i0f(a); 
#line 749
} 
#line 751
static inline float cyl_bessel_i1(float a) 
#line 752
{ 
#line 753
return cyl_bessel_i1f(a); 
#line 754
} 
#line 756
static inline float erfinv(float a) 
#line 757
{ 
#line 758
return erfinvf(a); 
#line 759
} 
#line 761
static inline float erfcinv(float a) 
#line 762
{ 
#line 763
return erfcinvf(a); 
#line 764
} 
#line 766
static inline float normcdfinv(float a) 
#line 767
{ 
#line 768
return normcdfinvf(a); 
#line 769
} 
#line 771
static inline float normcdf(float a) 
#line 772
{ 
#line 773
return normcdff(a); 
#line 774
} 
#line 776
static inline float erfcx(float a) 
#line 777
{ 
#line 778
return erfcxf(a); 
#line 779
} 
#line 781
static inline double copysign(double a, float b) 
#line 782
{ 
#line 783
return copysign(a, (double)b); 
#line 784
} 
#line 786
static inline float copysign(float a, double b) 
#line 787
{ 
#line 788
return copysignf(a, (float)b); 
#line 789
} 
#line 791
static inline unsigned min(unsigned a, unsigned b) 
#line 792
{ 
#line 793
return umin(a, b); 
#line 794
} 
#line 796
static inline unsigned min(int a, unsigned b) 
#line 797
{ 
#line 798
return umin((unsigned)a, b); 
#line 799
} 
#line 801
static inline unsigned min(unsigned a, int b) 
#line 802
{ 
#line 803
return umin(a, (unsigned)b); 
#line 804
} 
#line 806
static inline long min(long a, long b) 
#line 807
{ 
#line 810
#pragma warning (disable: 4127)
#line 813 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 815
#pragma warning (default: 4127)
#line 817 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (long)min((int)a, (int)b); 
#line 818
} else { 
#line 819
return (long)llmin((__int64)a, (__int64)b); 
#line 820
}  
#line 821
} 
#line 823
static inline unsigned long min(unsigned long a, unsigned long b) 
#line 824
{ 
#line 826
#pragma warning (disable: 4127)
#line 828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 830
#pragma warning (default: 4127)
#line 832 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 833
} else { 
#line 834
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 835
}  
#line 836
} 
#line 838
static inline unsigned long min(long a, unsigned long b) 
#line 839
{ 
#line 841
#pragma warning (disable: 4127)
#line 843 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 845
#pragma warning (default: 4127)
#line 847 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 848
} else { 
#line 849
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 850
}  
#line 851
} 
#line 853
static inline unsigned long min(unsigned long a, long b) 
#line 854
{ 
#line 856
#pragma warning (disable: 4127)
#line 858 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 860
#pragma warning (default: 4127)
#line 862 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 863
} else { 
#line 864
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 865
}  
#line 866
} 
#line 868
static inline __int64 min(__int64 a, __int64 b) 
#line 869
{ 
#line 870
return llmin(a, b); 
#line 871
} 
#line 873
static inline unsigned __int64 min(unsigned __int64 a, unsigned __int64 b) 
#line 874
{ 
#line 875
return ullmin(a, b); 
#line 876
} 
#line 878
static inline unsigned __int64 min(__int64 a, unsigned __int64 b) 
#line 879
{ 
#line 880
return ullmin((unsigned __int64)a, b); 
#line 881
} 
#line 883
static inline unsigned __int64 min(unsigned __int64 a, __int64 b) 
#line 884
{ 
#line 885
return ullmin(a, (unsigned __int64)b); 
#line 886
} 
#line 888
static inline float min(float a, float b) 
#line 889
{ 
#line 890
return fminf(a, b); 
#line 891
} 
#line 893
static inline double min(double a, double b) 
#line 894
{ 
#line 895
return fmin(a, b); 
#line 896
} 
#line 898
static inline double min(float a, double b) 
#line 899
{ 
#line 900
return fmin((double)a, b); 
#line 901
} 
#line 903
static inline double min(double a, float b) 
#line 904
{ 
#line 905
return fmin(a, (double)b); 
#line 906
} 
#line 908
static inline unsigned max(unsigned a, unsigned b) 
#line 909
{ 
#line 910
return umax(a, b); 
#line 911
} 
#line 913
static inline unsigned max(int a, unsigned b) 
#line 914
{ 
#line 915
return umax((unsigned)a, b); 
#line 916
} 
#line 918
static inline unsigned max(unsigned a, int b) 
#line 919
{ 
#line 920
return umax(a, (unsigned)b); 
#line 921
} 
#line 923
static inline long max(long a, long b) 
#line 924
{ 
#line 927
#pragma warning (disable: 4127)
#line 929 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 931
#pragma warning (default: 4127)
#line 933 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (long)max((int)a, (int)b); 
#line 934
} else { 
#line 935
return (long)llmax((__int64)a, (__int64)b); 
#line 936
}  
#line 937
} 
#line 939
static inline unsigned long max(unsigned long a, unsigned long b) 
#line 940
{ 
#line 942
#pragma warning (disable: 4127)
#line 944 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 946
#pragma warning (default: 4127)
#line 948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 949
} else { 
#line 950
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 951
}  
#line 952
} 
#line 954
static inline unsigned long max(long a, unsigned long b) 
#line 955
{ 
#line 957
#pragma warning (disable: 4127)
#line 959 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 961
#pragma warning (default: 4127)
#line 963 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 964
} else { 
#line 965
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 966
}  
#line 967
} 
#line 969
static inline unsigned long max(unsigned long a, long b) 
#line 970
{ 
#line 972
#pragma warning (disable: 4127)
#line 974 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 976
#pragma warning (default: 4127)
#line 978 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 979
} else { 
#line 980
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 981
}  
#line 982
} 
#line 984
static inline __int64 max(__int64 a, __int64 b) 
#line 985
{ 
#line 986
return llmax(a, b); 
#line 987
} 
#line 989
static inline unsigned __int64 max(unsigned __int64 a, unsigned __int64 b) 
#line 990
{ 
#line 991
return ullmax(a, b); 
#line 992
} 
#line 994
static inline unsigned __int64 max(__int64 a, unsigned __int64 b) 
#line 995
{ 
#line 996
return ullmax((unsigned __int64)a, b); 
#line 997
} 
#line 999
static inline unsigned __int64 max(unsigned __int64 a, __int64 b) 
#line 1000
{ 
#line 1001
return ullmax(a, (unsigned __int64)b); 
#line 1002
} 
#line 1004
static inline float max(float a, float b) 
#line 1005
{ 
#line 1006
return fmaxf(a, b); 
#line 1007
} 
#line 1009
static inline double max(double a, double b) 
#line 1010
{ 
#line 1011
return fmax(a, b); 
#line 1012
} 
#line 1014
static inline double max(float a, double b) 
#line 1015
{ 
#line 1016
return fmax((double)a, b); 
#line 1017
} 
#line 1019
static inline double max(double a, float b) 
#line 1020
{ 
#line 1021
return fmax(a, (double)b); 
#line 1022
} 
#line 1028
#pragma warning(disable : 4211)
#line 1033 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
static inline int min(int a, int b) 
#line 1034
{ 
#line 1035
return (a < b) ? a : b; 
#line 1036
} 
#line 1038
static inline unsigned umin(unsigned a, unsigned b) 
#line 1039
{ 
#line 1040
return (a < b) ? a : b; 
#line 1041
} 
#line 1043
static inline __int64 llmin(__int64 a, __int64 b) 
#line 1044
{ 
#line 1045
return (a < b) ? a : b; 
#line 1046
} 
#line 1048
static inline unsigned __int64 ullmin(unsigned __int64 a, unsigned __int64 
#line 1049
b) 
#line 1050
{ 
#line 1051
return (a < b) ? a : b; 
#line 1052
} 
#line 1054
static inline int max(int a, int b) 
#line 1055
{ 
#line 1056
return (a > b) ? a : b; 
#line 1057
} 
#line 1059
static inline unsigned umax(unsigned a, unsigned b) 
#line 1060
{ 
#line 1061
return (a > b) ? a : b; 
#line 1062
} 
#line 1064
static inline __int64 llmax(__int64 a, __int64 b) 
#line 1065
{ 
#line 1066
return (a > b) ? a : b; 
#line 1067
} 
#line 1069
static inline unsigned __int64 ullmax(unsigned __int64 a, unsigned __int64 
#line 1070
b) 
#line 1071
{ 
#line 1072
return (a > b) ? a : b; 
#line 1073
} 
#line 1076
#pragma warning(default: 4211)
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_surface_types.h"
template< class T, int dim = 1> 
#line 78
struct surface : public surfaceReference { 
#line 81
surface() 
#line 82
{ 
#line 83
(::surfaceReference::channelDesc) = cudaCreateChannelDesc< T> (); 
#line 84
} 
#line 86
surface(::cudaChannelFormatDesc desc) 
#line 87
{ 
#line 88
(::surfaceReference::channelDesc) = desc; 
#line 89
} 
#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_surface_types.h"
}; 
#line 93
template< int dim> 
#line 94
struct surface< void, dim>  : public surfaceReference { 
#line 97
surface() 
#line 98
{ 
#line 99
(::surfaceReference::channelDesc) = cudaCreateChannelDesc< void> (); 
#line 100
} 
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_surface_types.h"
}; 
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
#line 78
struct texture : public textureReference { 
#line 81
texture(int norm = 0, ::cudaTextureFilterMode 
#line 82
fMode = cudaFilterModePoint, ::cudaTextureAddressMode 
#line 83
aMode = cudaAddressModeClamp) 
#line 84
{ 
#line 85
(::textureReference::normalized) = norm; 
#line 86
(::textureReference::filterMode) = fMode; 
#line 87
((::textureReference::addressMode)[0]) = aMode; 
#line 88
((::textureReference::addressMode)[1]) = aMode; 
#line 89
((::textureReference::addressMode)[2]) = aMode; 
#line 90
(::textureReference::channelDesc) = cudaCreateChannelDesc< T> (); 
#line 91
(::textureReference::sRGB) = 0; 
#line 92
} 
#line 94
texture(int norm, ::cudaTextureFilterMode 
#line 95
fMode, ::cudaTextureAddressMode 
#line 96
aMode, ::cudaChannelFormatDesc 
#line 97
desc) 
#line 98
{ 
#line 99
(::textureReference::normalized) = norm; 
#line 100
(::textureReference::filterMode) = fMode; 
#line 101
((::textureReference::addressMode)[0]) = aMode; 
#line 102
((::textureReference::addressMode)[1]) = aMode; 
#line 103
((::textureReference::addressMode)[2]) = aMode; 
#line 104
(::textureReference::channelDesc) = desc; 
#line 105
(::textureReference::sRGB) = 0; 
#line 106
} 
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_texture_types.h"
}; 
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
extern "C" {
#line 3230
}
#line 3238
static __inline int mulhi(int a, int b); 
#line 3240
static __inline unsigned mulhi(unsigned a, unsigned b); 
#line 3242
static __inline unsigned mulhi(int a, unsigned b); 
#line 3244
static __inline unsigned mulhi(unsigned a, int b); 
#line 3246
static __inline __int64 mul64hi(__int64 a, __int64 b); 
#line 3248
static __inline unsigned __int64 mul64hi(unsigned __int64 a, unsigned __int64 b); 
#line 3250
static __inline unsigned __int64 mul64hi(__int64 a, unsigned __int64 b); 
#line 3252
static __inline unsigned __int64 mul64hi(unsigned __int64 a, __int64 b); 
#line 3254
static __inline int float_as_int(float a); 
#line 3256
static __inline float int_as_float(int a); 
#line 3258
static __inline unsigned float_as_uint(float a); 
#line 3260
static __inline float uint_as_float(unsigned a); 
#line 3262
static __inline float saturate(float a); 
#line 3264
static __inline int mul24(int a, int b); 
#line 3266
static __inline unsigned umul24(unsigned a, unsigned b); 
#line 3268
__declspec(deprecated("Please use __trap() instead.")) static __inline void trap(); 
#line 3271
__declspec(deprecated("Please use __brkpt() instead.")) static __inline void brkpt(int c = 0); 
#line 3273
__declspec(deprecated("Please use __syncthreads() instead.")) static __inline void syncthreads(); 
#line 3275
__declspec(deprecated("Please use __prof_trigger() instead.")) static __inline void prof_trigger(int e); 
#line 3277
__declspec(deprecated("Please use __threadfence() instead.")) static __inline void threadfence(bool global = true); 
#line 3279
static __inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3281
static __inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3283
static __inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 3285
static __inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline int mulhi(int a, int b) 
#line 84
{int volatile ___ = 1;(void)a;(void)b;
#line 86
::exit(___);}
#if 0
#line 84
{ 
#line 85
return __mulhi(a, b); 
#line 86
} 
#endif
#line 88 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned mulhi(unsigned a, unsigned b) 
#line 89
{int volatile ___ = 1;(void)a;(void)b;
#line 91
::exit(___);}
#if 0
#line 89
{ 
#line 90
return __umulhi(a, b); 
#line 91
} 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned mulhi(int a, unsigned b) 
#line 94
{int volatile ___ = 1;(void)a;(void)b;
#line 96
::exit(___);}
#if 0
#line 94
{ 
#line 95
return __umulhi((unsigned)a, b); 
#line 96
} 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned mulhi(unsigned a, int b) 
#line 99
{int volatile ___ = 1;(void)a;(void)b;
#line 101
::exit(___);}
#if 0
#line 99
{ 
#line 100
return __umulhi(a, (unsigned)b); 
#line 101
} 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline __int64 mul64hi(__int64 a, __int64 b) 
#line 104
{int volatile ___ = 1;(void)a;(void)b;
#line 106
::exit(___);}
#if 0
#line 104
{ 
#line 105
return __mul64hi(a, b); 
#line 106
} 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(unsigned __int64 a, unsigned __int64 b) 
#line 109
{int volatile ___ = 1;(void)a;(void)b;
#line 111
::exit(___);}
#if 0
#line 109
{ 
#line 110
return __umul64hi(a, b); 
#line 111
} 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(__int64 a, unsigned __int64 b) 
#line 114
{int volatile ___ = 1;(void)a;(void)b;
#line 116
::exit(___);}
#if 0
#line 114
{ 
#line 115
return __umul64hi((unsigned __int64)a, b); 
#line 116
} 
#endif
#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(unsigned __int64 a, __int64 b) 
#line 119
{int volatile ___ = 1;(void)a;(void)b;
#line 121
::exit(___);}
#if 0
#line 119
{ 
#line 120
return __umul64hi(a, (unsigned __int64)b); 
#line 121
} 
#endif
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline int float_as_int(float a) 
#line 124
{int volatile ___ = 1;(void)a;
#line 126
::exit(___);}
#if 0
#line 124
{ 
#line 125
return __float_as_int(a); 
#line 126
} 
#endif
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline float int_as_float(int a) 
#line 129
{int volatile ___ = 1;(void)a;
#line 131
::exit(___);}
#if 0
#line 129
{ 
#line 130
return __int_as_float(a); 
#line 131
} 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned float_as_uint(float a) 
#line 134
{int volatile ___ = 1;(void)a;
#line 136
::exit(___);}
#if 0
#line 134
{ 
#line 135
return __float_as_uint(a); 
#line 136
} 
#endif
#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline float uint_as_float(unsigned a) 
#line 139
{int volatile ___ = 1;(void)a;
#line 141
::exit(___);}
#if 0
#line 139
{ 
#line 140
return __uint_as_float(a); 
#line 141
} 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline float saturate(float a) 
#line 143
{int volatile ___ = 1;(void)a;
#line 145
::exit(___);}
#if 0
#line 143
{ 
#line 144
return __saturatef(a); 
#line 145
} 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline int mul24(int a, int b) 
#line 148
{int volatile ___ = 1;(void)a;(void)b;
#line 150
::exit(___);}
#if 0
#line 148
{ 
#line 149
return __mul24(a, b); 
#line 150
} 
#endif
#line 152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned umul24(unsigned a, unsigned b) 
#line 153
{int volatile ___ = 1;(void)a;(void)b;
#line 155
::exit(___);}
#if 0
#line 153
{ 
#line 154
return __umul24(a, b); 
#line 155
} 
#endif
#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline void trap() 
#line 158
{int volatile ___ = 1;
#line 160
::exit(___);}
#if 0
#line 158
{ 
#line 159
__trap(); 
#line 160
} 
#endif
#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline void brkpt(int c) 
#line 164
{int volatile ___ = 1;(void)c;
#line 166
::exit(___);}
#if 0
#line 164
{ 
#line 165
__brkpt(c); 
#line 166
} 
#endif
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline void syncthreads() 
#line 169
{int volatile ___ = 1;
#line 171
::exit(___);}
#if 0
#line 169
{ 
#line 170
__syncthreads(); 
#line 171
} 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline void prof_trigger(int e) 
#line 174
{int volatile ___ = 1;(void)e;
#line 191
::exit(___);}
#if 0
#line 174
{ 
#line 175
if (e == 0) { __prof_trigger(0); } else { 
#line 176
if (e == 1) { __prof_trigger(1); } else { 
#line 177
if (e == 2) { __prof_trigger(2); } else { 
#line 178
if (e == 3) { __prof_trigger(3); } else { 
#line 179
if (e == 4) { __prof_trigger(4); } else { 
#line 180
if (e == 5) { __prof_trigger(5); } else { 
#line 181
if (e == 6) { __prof_trigger(6); } else { 
#line 182
if (e == 7) { __prof_trigger(7); } else { 
#line 183
if (e == 8) { __prof_trigger(8); } else { 
#line 184
if (e == 9) { __prof_trigger(9); } else { 
#line 185
if (e == 10) { __prof_trigger(10); } else { 
#line 186
if (e == 11) { __prof_trigger(11); } else { 
#line 187
if (e == 12) { __prof_trigger(12); } else { 
#line 188
if (e == 13) { __prof_trigger(13); } else { 
#line 189
if (e == 14) { __prof_trigger(14); } else { 
#line 190
if (e == 15) { __prof_trigger(15); }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  
#line 191
} 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline void threadfence(bool global) 
#line 194
{int volatile ___ = 1;(void)global;
#line 196
::exit(___);}
#if 0
#line 194
{ 
#line 195
global ? __threadfence() : __threadfence_block(); 
#line 196
} 
#endif
#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline int float2int(float a, cudaRoundMode mode) 
#line 199
{int volatile ___ = 1;(void)a;(void)mode;
#line 204
::exit(___);}
#if 0
#line 199
{ 
#line 200
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
#line 204
} 
#endif
#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline unsigned float2uint(float a, cudaRoundMode mode) 
#line 207
{int volatile ___ = 1;(void)a;(void)mode;
#line 212
::exit(___);}
#if 0
#line 207
{ 
#line 208
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
#line 212
} 
#endif
#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline float int2float(int a, cudaRoundMode mode) 
#line 215
{int volatile ___ = 1;(void)a;(void)mode;
#line 220
::exit(___);}
#if 0
#line 215
{ 
#line 216
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
#line 220
} 
#endif
#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline float uint2float(unsigned a, cudaRoundMode mode) 
#line 223
{int volatile ___ = 1;(void)a;(void)mode;
#line 228
::exit(___);}
#if 0
#line 223
{ 
#line 224
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
#line 228
} 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
extern "C" {
#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
}
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicAdd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicExch(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicCAS(unsigned __int64 *address, unsigned __int64 compare, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
static __inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 80 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"
extern "C" {
#line 1134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"
}
#line 1143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"
static __inline double fma(double a, double b, double c, cudaRoundMode mode); 
#line 1145
static __inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1147
static __inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1149
static __inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1151
static __inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1153
static __inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1155
static __inline __int64 double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1157
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1159
static __inline double ll2double(__int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1161
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1163
static __inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 1165
static __inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 1167
static __inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
#line 85 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double fma(double a, double b, double c, cudaRoundMode mode) 
#line 86
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
#line 91
::exit(___);}
#if 0
#line 86
{ 
#line 87
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
#line 91
} 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double dmul(double a, double b, cudaRoundMode mode) 
#line 94
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
#line 99
::exit(___);}
#if 0
#line 94
{ 
#line 95
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
#line 99
} 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double dadd(double a, double b, cudaRoundMode mode) 
#line 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
#line 107
::exit(___);}
#if 0
#line 102
{ 
#line 103
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
#line 107
} 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double dsub(double a, double b, cudaRoundMode mode) 
#line 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
#line 115
::exit(___);}
#if 0
#line 110
{ 
#line 111
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
#line 115
} 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline int double2int(double a, cudaRoundMode mode) 
#line 118
{int volatile ___ = 1;(void)a;(void)mode;
#line 123
::exit(___);}
#if 0
#line 118
{ 
#line 119
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
#line 123
} 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline unsigned double2uint(double a, cudaRoundMode mode) 
#line 126
{int volatile ___ = 1;(void)a;(void)mode;
#line 131
::exit(___);}
#if 0
#line 126
{ 
#line 127
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
#line 131
} 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline __int64 double2ll(double a, cudaRoundMode mode) 
#line 134
{int volatile ___ = 1;(void)a;(void)mode;
#line 139
::exit(___);}
#if 0
#line 134
{ 
#line 135
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
#line 139
} 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode) 
#line 142
{int volatile ___ = 1;(void)a;(void)mode;
#line 147
::exit(___);}
#if 0
#line 142
{ 
#line 143
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
#line 147
} 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double ll2double(__int64 a, cudaRoundMode mode) 
#line 150
{int volatile ___ = 1;(void)a;(void)mode;
#line 155
::exit(___);}
#if 0
#line 150
{ 
#line 151
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
#line 155
} 
#endif
#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode) 
#line 158
{int volatile ___ = 1;(void)a;(void)mode;
#line 163
::exit(___);}
#if 0
#line 158
{ 
#line 159
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
#line 163
} 
#endif
#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double int2double(int a, cudaRoundMode mode) 
#line 166
{int volatile ___ = 1;(void)a;(void)mode;
#line 168
::exit(___);}
#if 0
#line 166
{ 
#line 167
return (double)a; 
#line 168
} 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double uint2double(unsigned a, cudaRoundMode mode) 
#line 171
{int volatile ___ = 1;(void)a;(void)mode;
#line 173
::exit(___);}
#if 0
#line 171
{ 
#line 172
return (double)a; 
#line 173
} 
#endif
#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"
static __inline double float2double(float a, cudaRoundMode mode) 
#line 176
{int volatile ___ = 1;(void)a;(void)mode;
#line 178
::exit(___);}
#if 0
#line 176
{ 
#line 177
return (double)a; 
#line 178
} 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"
static __inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMin(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMax(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMin(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMax(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicAnd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicOr(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicXor(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 308
{ } 
#endif
#line 311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 311
{ } 
#endif
#line 314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 314
{ } 
#endif
#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 317
{ } 
#endif
#line 320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 320
{ } 
#endif
#line 323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 323
{ } 
#endif
#line 326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 326
{ } 
#endif
#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 329
{ } 
#endif
#line 332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 332
{ } 
#endif
#line 335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 335
{ } 
#endif
#line 338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 338
{ } 
#endif
#line 341 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 341
{ } 
#endif
#line 344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 344
{ } 
#endif
#line 347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 347
{ } 
#endif
#line 350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 350
{ } 
#endif
#line 353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 353
{ } 
#endif
#line 356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 356
{ } 
#endif
#line 359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 359
{ } 
#endif
#line 362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 362
{ } 
#endif
#line 365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 365
{ } 
#endif
#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 368
{ } 
#endif
#line 371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 371
{ } 
#endif
#line 374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 374
{ } 
#endif
#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 377
{ } 
#endif
#line 380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 380
{ } 
#endif
#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 383
{ } 
#endif
#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 386
{ } 
#endif
#line 389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 389
{ } 
#endif
#line 392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 392
{ } 
#endif
#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 395
{ } 
#endif
#line 398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 398
{ } 
#endif
#line 401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 401
{ } 
#endif
#line 404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 404
{ } 
#endif
#line 407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 407
{ } 
#endif
#line 410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 410
{ } 
#endif
#line 413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 413
{ } 
#endif
#line 416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 416
{ } 
#endif
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 419
{ } 
#endif
#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 422
{ } 
#endif
#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 425
{ } 
#endif
#line 428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 428
{ } 
#endif
#line 431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
#line 432
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 432
{ } 
#endif
#line 435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
#line 436
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 436
{ } 
#endif
#line 439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_block(unsigned __int64 *address, unsigned __int64 
#line 440
compare, unsigned __int64 
#line 441
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 441
{ } 
#endif
#line 444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_system(unsigned __int64 *address, unsigned __int64 
#line 445
compare, unsigned __int64 
#line 446
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 446
{ } 
#endif
#line 449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 449
{ } 
#endif
#line 452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 452
{ } 
#endif
#line 455 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 455
{ } 
#endif
#line 458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 458
{ } 
#endif
#line 461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 461
{ } 
#endif
#line 464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 464
{ } 
#endif
#line 467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 467
{ } 
#endif
#line 470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 470
{ } 
#endif
#line 473 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 473
{ } 
#endif
#line 476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 476
{ } 
#endif
#line 479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 479
{ } 
#endif
#line 482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 482
{ } 
#endif
#line 485 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 485
{ } 
#endif
#line 488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 488
{ } 
#endif
#line 491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 491
{ } 
#endif
#line 494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 494
{ } 
#endif
#line 497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 497
{ } 
#endif
#line 500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 500
{ } 
#endif
#line 503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 503
{ } 
#endif
#line 506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 506
{ } 
#endif
#line 509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 509
{ } 
#endif
#line 512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 512
{ } 
#endif
#line 515 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 515
{ } 
#endif
#line 518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 518
{ } 
#endif
#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
extern "C" {
#line 1466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
}
#line 1475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
static __inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1475
{ } 
#endif
#line 1477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
static __inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1477
{ } 
#endif
#line 1479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1479
{ } 
#endif
#line 1481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1481
{ } 
#endif
#line 1486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1486
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl(__int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 122
{ } 
#endif
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl(unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_up(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_up(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_down(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_down(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_xor(__int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_xor(unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 152
{ } 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 158
{ } 
#endif
#line 160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 160
{ } 
#endif
#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 157
{ } 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 161
{ } 
#endif
#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldca(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 174
{ } 
#endif
#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 175
{ } 
#endif
#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldca(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 186
{ } 
#endif
#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 192
{ } 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 197
{ } 
#endif
#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 198
{ } 
#endif
#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 200
{ } 
#endif
#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 201
{ } 
#endif
#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 202
{ } 
#endif
#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 203
{ } 
#endif
#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcs(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 204
{ } 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 205
{ } 
#endif
#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 206
{ } 
#endif
#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 207
{ } 
#endif
#line 208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 208
{ } 
#endif
#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 209
{ } 
#endif
#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 210
{ } 
#endif
#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 211
{ } 
#endif
#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 213
{ } 
#endif
#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 214
{ } 
#endif
#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 215
{ } 
#endif
#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcs(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 217
{ } 
#endif
#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 218
{ } 
#endif
#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 220
{ } 
#endif
#line 221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 221
{ } 
#endif
#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 222
{ } 
#endif
#line 223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 223
{ } 
#endif
#line 225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 225
{ } 
#endif
#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 226
{ } 
#endif
#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 227
{ } 
#endif
#line 228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 228
{ } 
#endif
#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 229
{ } 
#endif
#line 236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 236
{ } 
#endif
#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 237
{ } 
#endif
#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 240
{ } 
#endif
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 241
{ } 
#endif
#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 91
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 101
surf1Dread(T *res, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 102
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 114
surf1Dread(::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 115
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 126
surf1Dread(T *res, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 127
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 261
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 262
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 274
surf2Dread(::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 275
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 286
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 287
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
#line 291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 423
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 424
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
#line 433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 435 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 436
surf3Dread(::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 437
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
#line 445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 448
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 449
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
#line 453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 583
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 584
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
#line 593 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 593 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 595 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 596
surf1DLayeredread(::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 597
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
#line 605 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 597 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 605 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 607 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 608
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 609
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
#line 613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 609 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 769
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 770
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
#line 779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 781 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 782
surf2DLayeredread(::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 783
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
#line 791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 794
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 795
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
#line 799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 795 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 919 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 920
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 921
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
#line 930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 921 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 932 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 933
surfCubemapread(::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 934
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 934 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 944 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 945
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 946
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
#line 950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 946 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1070 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1071
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1072
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
#line 1081 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1081 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1083 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline T 
#line 1084
surfCubemapLayeredread(::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1085
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
#line 1093 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1085 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1093 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1095 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1096
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1097
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
#line 1101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1097 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1233
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1234
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
#line 1254
::exit(___);}
#if 0
#line 1234
{ 
#line 1254
} 
#endif
#line 1256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1257
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1258
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
#line 1262
::exit(___);}
#if 0
#line 1258
{ 
#line 1262
} 
#endif
#line 1377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1378
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1379
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
#line 1399
::exit(___);}
#if 0
#line 1379
{ 
#line 1399
} 
#endif
#line 1401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1402
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
#line 1407
::exit(___);}
#if 0
#line 1403
{ 
#line 1407
} 
#endif
#line 1520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1521
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1522
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
#line 1542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1545
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1546
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
#line 1550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 1667
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1668
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
#line 1688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 1691
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1692
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
#line 1696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1822 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1823
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1824
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
#line 1844 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1844 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1847
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1848
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
#line 1852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1848 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1958 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1959
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1960
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
#line 1980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1960 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 1982 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> __forceinline void 
#line 1983
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 1984
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
#line 1988 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 1984 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 1988 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 2093 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 2094
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 2095
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
#line 2115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 2095 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 2115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 2117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 2118
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
#line 2119
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
#line 2123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
::exit(___);}
#if 0
#line 2119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
{ 
#line 2123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
} 
#endif
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 71
tex1Dfetch(texture< T, 1, cudaReadModeElementType> , int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 71
{ } 
#endif
#line 73 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> 
#line 74
struct __nv_tex_rmnf_ret { }; 
#line 76
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
#line 77
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
#line 78
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
#line 79
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
#line 80
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
#line 81
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
#line 82
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
#line 83
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
#line 84
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
#line 85
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
#line 86
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
#line 87
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
#line 88
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
#line 89
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
#line 90
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 91
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
#line 92
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 94
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 95
tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat> , int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 216
tex1D(texture< T, 1, cudaReadModeElementType> , float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 219
tex1D(texture< T, 1, cudaReadModeNormalizedFloat> , float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 346
tex2D(texture< T, 2, cudaReadModeElementType> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 346
{ } 
#endif
#line 348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 349
tex2D(texture< T, 2, cudaReadModeNormalizedFloat> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 349
{ } 
#endif
#line 475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 476
tex1DLayered(texture< T, 241, cudaReadModeElementType> , float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 476
{ } 
#endif
#line 478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 479
tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat> , float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 479
{ } 
#endif
#line 603 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 604
tex2DLayered(texture< T, 242, cudaReadModeElementType> , float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 604
{ } 
#endif
#line 606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 607
tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 607
{ } 
#endif
#line 735 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 736
tex3D(texture< T, 3, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 736
{ } 
#endif
#line 738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 739
tex3D(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 739
{ } 
#endif
#line 864 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 865
texCubemap(texture< T, 12, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 865
{ } 
#endif
#line 867 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 868
texCubemap(texture< T, 12, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 868
{ } 
#endif
#line 992 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 993
texCubemapLayered(texture< T, 252, cudaReadModeElementType> , float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 993
{ } 
#endif
#line 995 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 996
texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat> , float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 996
{ } 
#endif
#line 1121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> 
#line 1122
struct __nv_tex2dgather_ret { }; 
#line 1123
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
#line 1124
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
#line 1125
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
#line 1126
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
#line 1127
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
#line 1128
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
#line 1129
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
#line 1130
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
#line 1131
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
#line 1132
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
#line 1133
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
#line 1135
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
#line 1136
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
#line 1137
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
#line 1138
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
#line 1139
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
#line 1140
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
#line 1141
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
#line 1142
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
#line 1143
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
#line 1144
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
#line 1146
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
#line 1147
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
#line 1148
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
#line 1149
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
#line 1150
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
#line 1151
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
#line 1152
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
#line 1153
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
#line 1154
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
#line 1155
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
#line 1157
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
#line 1158
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
#line 1159
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
#line 1160
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
#line 1161
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
#line 1163
template< class T> static typename __nv_tex2dgather_ret< T> ::type 
#line 1164
tex2Dgather(texture< T, 2, cudaReadModeElementType> , float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1164
{ } 
#endif
#line 1166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static ::float4 
#line 1167
tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1167
{ } 
#endif
#line 1232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1233
tex1DLod(texture< T, 1, cudaReadModeElementType> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1233
{ } 
#endif
#line 1235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1236
tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1236
{ } 
#endif
#line 1360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1361
tex2DLod(texture< T, 2, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1361
{ } 
#endif
#line 1363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1364
tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1364
{ } 
#endif
#line 1484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1485
tex1DLayeredLod(texture< T, 241, cudaReadModeElementType> , float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1485
{ } 
#endif
#line 1487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1488
tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat> , float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1488
{ } 
#endif
#line 1612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1613
tex2DLayeredLod(texture< T, 242, cudaReadModeElementType> , float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1613
{ } 
#endif
#line 1615 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1616
tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1616
{ } 
#endif
#line 1740 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1741
tex3DLod(texture< T, 3, cudaReadModeElementType> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1741
{ } 
#endif
#line 1743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1744
tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1744
{ } 
#endif
#line 1868 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1869
texCubemapLod(texture< T, 12, cudaReadModeElementType> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1869
{ } 
#endif
#line 1871 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 1872
texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1872
{ } 
#endif
#line 1996 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 1997
texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType> , float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1997
{ } 
#endif
#line 1999 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2000
texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat> , float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2000
{ } 
#endif
#line 2124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 2125
tex1DGrad(texture< T, 1, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2125
{ } 
#endif
#line 2127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2128
tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2128
{ } 
#endif
#line 2252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 2253
tex2DGrad(texture< T, 2, cudaReadModeElementType> , float, float, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2253
{ } 
#endif
#line 2255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2256
tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2256
{ } 
#endif
#line 2380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 2381
tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType> , float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2381
{ } 
#endif
#line 2383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2384
tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat> , float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2384
{ } 
#endif
#line 2509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 2510
tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType> , float, float, int, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2510
{ } 
#endif
#line 2512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2513
tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2513
{ } 
#endif
#line 2637 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static T 
#line 2638
tex3DGrad(texture< T, 3, cudaReadModeElementType> , float, float, float, ::float4, ::float4) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2638
{ } 
#endif
#line 2640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"
template< class T> static typename __nv_tex_rmnf_ret< T> ::type 
#line 2641
tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float, ::float4, ::float4) {int volatile ___ = 1;::exit(___);}
#if 0
#line 2641
{ } 
#endif
#line 67 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 68
tex1Dfetch(T *, ::cudaTextureObject_t, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 68
{ } 
#endif
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 122
tex1Dfetch(::cudaTextureObject_t texObject, int x) 
#line 123
{int volatile ___ = 1;(void)texObject;(void)x;
#line 127
::exit(___);}
#if 0
#line 123
{ 
#line 124
T ret; 
#line 125
tex1Dfetch(&ret, texObject, x); 
#line 126
return ret; 
#line 127
} 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 136
tex1D(T *, ::cudaTextureObject_t, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 191
tex1D(::cudaTextureObject_t texObject, float x) 
#line 192
{int volatile ___ = 1;(void)texObject;(void)x;
#line 196
::exit(___);}
#if 0
#line 192
{ 
#line 193
T ret; 
#line 194
tex1D(&ret, texObject, x); 
#line 195
return ret; 
#line 196
} 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 206
tex2D(T *, ::cudaTextureObject_t, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 206
{ } 
#endif
#line 258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 259
tex2D(::cudaTextureObject_t texObject, float x, float y) 
#line 260
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
#line 264
::exit(___);}
#if 0
#line 260
{ 
#line 261
T ret; 
#line 262
tex2D(&ret, texObject, x, y); 
#line 263
return ret; 
#line 264
} 
#endif
#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 273
tex3D(T *, ::cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 273
{ } 
#endif
#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 326
tex3D(::cudaTextureObject_t texObject, float x, float y, float z) 
#line 327
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
#line 331
::exit(___);}
#if 0
#line 327
{ 
#line 328
T ret; 
#line 329
tex3D(&ret, texObject, x, y, z); 
#line 330
return ret; 
#line 331
} 
#endif
#line 340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 341
tex1DLayered(T *, ::cudaTextureObject_t, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 341
{ } 
#endif
#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 394
tex1DLayered(::cudaTextureObject_t texObject, float x, int layer) 
#line 395
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
#line 399
::exit(___);}
#if 0
#line 395
{ 
#line 396
T ret; 
#line 397
tex1DLayered(&ret, texObject, x, layer); 
#line 398
return ret; 
#line 399
} 
#endif
#line 408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 409
tex2DLayered(T *, ::cudaTextureObject_t, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 409
{ } 
#endif
#line 461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 462
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer) 
#line 463
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
#line 467
::exit(___);}
#if 0
#line 463
{ 
#line 464
T ret; 
#line 465
tex2DLayered(&ret, texObject, x, y, layer); 
#line 466
return ret; 
#line 467
} 
#endif
#line 476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 477
texCubemap(T *, ::cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 477
{ } 
#endif
#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 530
texCubemap(::cudaTextureObject_t texObject, float x, float y, float z) 
#line 531
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
#line 535
::exit(___);}
#if 0
#line 531
{ 
#line 532
T ret; 
#line 533
texCubemap(&ret, texObject, x, y, z); 
#line 534
return ret; 
#line 535
} 
#endif
#line 544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 545
texCubemapLayered(T *, ::cudaTextureObject_t, float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
#line 545
{ } 
#endif
#line 598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 599
texCubemapLayered(::cudaTextureObject_t texObject, float x, float y, float z, int layer) 
#line 600
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
#line 604
::exit(___);}
#if 0
#line 600
{ 
#line 601
T ret; 
#line 602
texCubemapLayered(&ret, texObject, x, y, z, layer); 
#line 603
return ret; 
#line 604
} 
#endif
#line 613 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 614
tex2Dgather(T *, ::cudaTextureObject_t, float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
#line 614
{ } 
#endif
#line 660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 661
tex2Dgather(::cudaTextureObject_t to, float x, float y, int comp = 0) 
#line 662
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
#line 666
::exit(___);}
#if 0
#line 662
{ 
#line 663
T ret; 
#line 664
tex2Dgather(&ret, to, x, y, comp); 
#line 665
return ret; 
#line 666
} 
#endif
#line 675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 676
tex1DLod(T *, ::cudaTextureObject_t, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 676
{ } 
#endif
#line 728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 729
tex1DLod(::cudaTextureObject_t texObject, float x, float level) 
#line 730
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
#line 734
::exit(___);}
#if 0
#line 730
{ 
#line 731
T ret; 
#line 732
tex1DLod(&ret, texObject, x, level); 
#line 733
return ret; 
#line 734
} 
#endif
#line 743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 744
tex2DLod(T *, ::cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 744
{ } 
#endif
#line 797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 798
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level) 
#line 799
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
#line 803
::exit(___);}
#if 0
#line 799
{ 
#line 800
T ret; 
#line 801
tex2DLod(&ret, texObject, x, y, level); 
#line 802
return ret; 
#line 803
} 
#endif
#line 812 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 813
tex3DLod(T *, ::cudaTextureObject_t, float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 813
{ } 
#endif
#line 865 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 866
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) 
#line 867
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
#line 871
::exit(___);}
#if 0
#line 867
{ 
#line 868
T ret; 
#line 869
tex3DLod(&ret, texObject, x, y, z, level); 
#line 870
return ret; 
#line 871
} 
#endif
#line 879 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 880
tex1DLayeredLod(T *, ::cudaTextureObject_t, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 880
{ } 
#endif
#line 932 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 933
tex1DLayeredLod(::cudaTextureObject_t texObject, float x, int layer, float level) 
#line 934
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
#line 938
::exit(___);}
#if 0
#line 934
{ 
#line 935
T ret; 
#line 936
tex1DLayeredLod(&ret, texObject, x, layer, level); 
#line 937
return ret; 
#line 938
} 
#endif
#line 947 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 948
tex2DLayeredLod(T *, ::cudaTextureObject_t, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 948
{ } 
#endif
#line 1000 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1001
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level) 
#line 1002
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
#line 1006
::exit(___);}
#if 0
#line 1002
{ 
#line 1003
T ret; 
#line 1004
tex2DLayeredLod(&ret, texObject, x, y, layer, level); 
#line 1005
return ret; 
#line 1006
} 
#endif
#line 1014 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1015
texCubemapLod(T *, ::cudaTextureObject_t, float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1015
{ } 
#endif
#line 1067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1068
texCubemapLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) 
#line 1069
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
#line 1073
::exit(___);}
#if 0
#line 1069
{ 
#line 1070
T ret; 
#line 1071
texCubemapLod(&ret, texObject, x, y, z, level); 
#line 1072
return ret; 
#line 1073
} 
#endif
#line 1081 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1082
texCubemapLayeredLod(T *, ::cudaTextureObject_t, float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1082
{ } 
#endif
#line 1134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1135
texCubemapLayeredLod(::cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
#line 1136
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
#line 1140
::exit(___);}
#if 0
#line 1136
{ 
#line 1137
T ret; 
#line 1138
texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level); 
#line 1139
return ret; 
#line 1140
} 
#endif
#line 1148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1149
tex1DGrad(T *, ::cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1149
{ } 
#endif
#line 1202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1203
tex1DGrad(::cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
#line 1204
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
#line 1208
::exit(___);}
#if 0
#line 1204
{ 
#line 1205
T ret; 
#line 1206
tex1DGrad(&ret, texObject, x, dPdx, dPdy); 
#line 1207
return ret; 
#line 1208
} 
#endif
#line 1216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1217
tex2DGrad(T *, ::cudaTextureObject_t, float, float, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1217
{ } 
#endif
#line 1269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1270
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy) 
#line 1271
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
#line 1275
::exit(___);}
#if 0
#line 1271
{ 
#line 1272
T ret; 
#line 1273
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy); 
#line 1274
return ret; 
#line 1275
} 
#endif
#line 1283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1284
tex3DGrad(T *, ::cudaTextureObject_t, float, float, float, ::float4, ::float4) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1284
{ } 
#endif
#line 1336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1337
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) 
#line 1338
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
#line 1342
::exit(___);}
#if 0
#line 1338
{ 
#line 1339
T ret; 
#line 1340
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
#line 1341
return ret; 
#line 1342
} 
#endif
#line 1350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1351
tex1DLayeredGrad(T *, ::cudaTextureObject_t, float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1351
{ } 
#endif
#line 1404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1405
tex1DLayeredGrad(::cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
#line 1406
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
#line 1410
::exit(___);}
#if 0
#line 1406
{ 
#line 1407
T ret; 
#line 1408
tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy); 
#line 1409
return ret; 
#line 1410
} 
#endif
#line 1418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static void 
#line 1419
tex2DLayeredGrad(T *, ::cudaTextureObject_t, float, float, int, ::float2, ::float2) {int volatile ___ = 1;::exit(___);}
#if 0
#line 1419
{ } 
#endif
#line 1471 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 1472
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) 
#line 1473
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
#line 1477
::exit(___);}
#if 0
#line 1473
{ 
#line 1474
T ret; 
#line 1475
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy); 
#line 1476
return ret; 
#line 1477
} 
#endif
#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 69
surf1Dread(T *, ::cudaSurfaceObject_t, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 69
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 112
surf1Dread(::cudaSurfaceObject_t surfObject, int x, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 113
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
#line 119
::exit(___);}
#if 0
#line 113
{ 
#line 119
} 
#endif
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 129
surf2Dread(T *, ::cudaSurfaceObject_t, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 173
surf2Dread(::cudaSurfaceObject_t surfObject, int x, int y, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 174
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
#line 180
::exit(___);}
#if 0
#line 174
{ 
#line 180
} 
#endif
#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 190
surf3Dread(T *, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 232
surf3Dread(::cudaSurfaceObject_t surfObject, int x, int y, int z, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 233
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
#line 239
::exit(___);}
#if 0
#line 233
{ 
#line 239
} 
#endif
#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 248
surf1DLayeredread(T *, ::cudaSurfaceObject_t, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 248
{ } 
#endif
#line 290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 291
surf1DLayeredread(::cudaSurfaceObject_t surfObject, int x, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 292
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
#line 298
::exit(___);}
#if 0
#line 292
{ 
#line 298
} 
#endif
#line 306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 307
surf2DLayeredread(T *, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 307
{ } 
#endif
#line 348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 349
surf2DLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 350
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
#line 356
::exit(___);}
#if 0
#line 350
{ 
#line 356
} 
#endif
#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 365
surfCubemapread(T *, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 365
{ } 
#endif
#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 407
surfCubemapread(::cudaSurfaceObject_t surfObject, int x, int y, int face, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 408
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
#line 414
::exit(___);}
#if 0
#line 408
{ 
#line 414
} 
#endif
#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 423
surfCubemapLayeredread(T *, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 423
{ } 
#endif
#line 464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 465
surfCubemapLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layerface, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
#line 466
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
#line 472
::exit(___);}
#if 0
#line 466
{ 
#line 472
} 
#endif
#line 480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 481
surf1Dwrite(T, ::cudaSurfaceObject_t, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 481
{ } 
#endif
#line 528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 529
surf2Dwrite(T, ::cudaSurfaceObject_t, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 529
{ } 
#endif
#line 576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 577
surf3Dwrite(T, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 577
{ } 
#endif
#line 626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 627
surf1DLayeredwrite(T, ::cudaSurfaceObject_t, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 627
{ } 
#endif
#line 675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 676
surf2DLayeredwrite(T, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 676
{ } 
#endif
#line 723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 724
surfCubemapwrite(T, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 724
{ } 
#endif
#line 771 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"
template< class T> static void 
#line 772
surfCubemapLayeredwrite(T, ::cudaSurfaceObject_t, int, int, int, ::cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
#line 772
{ } 
#endif
#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_launch_parameters.h"
extern "C" {
#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_launch_parameters.h"
extern const uint3 __device_builtin_variable_threadIdx; 
#line 72
extern const uint3 __device_builtin_variable_blockIdx; 
#line 73
extern const dim3 __device_builtin_variable_blockDim; 
#line 74
extern const dim3 __device_builtin_variable_gridDim; 
#line 75
extern const int __device_builtin_variable_warpSize; 
#line 80
}
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 184
cudaLaunchKernel(const T *
#line 185
func, ::dim3 
#line 186
gridDim, ::dim3 
#line 187
blockDim, void **
#line 188
args, ::size_t 
#line 189
sharedMem = 0, ::cudaStream_t 
#line 190
stream = 0) 
#line 192
{ 
#line 193
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 194
} 
#line 221
template< class T> static __inline ::cudaError_t 
#line 222
cudaSetupArgument(T 
#line 223
arg, ::size_t 
#line 224
offset) 
#line 226
{ 
#line 227
return ::cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
#line 228
} 
#line 260
static __inline cudaError_t cudaEventCreate(cudaEvent_t *
#line 261
event, unsigned 
#line 262
flags) 
#line 264
{ 
#line 265
return ::cudaEventCreateWithFlags(event, flags); 
#line 266
} 
#line 323
static __inline cudaError_t cudaMallocHost(void **
#line 324
ptr, size_t 
#line 325
size, unsigned 
#line 326
flags) 
#line 328
{ 
#line 329
return ::cudaHostAlloc(ptr, size, flags); 
#line 330
} 
#line 332
template< class T> static __inline ::cudaError_t 
#line 333
cudaHostAlloc(T **
#line 334
ptr, ::size_t 
#line 335
size, unsigned 
#line 336
flags) 
#line 338
{ 
#line 339
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
#line 340
} 
#line 342
template< class T> static __inline ::cudaError_t 
#line 343
cudaHostGetDevicePointer(T **
#line 344
pDevice, void *
#line 345
pHost, unsigned 
#line 346
flags) 
#line 348
{ 
#line 349
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
#line 350
} 
#line 449
template< class T> static __inline ::cudaError_t 
#line 450
cudaMallocManaged(T **
#line 451
devPtr, ::size_t 
#line 452
size, unsigned 
#line 453
flags = 1) 
#line 455
{ 
#line 456
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
#line 457
} 
#line 528
template< class T> static __inline ::cudaError_t 
#line 529
cudaStreamAttachMemAsync(::cudaStream_t 
#line 530
stream, T *
#line 531
devPtr, ::size_t 
#line 532
length = 0, unsigned 
#line 533
flags = 4) 
#line 535
{ 
#line 536
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
#line 537
} 
#line 539
template< class T> __inline ::cudaError_t 
#line 540
cudaMalloc(T **
#line 541
devPtr, ::size_t 
#line 542
size) 
#line 544
{ 
#line 545
return ::cudaMalloc((void **)((void *)devPtr), size); 
#line 546
} 
#line 548
template< class T> static __inline ::cudaError_t 
#line 549
cudaMallocHost(T **
#line 550
ptr, ::size_t 
#line 551
size, unsigned 
#line 552
flags = 0) 
#line 554
{ 
#line 555
return cudaMallocHost((void **)((void *)ptr), size, flags); 
#line 556
} 
#line 558
template< class T> static __inline ::cudaError_t 
#line 559
cudaMallocPitch(T **
#line 560
devPtr, ::size_t *
#line 561
pitch, ::size_t 
#line 562
width, ::size_t 
#line 563
height) 
#line 565
{ 
#line 566
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
#line 567
} 
#line 604
template< class T> static __inline ::cudaError_t 
#line 605
cudaMemcpyToSymbol(const T &
#line 606
symbol, const void *
#line 607
src, ::size_t 
#line 608
count, ::size_t 
#line 609
offset = 0, ::cudaMemcpyKind 
#line 610
kind = cudaMemcpyHostToDevice) 
#line 612
{ 
#line 613
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
#line 614
} 
#line 656
template< class T> static __inline ::cudaError_t 
#line 657
cudaMemcpyToSymbolAsync(const T &
#line 658
symbol, const void *
#line 659
src, ::size_t 
#line 660
count, ::size_t 
#line 661
offset = 0, ::cudaMemcpyKind 
#line 662
kind = cudaMemcpyHostToDevice, ::cudaStream_t 
#line 663
stream = 0) 
#line 665
{ 
#line 666
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
#line 667
} 
#line 702
template< class T> static __inline ::cudaError_t 
#line 703
cudaMemcpyFromSymbol(void *
#line 704
dst, const T &
#line 705
symbol, ::size_t 
#line 706
count, ::size_t 
#line 707
offset = 0, ::cudaMemcpyKind 
#line 708
kind = cudaMemcpyDeviceToHost) 
#line 710
{ 
#line 711
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
#line 712
} 
#line 754
template< class T> static __inline ::cudaError_t 
#line 755
cudaMemcpyFromSymbolAsync(void *
#line 756
dst, const T &
#line 757
symbol, ::size_t 
#line 758
count, ::size_t 
#line 759
offset = 0, ::cudaMemcpyKind 
#line 760
kind = cudaMemcpyDeviceToHost, ::cudaStream_t 
#line 761
stream = 0) 
#line 763
{ 
#line 764
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
#line 765
} 
#line 787
template< class T> static __inline ::cudaError_t 
#line 788
cudaGetSymbolAddress(void **
#line 789
devPtr, const T &
#line 790
symbol) 
#line 792
{ 
#line 793
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
#line 794
} 
#line 816
template< class T> static __inline ::cudaError_t 
#line 817
cudaGetSymbolSize(::size_t *
#line 818
size, const T &
#line 819
symbol) 
#line 821
{ 
#line 822
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
#line 823
} 
#line 859
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 860
cudaBindTexture(::size_t *
#line 861
offset, const texture< T, dim, readMode>  &
#line 862
tex, const void *
#line 863
devPtr, const ::cudaChannelFormatDesc &
#line 864
desc, ::size_t 
#line 865
size = 4294967295U) 
#line 867
{ 
#line 868
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
#line 869
} 
#line 904
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 905
cudaBindTexture(::size_t *
#line 906
offset, const texture< T, dim, readMode>  &
#line 907
tex, const void *
#line 908
devPtr, ::size_t 
#line 909
size = 4294967295U) 
#line 911
{ 
#line 912
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
#line 913
} 
#line 960
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 961
cudaBindTexture2D(::size_t *
#line 962
offset, const texture< T, dim, readMode>  &
#line 963
tex, const void *
#line 964
devPtr, const ::cudaChannelFormatDesc &
#line 965
desc, ::size_t 
#line 966
width, ::size_t 
#line 967
height, ::size_t 
#line 968
pitch) 
#line 970
{ 
#line 971
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
#line 972
} 
#line 1018
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1019
cudaBindTexture2D(::size_t *
#line 1020
offset, const texture< T, dim, readMode>  &
#line 1021
tex, const void *
#line 1022
devPtr, ::size_t 
#line 1023
width, ::size_t 
#line 1024
height, ::size_t 
#line 1025
pitch) 
#line 1027
{ 
#line 1028
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
#line 1029
} 
#line 1060
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1061
cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1062
tex, ::cudaArray_const_t 
#line 1063
array, const ::cudaChannelFormatDesc &
#line 1064
desc) 
#line 1066
{ 
#line 1067
return ::cudaBindTextureToArray(&tex, array, &desc); 
#line 1068
} 
#line 1098
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1099
cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1100
tex, ::cudaArray_const_t 
#line 1101
array) 
#line 1103
{ 
#line 1104
::cudaChannelFormatDesc desc; 
#line 1105
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 1107
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
#line 1108
} 
#line 1139
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1140
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1141
tex, ::cudaMipmappedArray_const_t 
#line 1142
mipmappedArray, const ::cudaChannelFormatDesc &
#line 1143
desc) 
#line 1145
{ 
#line 1146
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
#line 1147
} 
#line 1177
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1178
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1179
tex, ::cudaMipmappedArray_const_t 
#line 1180
mipmappedArray) 
#line 1182
{ 
#line 1183
::cudaChannelFormatDesc desc; 
#line 1184
::cudaArray_t levelArray; 
#line 1185
::cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
#line 1187
if (err != (cudaSuccess)) { 
#line 1188
return err; 
#line 1189
}  
#line 1190
err = ::cudaGetChannelDesc(&desc, levelArray); 
#line 1192
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
#line 1193
} 
#line 1216
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1217
cudaUnbindTexture(const texture< T, dim, readMode>  &
#line 1218
tex) 
#line 1220
{ 
#line 1221
return ::cudaUnbindTexture(&tex); 
#line 1222
} 
#line 1250
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1251
cudaGetTextureAlignmentOffset(::size_t *
#line 1252
offset, const texture< T, dim, readMode>  &
#line 1253
tex) 
#line 1255
{ 
#line 1256
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
#line 1257
} 
#line 1302
template< class T> static __inline ::cudaError_t 
#line 1303
cudaFuncSetCacheConfig(T *
#line 1304
func, ::cudaFuncCache 
#line 1305
cacheConfig) 
#line 1307
{ 
#line 1308
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
#line 1309
} 
#line 1311
template< class T> static __inline ::cudaError_t 
#line 1312
cudaFuncSetSharedMemConfig(T *
#line 1313
func, ::cudaSharedMemConfig 
#line 1314
config) 
#line 1316
{ 
#line 1317
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
#line 1318
} 
#line 1347
template< class T> __inline ::cudaError_t 
#line 1348
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
#line 1349
numBlocks, T 
#line 1350
func, int 
#line 1351
blockSize, ::size_t 
#line 1352
dynamicSMemSize) 
#line 1353
{ 
#line 1354
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
#line 1355
} 
#line 1398
template< class T> __inline ::cudaError_t 
#line 1399
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
#line 1400
numBlocks, T 
#line 1401
func, int 
#line 1402
blockSize, ::size_t 
#line 1403
dynamicSMemSize, unsigned 
#line 1404
flags) 
#line 1405
{ 
#line 1406
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
#line 1407
} 
#line 1412
class __cudaOccupancyB2DHelper { 
#line 1413
size_t n; 
#line 1415
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
#line 1416
size_t operator()(int) 
#line 1417
{ 
#line 1418
return n; 
#line 1419
} 
#line 1420
}; 
#line 1467
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1468
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
#line 1469
minGridSize, int *
#line 1470
blockSize, T 
#line 1471
func, UnaryFunction 
#line 1472
blockSizeToDynamicSMemSize, int 
#line 1473
blockSizeLimit = 0, unsigned 
#line 1474
flags = 0) 
#line 1475
{ 
#line 1476
::cudaError_t status; 
#line 1479
int device; 
#line 1480
::cudaFuncAttributes attr; 
#line 1483
int maxThreadsPerMultiProcessor; 
#line 1484
int warpSize; 
#line 1485
int devMaxThreadsPerBlock; 
#line 1486
int multiProcessorCount; 
#line 1487
int funcMaxThreadsPerBlock; 
#line 1488
int occupancyLimit; 
#line 1489
int granularity; 
#line 1492
int maxBlockSize = 0; 
#line 1493
int numBlocks = 0; 
#line 1494
int maxOccupancy = 0; 
#line 1497
int blockSizeToTryAligned; 
#line 1498
int blockSizeToTry; 
#line 1499
int blockSizeLimitAligned; 
#line 1500
int occupancyInBlocks; 
#line 1501
int occupancyInThreads; 
#line 1502
::size_t dynamicSMemSize; 
#line 1508
if (((!minGridSize) || (!blockSize)) || (!func)) { 
#line 1509
return cudaErrorInvalidValue; 
#line 1510
}  
#line 1516
status = ::cudaGetDevice(&device); 
#line 1517
if (status != (cudaSuccess)) { 
#line 1518
return status; 
#line 1519
}  
#line 1521
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
#line 1525
if (status != (cudaSuccess)) { 
#line 1526
return status; 
#line 1527
}  
#line 1529
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
#line 1533
if (status != (cudaSuccess)) { 
#line 1534
return status; 
#line 1535
}  
#line 1537
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
#line 1541
if (status != (cudaSuccess)) { 
#line 1542
return status; 
#line 1543
}  
#line 1545
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
#line 1549
if (status != (cudaSuccess)) { 
#line 1550
return status; 
#line 1551
}  
#line 1553
status = cudaFuncGetAttributes(&attr, func); 
#line 1554
if (status != (cudaSuccess)) { 
#line 1555
return status; 
#line 1556
}  
#line 1558
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
#line 1564
occupancyLimit = maxThreadsPerMultiProcessor; 
#line 1565
granularity = warpSize; 
#line 1567
if (blockSizeLimit == 0) { 
#line 1568
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1569
}  
#line 1571
if (devMaxThreadsPerBlock < blockSizeLimit) { 
#line 1572
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1573
}  
#line 1575
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
#line 1576
blockSizeLimit = funcMaxThreadsPerBlock; 
#line 1577
}  
#line 1579
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
#line 1581
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
#line 1585
if (blockSizeLimit < blockSizeToTryAligned) { 
#line 1586
blockSizeToTry = blockSizeLimit; 
#line 1587
} else { 
#line 1588
blockSizeToTry = blockSizeToTryAligned; 
#line 1589
}  
#line 1591
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
#line 1593
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
#line 1600
if (status != (cudaSuccess)) { 
#line 1601
return status; 
#line 1602
}  
#line 1604
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
#line 1606
if (occupancyInThreads > maxOccupancy) { 
#line 1607
maxBlockSize = blockSizeToTry; 
#line 1608
numBlocks = occupancyInBlocks; 
#line 1609
maxOccupancy = occupancyInThreads; 
#line 1610
}  
#line 1614
if (occupancyLimit == maxOccupancy) { 
#line 1615
break; 
#line 1616
}  
#line 1617
}  
#line 1625
(*minGridSize) = (numBlocks * multiProcessorCount); 
#line 1626
(*blockSize) = maxBlockSize; 
#line 1628
return status; 
#line 1629
} 
#line 1662
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1663
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
#line 1664
minGridSize, int *
#line 1665
blockSize, T 
#line 1666
func, UnaryFunction 
#line 1667
blockSizeToDynamicSMemSize, int 
#line 1668
blockSizeLimit = 0) 
#line 1669
{ 
#line 1670
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
#line 1671
} 
#line 1707
template< class T> static __inline ::cudaError_t 
#line 1708
cudaOccupancyMaxPotentialBlockSize(int *
#line 1709
minGridSize, int *
#line 1710
blockSize, T 
#line 1711
func, ::size_t 
#line 1712
dynamicSMemSize = 0, int 
#line 1713
blockSizeLimit = 0) 
#line 1714
{ 
#line 1715
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
#line 1716
} 
#line 1766
template< class T> static __inline ::cudaError_t 
#line 1767
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
#line 1768
minGridSize, int *
#line 1769
blockSize, T 
#line 1770
func, ::size_t 
#line 1771
dynamicSMemSize = 0, int 
#line 1772
blockSizeLimit = 0, unsigned 
#line 1773
flags = 0) 
#line 1774
{ 
#line 1775
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
#line 1776
} 
#line 1814
template< class T> static __inline ::cudaError_t 
#line 1815
cudaLaunch(T *
#line 1816
func) 
#line 1818
{ 
#line 1819
return ::cudaLaunch((const void *)func); 
#line 1820
} 
#line 1851
template< class T> __inline ::cudaError_t 
#line 1852
cudaFuncGetAttributes(::cudaFuncAttributes *
#line 1853
attr, T *
#line 1854
entry) 
#line 1856
{ 
#line 1857
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
#line 1858
} 
#line 1880
template< class T, int dim> static __inline ::cudaError_t 
#line 1881
cudaBindSurfaceToArray(const surface< T, dim>  &
#line 1882
surf, ::cudaArray_const_t 
#line 1883
array, const ::cudaChannelFormatDesc &
#line 1884
desc) 
#line 1886
{ 
#line 1887
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
#line 1888
} 
#line 1909
template< class T, int dim> static __inline ::cudaError_t 
#line 1910
cudaBindSurfaceToArray(const surface< T, dim>  &
#line 1911
surf, ::cudaArray_const_t 
#line 1912
array) 
#line 1914
{ 
#line 1915
::cudaChannelFormatDesc desc; 
#line 1916
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 1918
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
#line 1919
} 
#line 1933 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\cuda_runtime.h"
#pragma warning(pop)
#line 62 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuComplex.h"
extern "C" {
#line 67 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuComplex.h"
typedef float2 cuFloatComplex; 
#line 69
static __inline float cuCrealf(cuFloatComplex x) 
#line 70
{ 
#line 71
return x.x; 
#line 72
} 
#line 74
static __inline float cuCimagf(cuFloatComplex x) 
#line 75
{ 
#line 76
return x.y; 
#line 77
} 
#line 79
static __inline cuFloatComplex make_cuFloatComplex(float 
#line 80
r, float i) 
#line 81
{ 
#line 82
cuFloatComplex res; 
#line 83
(res.x) = r; 
#line 84
(res.y) = i; 
#line 85
return res; 
#line 86
} 
#line 88
static __inline cuFloatComplex cuConjf(cuFloatComplex x) 
#line 89
{ 
#line 90
return make_cuFloatComplex(cuCrealf(x), -cuCimagf(x)); 
#line 91
} 
#line 92
static __inline cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex 
#line 93
y) 
#line 94
{ 
#line 95
return make_cuFloatComplex(cuCrealf(x) + cuCrealf(y), cuCimagf(x) + cuCimagf(y)); 
#line 97
} 
#line 99
static __inline cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex 
#line 100
y) 
#line 101
{ 
#line 102
return make_cuFloatComplex(cuCrealf(x) - cuCrealf(y), cuCimagf(x) - cuCimagf(y)); 
#line 104
} 
#line 111
static __inline cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex 
#line 112
y) 
#line 113
{ 
#line 114
cuFloatComplex prod; 
#line 115
prod = make_cuFloatComplex((cuCrealf(x) * cuCrealf(y)) - (cuCimagf(x) * cuCimagf(y)), (cuCrealf(x) * cuCimagf(y)) + (cuCimagf(x) * cuCrealf(y))); 
#line 119
return prod; 
#line 120
} 
#line 127
static __inline cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex 
#line 128
y) 
#line 129
{ 
#line 130
cuFloatComplex quot; 
#line 131
float s = fabsf(cuCrealf(y)) + fabsf(cuCimagf(y)); 
#line 132
float oos = (1.0F) / s; 
#line 133
float ars = cuCrealf(x) * oos; 
#line 134
float ais = cuCimagf(x) * oos; 
#line 135
float brs = cuCrealf(y) * oos; 
#line 136
float bis = cuCimagf(y) * oos; 
#line 137
s = ((brs * brs) + (bis * bis)); 
#line 138
oos = ((1.0F) / s); 
#line 139
quot = make_cuFloatComplex(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos); 
#line 141
return quot; 
#line 142
} 
#line 152
static __inline float cuCabsf(cuFloatComplex x) 
#line 153
{ 
#line 154
float a = cuCrealf(x); 
#line 155
float b = cuCimagf(x); 
#line 156
float v, w, t; 
#line 157
a = fabsf(a); 
#line 158
b = fabsf(b); 
#line 159
if (a > b) { 
#line 160
v = a; 
#line 161
w = b; 
#line 162
} else { 
#line 163
v = b; 
#line 164
w = a; 
#line 165
}  
#line 166
t = (w / v); 
#line 167
t = ((1.0F) + (t * t)); 
#line 168
t = (v * sqrtf(t)); 
#line 169
if (((v == (0.0F)) || (v > (3.402823466e+038F))) || (w > (3.402823466e+038F))) { 
#line 170
t = (v + w); 
#line 171
}  
#line 172
return t; 
#line 173
} 
#line 176
typedef double2 cuDoubleComplex; 
#line 178
static __inline double cuCreal(cuDoubleComplex x) 
#line 179
{ 
#line 180
return x.x; 
#line 181
} 
#line 183
static __inline double cuCimag(cuDoubleComplex x) 
#line 184
{ 
#line 185
return x.y; 
#line 186
} 
#line 188
static __inline cuDoubleComplex make_cuDoubleComplex(double 
#line 189
r, double i) 
#line 190
{ 
#line 191
cuDoubleComplex res; 
#line 192
(res.x) = r; 
#line 193
(res.y) = i; 
#line 194
return res; 
#line 195
} 
#line 197
static __inline cuDoubleComplex cuConj(cuDoubleComplex x) 
#line 198
{ 
#line 199
return make_cuDoubleComplex(cuCreal(x), -cuCimag(x)); 
#line 200
} 
#line 202
static __inline cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex 
#line 203
y) 
#line 204
{ 
#line 205
return make_cuDoubleComplex(cuCreal(x) + cuCreal(y), cuCimag(x) + cuCimag(y)); 
#line 207
} 
#line 209
static __inline cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex 
#line 210
y) 
#line 211
{ 
#line 212
return make_cuDoubleComplex(cuCreal(x) - cuCreal(y), cuCimag(x) - cuCimag(y)); 
#line 214
} 
#line 221
static __inline cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex 
#line 222
y) 
#line 223
{ 
#line 224
cuDoubleComplex prod; 
#line 225
prod = make_cuDoubleComplex((cuCreal(x) * cuCreal(y)) - (cuCimag(x) * cuCimag(y)), (cuCreal(x) * cuCimag(y)) + (cuCimag(x) * cuCreal(y))); 
#line 229
return prod; 
#line 230
} 
#line 237
static __inline cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex 
#line 238
y) 
#line 239
{ 
#line 240
cuDoubleComplex quot; 
#line 241
double s = fabs(cuCreal(y)) + fabs(cuCimag(y)); 
#line 242
double oos = (1.0) / s; 
#line 243
double ars = cuCreal(x) * oos; 
#line 244
double ais = cuCimag(x) * oos; 
#line 245
double brs = cuCreal(y) * oos; 
#line 246
double bis = cuCimag(y) * oos; 
#line 247
s = ((brs * brs) + (bis * bis)); 
#line 248
oos = ((1.0) / s); 
#line 249
quot = make_cuDoubleComplex(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos); 
#line 251
return quot; 
#line 252
} 
#line 260
static __inline double cuCabs(cuDoubleComplex x) 
#line 261
{ 
#line 262
double a = cuCreal(x); 
#line 263
double b = cuCimag(x); 
#line 264
double v, w, t; 
#line 265
a = fabs(a); 
#line 266
b = fabs(b); 
#line 267
if (a > b) { 
#line 268
v = a; 
#line 269
w = b; 
#line 270
} else { 
#line 271
v = b; 
#line 272
w = a; 
#line 273
}  
#line 274
t = (w / v); 
#line 275
t = ((1.0) + (t * t)); 
#line 276
t = (v * sqrt(t)); 
#line 277
if (((v == (0.0)) || (v > (1.7976931348623157e+308))) || (w > (1.7976931348623157e+308))) 
#line 278
{ 
#line 279
t = (v + w); 
#line 280
}  
#line 281
return t; 
#line 282
} 
#line 285
}
#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuComplex.h"
typedef cuFloatComplex cuComplex; 
#line 290
static __inline cuComplex make_cuComplex(float x, float 
#line 291
y) 
#line 292
{ 
#line 293
return make_cuFloatComplex(x, y); 
#line 294
} 
#line 297
static __inline cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex 
#line 298
c) 
#line 299
{ 
#line 300
return make_cuDoubleComplex((double)cuCrealf(c), (double)cuCimagf(c)); 
#line 301
} 
#line 303
static __inline cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex 
#line 304
c) 
#line 305
{ 
#line 306
return make_cuFloatComplex((float)cuCreal(c), (float)cuCimag(c)); 
#line 307
} 
#line 310
static __inline cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d) 
#line 311
{ 
#line 312
float real_res; 
#line 313
float imag_res; 
#line 315
real_res = ((cuCrealf(x) * cuCrealf(y)) + cuCrealf(d)); 
#line 316
imag_res = ((cuCrealf(x) * cuCimagf(y)) + cuCimagf(d)); 
#line 318
real_res = ((-(cuCimagf(x) * cuCimagf(y))) + real_res); 
#line 319
imag_res = ((cuCimagf(x) * cuCrealf(y)) + imag_res); 
#line 321
return make_cuComplex(real_res, imag_res); 
#line 322
} 
#line 324
static __inline cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d) 
#line 325
{ 
#line 326
double real_res; 
#line 327
double imag_res; 
#line 329
real_res = ((cuCreal(x) * cuCreal(y)) + cuCreal(d)); 
#line 330
imag_res = ((cuCreal(x) * cuCimag(y)) + cuCimag(d)); 
#line 332
real_res = ((-(cuCimag(x) * cuCimag(y))) + real_res); 
#line 333
imag_res = ((cuCimag(x) * cuCreal(y)) + imag_res); 
#line 335
return make_cuDoubleComplex(real_res, imag_res); 
#line 336
} 
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
typedef 
#line 94
struct __declspec(align(2)) { 
#line 95
unsigned short x; 
#line 96
} __half; 
#line 100
typedef 
#line 98
struct __declspec(align(4)) { 
#line 99
unsigned x; 
#line 100
} __half2; 
#line 103
typedef __half half; 
#line 104
typedef __half2 half2; 
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __float2half(const float a); 
#line 138
static __inline __half __float2half_rz(const float a); 
#line 148
static __inline __half __float2half_rd(const float a); 
#line 158
static __inline __half __float2half_ru(const float a); 
#line 167
static __inline float __half2float(const __half a); 
#line 178
static __inline int __half2int_rn(__half h); 
#line 188
static __inline int __half2int_rz(__half h); 
#line 198
static __inline int __half2int_rd(__half h); 
#line 208
static __inline int __half2int_ru(__half h); 
#line 219
static __inline __half __int2half_rn(int i); 
#line 229
static __inline __half __int2half_rz(int i); 
#line 239
static __inline __half __int2half_rd(int i); 
#line 249
static __inline __half __int2half_ru(int i); 
#line 261
static __inline short __half2short_rn(__half h); 
#line 271
static __inline short __half2short_rz(__half h); 
#line 281
static __inline short __half2short_rd(__half h); 
#line 291
static __inline short __half2short_ru(__half h); 
#line 303
static __inline __half __short2half_rn(short i); 
#line 313
static __inline __half __short2half_rz(short i); 
#line 323
static __inline __half __short2half_rd(short i); 
#line 333
static __inline __half __short2half_ru(short i); 
#line 344
static __inline unsigned __half2uint_rn(__half h); 
#line 354
static __inline unsigned __half2uint_rz(__half h); 
#line 364
static __inline unsigned __half2uint_rd(__half h); 
#line 374
static __inline unsigned __half2uint_ru(__half h); 
#line 385
static __inline __half __uint2half_rn(unsigned i); 
#line 395
static __inline __half __uint2half_rz(unsigned i); 
#line 405
static __inline __half __uint2half_rd(unsigned i); 
#line 415
static __inline __half __uint2half_ru(unsigned i); 
#line 427
static __inline unsigned short __half2ushort_rn(__half h); 
#line 438
static __inline unsigned short __half2ushort_rz(__half h); 
#line 448
static __inline unsigned short __half2ushort_rd(__half h); 
#line 458
static __inline unsigned short __half2ushort_ru(__half h); 
#line 470
static __inline __half __ushort2half_rn(unsigned short i); 
#line 481
static __inline __half __ushort2half_rz(unsigned short i); 
#line 491
static __inline __half __ushort2half_rd(unsigned short i); 
#line 501
static __inline __half __ushort2half_ru(unsigned short i); 
#line 513
static __inline unsigned __int64 __half2ull_rn(__half h); 
#line 524
static __inline unsigned __int64 __half2ull_rz(__half h); 
#line 534
static __inline unsigned __int64 __half2ull_rd(__half h); 
#line 544
static __inline unsigned __int64 __half2ull_ru(__half h); 
#line 556
static __inline __half __ull2half_rn(unsigned __int64 i); 
#line 567
static __inline __half __ull2half_rz(unsigned __int64 i); 
#line 577
static __inline __half __ull2half_rd(unsigned __int64 i); 
#line 587
static __inline __half __ull2half_ru(unsigned __int64 i); 
#line 599
static __inline __int64 __half2ll_rn(__half h); 
#line 609
static __inline __int64 __half2ll_rz(__half h); 
#line 619
static __inline __int64 __half2ll_rd(__half h); 
#line 629
static __inline __int64 __half2ll_ru(__half h); 
#line 641
static __inline __half __ll2half_rn(__int64 i); 
#line 651
static __inline __half __ll2half_rz(__int64 i); 
#line 661
static __inline __half __ll2half_rd(__int64 i); 
#line 671
static __inline __half __ll2half_ru(__int64 i); 
#line 682
static __inline __half htrunc(const __half h); 
#line 691
static __inline __half hceil(const __half h); 
#line 700
static __inline __half hfloor(const __half h); 
#line 712
static __inline __half hrint(const __half h); 
#line 723
static __inline __half2 h2trunc(const __half2 h); 
#line 734
static __inline __half2 h2ceil(const __half2 h); 
#line 745
static __inline __half2 h2floor(const __half2 h); 
#line 758
static __inline __half2 h2rint(const __half2 h); 
#line 771
static __inline __half2 __float2half2_rn(const float a); 
#line 785
static __inline __half2 __floats2half2_rn(const float a, const float b); 
#line 799
static __inline __half2 __float22half2_rn(const float2 a); 
#line 809
static __inline float2 __half22float2(const __half2 a); 
#line 819
static __inline float __low2float(const __half2 a); 
#line 829
static __inline __half2 __half2half2(const __half a); 
#line 839
static __inline float __high2float(const __half2 a); 
#line 849
static __inline __half2 __lowhigh2highlow(const __half2 a); 
#line 862
static __inline __half2 __lows2half2(const __half2 a, const __half2 b); 
#line 875
static __inline __half2 __highs2half2(const __half2 a, const __half2 b); 
#line 884
static __inline __half __high2half(const __half2 a); 
#line 893
static __inline __half __low2half(const __half2 a); 
#line 903
static __inline int __hisinf(const __half a); 
#line 915
static __inline __half2 __halves2half2(const __half a, const __half b); 
#line 926
static __inline __half2 __low2half2(const __half2 a); 
#line 937
static __inline __half2 __high2half2(const __half2 a); 
#line 948
static __inline short __half_as_short(const __half h); 
#line 958
static __inline unsigned short __half_as_ushort(const __half h); 
#line 968
static __inline __half __short_as_half(const short i); 
#line 978
static __inline __half __ushort_as_half(const unsigned short i); 
#line 985 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __shfl(__half2 var, int delta, int width = 32); 
#line 986
static __inline __half2 __shfl_up(__half2 var, unsigned delta, int width = 32); 
#line 987
static __inline __half2 __shfl_down(__half2 var, unsigned delta, int width = 32); 
#line 988
static __inline __half2 __shfl_xor(__half2 var, int delta, int width = 32); 
#line 989
static __inline __half __shfl(__half var, int delta, int width = 32); 
#line 990
static __inline __half __shfl_up(__half var, unsigned delta, int width = 32); 
#line 991
static __inline __half __shfl_down(__half var, unsigned delta, int width = 32); 
#line 992
static __inline __half __shfl_xor(__half var, int delta, int width = 32); 
#line 1000 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __ldg(const __half2 * ptr); 
#line 1001
static __inline __half __ldg(const __half * ptr); 
#line 1002
static __inline __half2 __ldcg(const __half2 * ptr); 
#line 1003
static __inline __half __ldcg(const __half * ptr); 
#line 1004
static __inline __half2 __ldca(const __half2 * ptr); 
#line 1005
static __inline __half __ldca(const __half * ptr); 
#line 1006
static __inline __half2 __ldcs(const __half2 * ptr); 
#line 1007
static __inline __half __ldcs(const __half * ptr); 
#line 1022 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __heq2(const __half2 a, const __half2 b); 
#line 1034
static __inline __half2 __hne2(const __half2 a, const __half2 b); 
#line 1046
static __inline __half2 __hle2(const __half2 a, const __half2 b); 
#line 1058
static __inline __half2 __hge2(const __half2 a, const __half2 b); 
#line 1070
static __inline __half2 __hlt2(const __half2 a, const __half2 b); 
#line 1082
static __inline __half2 __hgt2(const __half2 a, const __half2 b); 
#line 1094
static __inline __half2 __hequ2(const __half2 a, const __half2 b); 
#line 1106
static __inline __half2 __hneu2(const __half2 a, const __half2 b); 
#line 1118
static __inline __half2 __hleu2(const __half2 a, const __half2 b); 
#line 1130
static __inline __half2 __hgeu2(const __half2 a, const __half2 b); 
#line 1142
static __inline __half2 __hltu2(const __half2 a, const __half2 b); 
#line 1154
static __inline __half2 __hgtu2(const __half2 a, const __half2 b); 
#line 1164
static __inline __half2 __hisnan2(const __half2 a); 
#line 1174
static __inline __half2 __hadd2(const __half2 a, const __half2 b); 
#line 1185
static __inline __half2 __hsub2(const __half2 a, const __half2 b); 
#line 1196
static __inline __half2 __hmul2(const __half2 a, const __half2 b); 
#line 1206
static __inline __half2 h2div(const __half2 a, const __half2 b); 
#line 1219
static __inline __half2 __hadd2_sat(const __half2 a, const __half2 b); 
#line 1232
static __inline __half2 __hsub2_sat(const __half2 a, const __half2 b); 
#line 1245
static __inline __half2 __hmul2_sat(const __half2 a, const __half2 b); 
#line 1258
static __inline __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c); 
#line 1272
static __inline __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c); 
#line 1282
static __inline __half2 __hneg2(const __half2 a); 
#line 1292
static __inline __half __hadd(const __half a, const __half b); 
#line 1302
static __inline __half __hsub(const __half a, const __half b); 
#line 1312
static __inline __half __hmul(const __half a, const __half b); 
#line 1322
static __inline __half hdiv(const __half a, const __half b); 
#line 1333
static __inline __half __hadd_sat(const __half a, const __half b); 
#line 1346
static __inline __half __hsub_sat(const __half a, const __half b); 
#line 1359
static __inline __half __hmul_sat(const __half a, const __half b); 
#line 1371
static __inline __half __hfma(const __half a, const __half b, const __half c); 
#line 1385
static __inline __half __hfma_sat(const __half a, const __half b, const __half c); 
#line 1394
static __inline __half __hneg(const __half a); 
#line 1408
static __inline bool __hbeq2(const __half2 a, const __half2 b); 
#line 1422
static __inline bool __hbne2(const __half2 a, const __half2 b); 
#line 1436
static __inline bool __hble2(const __half2 a, const __half2 b); 
#line 1450
static __inline bool __hbge2(const __half2 a, const __half2 b); 
#line 1464
static __inline bool __hblt2(const __half2 a, const __half2 b); 
#line 1478
static __inline bool __hbgt2(const __half2 a, const __half2 b); 
#line 1492
static __inline bool __hbequ2(const __half2 a, const __half2 b); 
#line 1506
static __inline bool __hbneu2(const __half2 a, const __half2 b); 
#line 1520
static __inline bool __hbleu2(const __half2 a, const __half2 b); 
#line 1536
static __inline bool __hbgeu2(const __half2 a, const __half2 b); 
#line 1550
static __inline bool __hbltu2(const __half2 a, const __half2 b); 
#line 1566
static __inline bool __hbgtu2(const __half2 a, const __half2 b); 
#line 1576
static __inline bool __heq(const __half a, const __half b); 
#line 1586
static __inline bool __hne(const __half a, const __half b); 
#line 1596
static __inline bool __hle(const __half a, const __half b); 
#line 1606
static __inline bool __hge(const __half a, const __half b); 
#line 1616
static __inline bool __hlt(const __half a, const __half b); 
#line 1626
static __inline bool __hgt(const __half a, const __half b); 
#line 1637
static __inline bool __hequ(const __half a, const __half b); 
#line 1648
static __inline bool __hneu(const __half a, const __half b); 
#line 1659
static __inline bool __hleu(const __half a, const __half b); 
#line 1670
static __inline bool __hgeu(const __half a, const __half b); 
#line 1681
static __inline bool __hltu(const __half a, const __half b); 
#line 1692
static __inline bool __hgtu(const __half a, const __half b); 
#line 1701
static __inline bool __hisnan(const __half a); 
#line 1710
static __inline __half hsqrt(const __half a); 
#line 1721
static __inline __half hrsqrt(const __half a); 
#line 1730
static __inline __half hrcp(const __half a); 
#line 1740
static __inline __half hlog(const __half a); 
#line 1750
static __inline __half hlog2(const __half a); 
#line 1760
static __inline __half hlog10(const __half a); 
#line 1771
static __inline __half hexp(const __half a); 
#line 1782
static __inline __half hexp2(const __half a); 
#line 1793
static __inline __half hexp10(const __half a); 
#line 1802
static __inline __half hcos(const __half a); 
#line 1811
static __inline __half hsin(const __half a); 
#line 1821
static __inline __half2 h2sqrt(const __half2 a); 
#line 1832
static __inline __half2 h2rsqrt(const __half2 a); 
#line 1842
static __inline __half2 h2rcp(const __half2 a); 
#line 1853
static __inline __half2 h2log(const __half2 a); 
#line 1864
static __inline __half2 h2log2(const __half2 a); 
#line 1875
static __inline __half2 h2log10(const __half2 a); 
#line 1886
static __inline __half2 h2exp(const __half2 a); 
#line 1897
static __inline __half2 h2exp2(const __half2 a); 
#line 1908
static __inline __half2 h2exp10(const __half2 a); 
#line 1918
static __inline __half2 h2cos(const __half2 a); 
#line 1927
static __inline __half2 h2sin(const __half2 a); 
#line 1931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline int __half2int_rn(__half h) 
#line 1932
{int volatile ___ = 1;(void)h;
#line 1936
::exit(___);}
#if 0
#line 1932
{ 
#line 1933
int i; 
#line 1934
__asm cvt.rni.s32.f16 %0, %1;
return i; 
#line 1936
} 
#endif
#line 1937 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline int __half2int_rz(__half h) 
#line 1938
{int volatile ___ = 1;(void)h;
#line 1942
::exit(___);}
#if 0
#line 1938
{ 
#line 1939
int i; 
#line 1940
__asm cvt.rzi.s32.f16 %0, %1;
return i; 
#line 1942
} 
#endif
#line 1943 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline int __half2int_rd(__half h) 
#line 1944
{int volatile ___ = 1;(void)h;
#line 1948
::exit(___);}
#if 0
#line 1944
{ 
#line 1945
int i; 
#line 1946
__asm cvt.rmi.s32.f16 %0, %1;
return i; 
#line 1948
} 
#endif
#line 1949 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline int __half2int_ru(__half h) 
#line 1950
{int volatile ___ = 1;(void)h;
#line 1954
::exit(___);}
#if 0
#line 1950
{ 
#line 1951
int i; 
#line 1952
__asm cvt.rpi.s32.f16 %0, %1;
return i; 
#line 1954
} 
#endif
#line 1955 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __int2half_rn(int i) 
#line 1956
{int volatile ___ = 1;(void)i;
#line 1960
::exit(___);}
#if 0
#line 1956
{ 
#line 1957
__half h; 
#line 1958
__asm cvt.rn.f16.s32 %0, %1;
return h; 
#line 1960
} 
#endif
#line 1961 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __int2half_rz(int i) 
#line 1962
{int volatile ___ = 1;(void)i;
#line 1966
::exit(___);}
#if 0
#line 1962
{ 
#line 1963
__half h; 
#line 1964
__asm cvt.rz.f16.s32 %0, %1;
return h; 
#line 1966
} 
#endif
#line 1967 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __int2half_rd(int i) 
#line 1968
{int volatile ___ = 1;(void)i;
#line 1972
::exit(___);}
#if 0
#line 1968
{ 
#line 1969
__half h; 
#line 1970
__asm cvt.rm.f16.s32 %0, %1;
return h; 
#line 1972
} 
#endif
#line 1973 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __int2half_ru(int i) 
#line 1974
{int volatile ___ = 1;(void)i;
#line 1978
::exit(___);}
#if 0
#line 1974
{ 
#line 1975
__half h; 
#line 1976
__asm cvt.rp.f16.s32 %0, %1;
return h; 
#line 1978
} 
#endif
#line 1980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline short __half2short_rn(__half h) 
#line 1981
{int volatile ___ = 1;(void)h;
#line 1985
::exit(___);}
#if 0
#line 1981
{ 
#line 1982
short i; 
#line 1983
__asm cvt.rni.s16.f16 %0, %1;
return i; 
#line 1985
} 
#endif
#line 1986 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline short __half2short_rz(__half h) 
#line 1987
{int volatile ___ = 1;(void)h;
#line 1991
::exit(___);}
#if 0
#line 1987
{ 
#line 1988
short i; 
#line 1989
__asm cvt.rzi.s16.f16 %0, %1;
return i; 
#line 1991
} 
#endif
#line 1992 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline short __half2short_rd(__half h) 
#line 1993
{int volatile ___ = 1;(void)h;
#line 1997
::exit(___);}
#if 0
#line 1993
{ 
#line 1994
short i; 
#line 1995
__asm cvt.rmi.s16.f16 %0, %1;
return i; 
#line 1997
} 
#endif
#line 1998 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline short __half2short_ru(__half h) 
#line 1999
{int volatile ___ = 1;(void)h;
#line 2003
::exit(___);}
#if 0
#line 1999
{ 
#line 2000
short i; 
#line 2001
__asm cvt.rpi.s16.f16 %0, %1;
return i; 
#line 2003
} 
#endif
#line 2004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __short2half_rn(short i) 
#line 2005
{int volatile ___ = 1;(void)i;
#line 2009
::exit(___);}
#if 0
#line 2005
{ 
#line 2006
__half h; 
#line 2007
__asm cvt.rn.f16.s16 %0, %1;
return h; 
#line 2009
} 
#endif
#line 2010 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __short2half_rz(short i) 
#line 2011
{int volatile ___ = 1;(void)i;
#line 2015
::exit(___);}
#if 0
#line 2011
{ 
#line 2012
__half h; 
#line 2013
__asm cvt.rz.f16.s16 %0, %1;
return h; 
#line 2015
} 
#endif
#line 2016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __short2half_rd(short i) 
#line 2017
{int volatile ___ = 1;(void)i;
#line 2021
::exit(___);}
#if 0
#line 2017
{ 
#line 2018
__half h; 
#line 2019
__asm cvt.rm.f16.s16 %0, %1;
return h; 
#line 2021
} 
#endif
#line 2022 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __short2half_ru(short i) 
#line 2023
{int volatile ___ = 1;(void)i;
#line 2027
::exit(___);}
#if 0
#line 2023
{ 
#line 2024
__half h; 
#line 2025
__asm cvt.rp.f16.s16 %0, %1;
return h; 
#line 2027
} 
#endif
#line 2029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __half2uint_rn(__half h) 
#line 2030
{int volatile ___ = 1;(void)h;
#line 2034
::exit(___);}
#if 0
#line 2030
{ 
#line 2031
unsigned i; 
#line 2032
__asm cvt.rni.u32.f16 %0, %1;
return i; 
#line 2034
} 
#endif
#line 2035 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __half2uint_rz(__half h) 
#line 2036
{int volatile ___ = 1;(void)h;
#line 2040
::exit(___);}
#if 0
#line 2036
{ 
#line 2037
unsigned i; 
#line 2038
__asm cvt.rzi.u32.f16 %0, %1;
return i; 
#line 2040
} 
#endif
#line 2041 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __half2uint_rd(__half h) 
#line 2042
{int volatile ___ = 1;(void)h;
#line 2046
::exit(___);}
#if 0
#line 2042
{ 
#line 2043
unsigned i; 
#line 2044
__asm cvt.rmi.u32.f16 %0, %1;
return i; 
#line 2046
} 
#endif
#line 2047 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __half2uint_ru(__half h) 
#line 2048
{int volatile ___ = 1;(void)h;
#line 2052
::exit(___);}
#if 0
#line 2048
{ 
#line 2049
unsigned i; 
#line 2050
__asm cvt.rpi.u32.f16 %0, %1;
return i; 
#line 2052
} 
#endif
#line 2053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __uint2half_rn(unsigned i) 
#line 2054
{int volatile ___ = 1;(void)i;
#line 2058
::exit(___);}
#if 0
#line 2054
{ 
#line 2055
__half h; 
#line 2056
__asm cvt.rn.f16.u32 %0, %1;
return h; 
#line 2058
} 
#endif
#line 2059 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __uint2half_rz(unsigned i) 
#line 2060
{int volatile ___ = 1;(void)i;
#line 2064
::exit(___);}
#if 0
#line 2060
{ 
#line 2061
__half h; 
#line 2062
__asm cvt.rz.f16.u32 %0, %1;
return h; 
#line 2064
} 
#endif
#line 2065 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __uint2half_rd(unsigned i) 
#line 2066
{int volatile ___ = 1;(void)i;
#line 2070
::exit(___);}
#if 0
#line 2066
{ 
#line 2067
__half h; 
#line 2068
__asm cvt.rm.f16.u32 %0, %1;
return h; 
#line 2070
} 
#endif
#line 2071 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __uint2half_ru(unsigned i) 
#line 2072
{int volatile ___ = 1;(void)i;
#line 2076
::exit(___);}
#if 0
#line 2072
{ 
#line 2073
__half h; 
#line 2074
__asm cvt.rp.f16.u32 %0, %1;
return h; 
#line 2076
} 
#endif
#line 2078 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned short __half2ushort_rn(__half h) 
#line 2079
{int volatile ___ = 1;(void)h;
#line 2083
::exit(___);}
#if 0
#line 2079
{ 
#line 2080
unsigned short i; 
#line 2081
__asm cvt.rni.u16.f16 %0, %1;
return i; 
#line 2083
} 
#endif
#line 2084 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned short __half2ushort_rz(__half h) 
#line 2085
{int volatile ___ = 1;(void)h;
#line 2089
::exit(___);}
#if 0
#line 2085
{ 
#line 2086
unsigned short i; 
#line 2087
__asm cvt.rzi.u16.f16 %0, %1;
return i; 
#line 2089
} 
#endif
#line 2090 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned short __half2ushort_rd(__half h) 
#line 2091
{int volatile ___ = 1;(void)h;
#line 2095
::exit(___);}
#if 0
#line 2091
{ 
#line 2092
unsigned short i; 
#line 2093
__asm cvt.rmi.u16.f16 %0, %1;
return i; 
#line 2095
} 
#endif
#line 2096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned short __half2ushort_ru(__half h) 
#line 2097
{int volatile ___ = 1;(void)h;
#line 2101
::exit(___);}
#if 0
#line 2097
{ 
#line 2098
unsigned short i; 
#line 2099
__asm cvt.rpi.u16.f16 %0, %1;
return i; 
#line 2101
} 
#endif
#line 2102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ushort2half_rn(unsigned short i) 
#line 2103
{int volatile ___ = 1;(void)i;
#line 2107
::exit(___);}
#if 0
#line 2103
{ 
#line 2104
__half h; 
#line 2105
__asm cvt.rn.f16.u16 %0, %1;
return h; 
#line 2107
} 
#endif
#line 2108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ushort2half_rz(unsigned short i) 
#line 2109
{int volatile ___ = 1;(void)i;
#line 2113
::exit(___);}
#if 0
#line 2109
{ 
#line 2110
__half h; 
#line 2111
__asm cvt.rz.f16.u16 %0, %1;
return h; 
#line 2113
} 
#endif
#line 2114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ushort2half_rd(unsigned short i) 
#line 2115
{int volatile ___ = 1;(void)i;
#line 2119
::exit(___);}
#if 0
#line 2115
{ 
#line 2116
__half h; 
#line 2117
__asm cvt.rm.f16.u16 %0, %1;
return h; 
#line 2119
} 
#endif
#line 2120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ushort2half_ru(unsigned short i) 
#line 2121
{int volatile ___ = 1;(void)i;
#line 2125
::exit(___);}
#if 0
#line 2121
{ 
#line 2122
__half h; 
#line 2123
__asm cvt.rp.f16.u16 %0, %1;
return h; 
#line 2125
} 
#endif
#line 2127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __int64 __half2ull_rn(__half h) 
#line 2128
{int volatile ___ = 1;(void)h;
#line 2132
::exit(___);}
#if 0
#line 2128
{ 
#line 2129
unsigned __int64 i; 
#line 2130
__asm cvt.rni.u64.f16 %0, %1;
return i; 
#line 2132
} 
#endif
#line 2133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __int64 __half2ull_rz(__half h) 
#line 2134
{int volatile ___ = 1;(void)h;
#line 2138
::exit(___);}
#if 0
#line 2134
{ 
#line 2135
unsigned __int64 i; 
#line 2136
__asm cvt.rzi.u64.f16 %0, %1;
return i; 
#line 2138
} 
#endif
#line 2139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __int64 __half2ull_rd(__half h) 
#line 2140
{int volatile ___ = 1;(void)h;
#line 2144
::exit(___);}
#if 0
#line 2140
{ 
#line 2141
unsigned __int64 i; 
#line 2142
__asm cvt.rmi.u64.f16 %0, %1;
return i; 
#line 2144
} 
#endif
#line 2145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned __int64 __half2ull_ru(__half h) 
#line 2146
{int volatile ___ = 1;(void)h;
#line 2150
::exit(___);}
#if 0
#line 2146
{ 
#line 2147
unsigned __int64 i; 
#line 2148
__asm cvt.rpi.u64.f16 %0, %1;
return i; 
#line 2150
} 
#endif
#line 2151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ull2half_rn(unsigned __int64 i) 
#line 2152
{int volatile ___ = 1;(void)i;
#line 2156
::exit(___);}
#if 0
#line 2152
{ 
#line 2153
__half h; 
#line 2154
__asm cvt.rn.f16.u64 %0, %1;
return h; 
#line 2156
} 
#endif
#line 2157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ull2half_rz(unsigned __int64 i) 
#line 2158
{int volatile ___ = 1;(void)i;
#line 2162
::exit(___);}
#if 0
#line 2158
{ 
#line 2159
__half h; 
#line 2160
__asm cvt.rz.f16.u64 %0, %1;
return h; 
#line 2162
} 
#endif
#line 2163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ull2half_rd(unsigned __int64 i) 
#line 2164
{int volatile ___ = 1;(void)i;
#line 2168
::exit(___);}
#if 0
#line 2164
{ 
#line 2165
__half h; 
#line 2166
__asm cvt.rm.f16.u64 %0, %1;
return h; 
#line 2168
} 
#endif
#line 2169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ull2half_ru(unsigned __int64 i) 
#line 2170
{int volatile ___ = 1;(void)i;
#line 2174
::exit(___);}
#if 0
#line 2170
{ 
#line 2171
__half h; 
#line 2172
__asm cvt.rp.f16.u64 %0, %1;
return h; 
#line 2174
} 
#endif
#line 2176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __int64 __half2ll_rn(__half h) 
#line 2177
{int volatile ___ = 1;(void)h;
#line 2181
::exit(___);}
#if 0
#line 2177
{ 
#line 2178
__int64 i; 
#line 2179
__asm cvt.rni.s64.f16 %0, %1;
return i; 
#line 2181
} 
#endif
#line 2182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __int64 __half2ll_rz(__half h) 
#line 2183
{int volatile ___ = 1;(void)h;
#line 2187
::exit(___);}
#if 0
#line 2183
{ 
#line 2184
__int64 i; 
#line 2185
__asm cvt.rzi.s64.f16 %0, %1;
return i; 
#line 2187
} 
#endif
#line 2188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __int64 __half2ll_rd(__half h) 
#line 2189
{int volatile ___ = 1;(void)h;
#line 2193
::exit(___);}
#if 0
#line 2189
{ 
#line 2190
__int64 i; 
#line 2191
__asm cvt.rmi.s64.f16 %0, %1;
return i; 
#line 2193
} 
#endif
#line 2194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __int64 __half2ll_ru(__half h) 
#line 2195
{int volatile ___ = 1;(void)h;
#line 2199
::exit(___);}
#if 0
#line 2195
{ 
#line 2196
__int64 i; 
#line 2197
__asm cvt.rpi.s64.f16 %0, %1;
return i; 
#line 2199
} 
#endif
#line 2200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ll2half_rn(__int64 i) 
#line 2201
{int volatile ___ = 1;(void)i;
#line 2205
::exit(___);}
#if 0
#line 2201
{ 
#line 2202
__half h; 
#line 2203
__asm cvt.rn.f16.s64 %0, %1;
return h; 
#line 2205
} 
#endif
#line 2206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ll2half_rz(__int64 i) 
#line 2207
{int volatile ___ = 1;(void)i;
#line 2211
::exit(___);}
#if 0
#line 2207
{ 
#line 2208
__half h; 
#line 2209
__asm cvt.rz.f16.s64 %0, %1;
return h; 
#line 2211
} 
#endif
#line 2212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ll2half_rd(__int64 i) 
#line 2213
{int volatile ___ = 1;(void)i;
#line 2217
::exit(___);}
#if 0
#line 2213
{ 
#line 2214
__half h; 
#line 2215
__asm cvt.rm.f16.s64 %0, %1;
return h; 
#line 2217
} 
#endif
#line 2218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ll2half_ru(__int64 i) 
#line 2219
{int volatile ___ = 1;(void)i;
#line 2223
::exit(___);}
#if 0
#line 2219
{ 
#line 2220
__half h; 
#line 2221
__asm cvt.rp.f16.s64 %0, %1;
return h; 
#line 2223
} 
#endif
#line 2225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half htrunc(const __half h) 
#line 2226
{int volatile ___ = 1;(void)h;
#line 2230
::exit(___);}
#if 0
#line 2226
{ 
#line 2227
__half r; 
#line 2228
__asm cvt.rzi.f16.f16 %0, %1;
return r; 
#line 2230
} 
#endif
#line 2231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hceil(const __half h) 
#line 2232
{int volatile ___ = 1;(void)h;
#line 2236
::exit(___);}
#if 0
#line 2232
{ 
#line 2233
__half r; 
#line 2234
__asm cvt.rpi.f16.f16 %0, %1;
return r; 
#line 2236
} 
#endif
#line 2237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hfloor(const __half h) 
#line 2238
{int volatile ___ = 1;(void)h;
#line 2242
::exit(___);}
#if 0
#line 2238
{ 
#line 2239
__half r; 
#line 2240
__asm cvt.rmi.f16.f16 %0, %1;
return r; 
#line 2242
} 
#endif
#line 2243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hrint(const __half h) 
#line 2244
{int volatile ___ = 1;(void)h;
#line 2248
::exit(___);}
#if 0
#line 2244
{ 
#line 2245
__half r; 
#line 2246
__asm cvt.rni.f16.f16 %0, %1;
return r; 
#line 2248
} 
#endif
#line 2250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2trunc(const __half2 h) 
#line 2251
{int volatile ___ = 1;(void)h;
#line 2259
::exit(___);}
#if 0
#line 2251
{ 
#line 2252
__half2 val; 
#line 2253
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  cvt.rzi.f16.f16 low, low;
  cvt.rzi.f16.f16 high, high;
  mov.b32 %0, {low,high};}

#line 2258
return val; 
#line 2259
} 
#endif
#line 2260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2ceil(const __half2 h) 
#line 2261
{int volatile ___ = 1;(void)h;
#line 2269
::exit(___);}
#if 0
#line 2261
{ 
#line 2262
__half2 val; 
#line 2263
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  cvt.rpi.f16.f16 low, low;
  cvt.rpi.f16.f16 high, high;
  mov.b32 %0, {low,high};}

#line 2268
return val; 
#line 2269
} 
#endif
#line 2270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2floor(const __half2 h) 
#line 2271
{int volatile ___ = 1;(void)h;
#line 2279
::exit(___);}
#if 0
#line 2271
{ 
#line 2272
__half2 val; 
#line 2273
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  cvt.rmi.f16.f16 low, low;
  cvt.rmi.f16.f16 high, high;
  mov.b32 %0, {low,high};}

#line 2278
return val; 
#line 2279
} 
#endif
#line 2280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2rint(const __half2 h) 
#line 2281
{int volatile ___ = 1;(void)h;
#line 2289
::exit(___);}
#if 0
#line 2281
{ 
#line 2282
__half2 val; 
#line 2283
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  cvt.rni.f16.f16 low, low;
  cvt.rni.f16.f16 high, high;
  mov.b32 %0, {low,high};}

#line 2288
return val; 
#line 2289
} 
#endif
#line 2291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline float2 __half22float2(const __half2 l) 
#line 2292
{int volatile ___ = 1;(void)l;
#line 2304
::exit(___);}
#if 0
#line 2292
{ 
#line 2293
float hi_float; 
#line 2294
float lo_float; 
#line 2295
__asm {.reg .f16 low,high;
  mov.b32 {low,high},%1;
  cvt.f32.f16 %0, low;}

__asm {.reg .f16 low,high;
  mov.b32 {low,high},%1;
  cvt.f32.f16 %0, high;}

return make_float2(lo_float, hi_float); 
#line 2304
} 
#endif
#line 2305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __float2half(const float f) 
#line 2306
{int volatile ___ = 1;(void)f;
#line 2310
::exit(___);}
#if 0
#line 2306
{ 
#line 2307
__half val; 
#line 2308
__asm {  cvt.rn.f16.f32 %0, %1;}

#line 2309
return val; 
#line 2310
} 
#endif
#line 2311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __float2half_rz(const float f) 
#line 2312
{int volatile ___ = 1;(void)f;
#line 2316
::exit(___);}
#if 0
#line 2312
{ 
#line 2313
__half val; 
#line 2314
__asm {  cvt.rz.f16.f32 %0, %1;}

#line 2315
return val; 
#line 2316
} 
#endif
#line 2317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __float2half_rd(const float f) 
#line 2318
{int volatile ___ = 1;(void)f;
#line 2322
::exit(___);}
#if 0
#line 2318
{ 
#line 2319
__half val; 
#line 2320
__asm {  cvt.rm.f16.f32 %0, %1;}

#line 2321
return val; 
#line 2322
} 
#endif
#line 2323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __float2half_ru(const float f) 
#line 2324
{int volatile ___ = 1;(void)f;
#line 2328
::exit(___);}
#if 0
#line 2324
{ 
#line 2325
__half val; 
#line 2326
__asm {  cvt.rp.f16.f32 %0, %1;}

#line 2327
return val; 
#line 2328
} 
#endif
#line 2329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline float __half2float(const __half h) 
#line 2330
{int volatile ___ = 1;(void)h;
#line 2334
::exit(___);}
#if 0
#line 2330
{ 
#line 2331
float val; 
#line 2332
__asm {  cvt.f32.f16 %0, %1;}

#line 2333
return val; 
#line 2334
} 
#endif
#line 2335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __float2half2_rn(const float f) 
#line 2336
{int volatile ___ = 1;(void)f;
#line 2342
::exit(___);}
#if 0
#line 2336
{ 
#line 2337
__half2 val; 
#line 2338
__asm {.reg .f16 low;
  cvt.rn.f16.f32 low, %1;
  mov.b32 %0, {low,low};}

#line 2341
return val; 
#line 2342
} 
#endif
#line 2343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __floats2half2_rn(const float f1, const float f2) 
#line 2344
{int volatile ___ = 1;(void)f1;(void)f2;
#line 2351
::exit(___);}
#if 0
#line 2344
{ 
#line 2345
__half2 val; 
#line 2346
__asm {.reg .f16 low,high;
  cvt.rn.f16.f32 low, %1;
  cvt.rn.f16.f32 high, %2;
  mov.b32 %0, {low,high};}

#line 2350
return val; 
#line 2351
} 
#endif
#line 2352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __float22half2_rn(const float2 f) 
#line 2353
{int volatile ___ = 1;(void)f;
#line 2356
::exit(___);}
#if 0
#line 2353
{ 
#line 2354
__half2 val = __floats2half2_rn(f.x, f.y); 
#line 2355
return val; 
#line 2356
} 
#endif
#line 2357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline float __low2float(const __half2 l) 
#line 2358
{int volatile ___ = 1;(void)l;
#line 2364
::exit(___);}
#if 0
#line 2358
{ 
#line 2359
float val; 
#line 2360
__asm {.reg .f16 low,high;
  mov.b32 {low,high},%1;
  cvt.f32.f16 %0, low;}

#line 2363
return val; 
#line 2364
} 
#endif
#line 2365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline float __high2float(const __half2 l) 
#line 2366
{int volatile ___ = 1;(void)l;
#line 2372
::exit(___);}
#if 0
#line 2366
{ 
#line 2367
float val; 
#line 2368
__asm {.reg .f16 low,high;
  mov.b32 {low,high},%1;
  cvt.f32.f16 %0, high;}

#line 2371
return val; 
#line 2372
} 
#endif
#line 2373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __lows2half2(const __half2 l, const __half2 h) 
#line 2374
{int volatile ___ = 1;(void)l;(void)h;
#line 2381
::exit(___);}
#if 0
#line 2374
{ 
#line 2375
__half2 val; 
#line 2376
__asm {.reg .f16 alow,ahigh,blow,bhigh;
  mov.b32 {alow,ahigh}, %1;
  mov.b32 {blow,bhigh}, %2;
  mov.b32 %0, {alow,blow};}

#line 2380
return val; 
#line 2381
} 
#endif
#line 2382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __highs2half2(const __half2 l, const __half2 h) 
#line 2383
{int volatile ___ = 1;(void)l;(void)h;
#line 2390
::exit(___);}
#if 0
#line 2383
{ 
#line 2384
__half2 val; 
#line 2385
__asm {.reg .f16 alow,ahigh,blow,bhigh;
  mov.b32 {alow,ahigh}, %1;
  mov.b32 {blow,bhigh}, %2;
  mov.b32 %0, {ahigh,bhigh};}

#line 2389
return val; 
#line 2390
} 
#endif
#line 2391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __low2half(const __half2 h) 
#line 2392
{int volatile ___ = 1;(void)h;
#line 2398
::exit(___);}
#if 0
#line 2392
{ 
#line 2393
__half ret; 
#line 2394
__asm {.reg .f16 low,high;
 mov.b32 {low,high}, %1;
 mov.b16 %0, low;}
return ret; 
#line 2398
} 
#endif
#line 2399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline int __hisinf(const __half a) 
#line 2400
{int volatile ___ = 1;(void)a;
#line 2406
::exit(___);}
#if 0
#line 2400
{ 
#line 2401
if ((a.x) == 64512) { 
#line 2402
return -1; }  
#line 2403
if ((a.x) == 31744) { 
#line 2404
return 1; }  
#line 2405
return 0; 
#line 2406
} 
#endif
#line 2407 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __low2half2(const __half2 l) 
#line 2408
{int volatile ___ = 1;(void)l;
#line 2414
::exit(___);}
#if 0
#line 2408
{ 
#line 2409
__half2 val; 
#line 2410
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  mov.b32 %0, {low,low};}

#line 2413
return val; 
#line 2414
} 
#endif
#line 2415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __high2half2(const __half2 l) 
#line 2416
{int volatile ___ = 1;(void)l;
#line 2422
::exit(___);}
#if 0
#line 2416
{ 
#line 2417
__half2 val; 
#line 2418
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  mov.b32 %0, {high,high};}

#line 2421
return val; 
#line 2422
} 
#endif
#line 2423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __high2half(const __half2 h) 
#line 2424
{int volatile ___ = 1;(void)h;
#line 2430
::exit(___);}
#if 0
#line 2424
{ 
#line 2425
__half ret; 
#line 2426
__asm {.reg .f16 low,high;
 mov.b32 {low,high}, %1;
 mov.b16 %0, high;}
return ret; 
#line 2430
} 
#endif
#line 2431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __halves2half2(const __half l, const __half h) 
#line 2432
{int volatile ___ = 1;(void)l;(void)h;
#line 2437
::exit(___);}
#if 0
#line 2432
{ 
#line 2433
__half2 val; 
#line 2434
__asm {  mov.b32 %0, {%1,%2};}

return val; 
#line 2437
} 
#endif
#line 2438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __half2half2(const __half lh) 
#line 2439
{int volatile ___ = 1;(void)lh;
#line 2444
::exit(___);}
#if 0
#line 2439
{ 
#line 2440
__half2 val; 
#line 2441
__asm {  mov.b32 %0, {%1,%1};}

return val; 
#line 2444
} 
#endif
#line 2445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __lowhigh2highlow(const __half2 lh) 
#line 2446
{int volatile ___ = 1;(void)lh;
#line 2452
::exit(___);}
#if 0
#line 2446
{ 
#line 2447
__half2 val; 
#line 2448
__asm {.reg .f16 low,high;
  mov.b32 {low,high}, %1;
  mov.b32 %0, {high,low};}

#line 2451
return val; 
#line 2452
} 
#endif
#line 2453 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline short __half_as_short(const __half h) 
#line 2454
{int volatile ___ = 1;(void)h;
#line 2456
::exit(___);}
#if 0
#line 2454
{ 
#line 2455
return (short)(h.x); 
#line 2456
} 
#endif
#line 2457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline unsigned short __half_as_ushort(const __half h) 
#line 2458
{int volatile ___ = 1;(void)h;
#line 2460
::exit(___);}
#if 0
#line 2458
{ 
#line 2459
return h.x; 
#line 2460
} 
#endif
#line 2461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __short_as_half(const short i) 
#line 2462
{int volatile ___ = 1;(void)i;
#line 2466
::exit(___);}
#if 0
#line 2462
{ 
#line 2463
__half h; 
#line 2464
(h.x) = ((unsigned short)i); 
#line 2465
return h; 
#line 2466
} 
#endif
#line 2467 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ushort_as_half(const unsigned short i) 
#line 2468
{int volatile ___ = 1;(void)i;
#line 2472
::exit(___);}
#if 0
#line 2468
{ 
#line 2469
__half h; 
#line 2470
(h.x) = i; 
#line 2471
return h; 
#line 2472
} 
#endif
#line 2484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __shfl(__half2 var, int delta, int width) 
#line 2485
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2490
::exit(___);}
#if 0
#line 2485
{ 
#line 2486
int warpSize; 
#line 2487
__asm {mov.u32 %0, WARP_SZ;
}
#line 2488
int c = ((warpSize - width) << 8) | 31; 
#line 2489
do { __half2 r; __asm {shfl.idx.b32 %0,%1,%2,%3;
}
#line 2489
return r; } while (0); ; 
#line 2490
} 
#endif
#line 2491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __shfl_up(__half2 var, unsigned delta, int width) 
#line 2492
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2497
::exit(___);}
#if 0
#line 2492
{ 
#line 2493
int warpSize; 
#line 2494
__asm {mov.u32 %0, WARP_SZ;
}
#line 2495
int c = (warpSize - width) << 8; 
#line 2496
do { __half2 r; __asm {shfl.up.b32 %0,%1,%2,%3;
}
#line 2496
return r; } while (0); ; 
#line 2497
} 
#endif
#line 2498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __shfl_down(__half2 var, unsigned delta, int width) 
#line 2499
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2504
::exit(___);}
#if 0
#line 2499
{ 
#line 2500
int warpSize; 
#line 2501
__asm {mov.u32 %0, WARP_SZ;
}
#line 2502
int c = ((warpSize - width) << 8) | 31; 
#line 2503
do { __half2 r; __asm {shfl.down.b32 %0,%1,%2,%3;
}
#line 2503
return r; } while (0); ; 
#line 2504
} 
#endif
#line 2505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __shfl_xor(__half2 var, int delta, int width) 
#line 2506
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2511
::exit(___);}
#if 0
#line 2506
{ 
#line 2507
int warpSize; 
#line 2508
__asm {mov.u32 %0, WARP_SZ;
}
#line 2509
int c = ((warpSize - width) << 8) | 31; 
#line 2510
do { __half2 r; __asm {shfl.bfly.b32 %0,%1,%2,%3;
}
#line 2510
return r; } while (0); ; 
#line 2511
} 
#endif
#line 2513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __shfl(__half var, int delta, int width) 
#line 2514
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2518
::exit(___);}
#if 0
#line 2514
{ 
#line 2515
__half2 temp1 = __halves2half2(var, var); 
#line 2516
__half2 temp2 = __shfl(temp1, delta, width); 
#line 2517
return __low2half(temp2); 
#line 2518
} 
#endif
#line 2519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __shfl_up(__half var, unsigned delta, int width) 
#line 2520
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2524
::exit(___);}
#if 0
#line 2520
{ 
#line 2521
__half2 temp1 = __halves2half2(var, var); 
#line 2522
__half2 temp2 = __shfl_up(temp1, delta, width); 
#line 2523
return __low2half(temp2); 
#line 2524
} 
#endif
#line 2525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __shfl_down(__half var, unsigned delta, int width) 
#line 2526
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2530
::exit(___);}
#if 0
#line 2526
{ 
#line 2527
__half2 temp1 = __halves2half2(var, var); 
#line 2528
__half2 temp2 = __shfl_down(temp1, delta, width); 
#line 2529
return __low2half(temp2); 
#line 2530
} 
#endif
#line 2531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __shfl_xor(__half var, int delta, int width) 
#line 2532
{int volatile ___ = 1;(void)var;(void)delta;(void)width;
#line 2536
::exit(___);}
#if 0
#line 2532
{ 
#line 2533
__half2 temp1 = __halves2half2(var, var); 
#line 2534
__half2 temp2 = __shfl_xor(temp1, delta, width); 
#line 2535
return __low2half(temp2); 
#line 2536
} 
#endif
#line 2548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __ldg(const __half2 *ptr) 
#line 2549
{int volatile ___ = 1;(void)ptr;
#line 2553
::exit(___);}
#if 0
#line 2549
{ 
#line 2550
__half2 ret; 
#line 2551
__asm ld.global.nc.b32 %0, [%1];
return ret; 
#line 2553
} 
#endif
#line 2554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ldg(const __half *ptr) 
#line 2555
{int volatile ___ = 1;(void)ptr;
#line 2559
::exit(___);}
#if 0
#line 2555
{ 
#line 2556
__half ret; 
#line 2557
__asm ld.global.nc.b16 %0, [%1];
return ret; 
#line 2559
} 
#endif
#line 2560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __ldcg(const __half2 *ptr) 
#line 2561
{int volatile ___ = 1;(void)ptr;
#line 2565
::exit(___);}
#if 0
#line 2561
{ 
#line 2562
__half2 ret; 
#line 2563
__asm ld.global.cg.b32 %0, [%1];
return ret; 
#line 2565
} 
#endif
#line 2566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ldcg(const __half *ptr) 
#line 2567
{int volatile ___ = 1;(void)ptr;
#line 2571
::exit(___);}
#if 0
#line 2567
{ 
#line 2568
__half ret; 
#line 2569
__asm ld.global.cg.b16 %0, [%1];
return ret; 
#line 2571
} 
#endif
#line 2572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __ldca(const __half2 *ptr) 
#line 2573
{int volatile ___ = 1;(void)ptr;
#line 2577
::exit(___);}
#if 0
#line 2573
{ 
#line 2574
__half2 ret; 
#line 2575
__asm ld.global.ca.b32 %0, [%1];
return ret; 
#line 2577
} 
#endif
#line 2578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ldca(const __half *ptr) 
#line 2579
{int volatile ___ = 1;(void)ptr;
#line 2583
::exit(___);}
#if 0
#line 2579
{ 
#line 2580
__half ret; 
#line 2581
__asm ld.global.ca.b16 %0, [%1];
return ret; 
#line 2583
} 
#endif
#line 2584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __ldcs(const __half2 *ptr) 
#line 2585
{int volatile ___ = 1;(void)ptr;
#line 2589
::exit(___);}
#if 0
#line 2585
{ 
#line 2586
__half2 ret; 
#line 2587
__asm ld.global.cs.b32 %0, [%1];
return ret; 
#line 2589
} 
#endif
#line 2590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __ldcs(const __half *ptr) 
#line 2591
{int volatile ___ = 1;(void)ptr;
#line 2595
::exit(___);}
#if 0
#line 2591
{ 
#line 2592
__half ret; 
#line 2593
__asm ld.global.cs.b16 %0, [%1];
return ret; 
#line 2595
} 
#endif
#line 2608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __heq2(const __half2 a, const __half2 b) 
#line 2609
{int volatile ___ = 1;(void)a;(void)b;
#line 2611
::exit(___);}
#if 0
#line 2609
{ 
#line 2610
do { __half2 val; __asm { set.eq.f16x2.f16x2 %0,%1,%2;
}
#line 2610
return val; } while (0); ; 
#line 2611
} 
#endif
#line 2612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hne2(const __half2 a, const __half2 b) 
#line 2613
{int volatile ___ = 1;(void)a;(void)b;
#line 2615
::exit(___);}
#if 0
#line 2613
{ 
#line 2614
do { __half2 val; __asm { set.ne.f16x2.f16x2 %0,%1,%2;
}
#line 2614
return val; } while (0); ; 
#line 2615
} 
#endif
#line 2616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hle2(const __half2 a, const __half2 b) 
#line 2617
{int volatile ___ = 1;(void)a;(void)b;
#line 2619
::exit(___);}
#if 0
#line 2617
{ 
#line 2618
do { __half2 val; __asm { set.le.f16x2.f16x2 %0,%1,%2;
}
#line 2618
return val; } while (0); ; 
#line 2619
} 
#endif
#line 2620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hge2(const __half2 a, const __half2 b) 
#line 2621
{int volatile ___ = 1;(void)a;(void)b;
#line 2623
::exit(___);}
#if 0
#line 2621
{ 
#line 2622
do { __half2 val; __asm { set.ge.f16x2.f16x2 %0,%1,%2;
}
#line 2622
return val; } while (0); ; 
#line 2623
} 
#endif
#line 2624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hlt2(const __half2 a, const __half2 b) 
#line 2625
{int volatile ___ = 1;(void)a;(void)b;
#line 2627
::exit(___);}
#if 0
#line 2625
{ 
#line 2626
do { __half2 val; __asm { set.lt.f16x2.f16x2 %0,%1,%2;
}
#line 2626
return val; } while (0); ; 
#line 2627
} 
#endif
#line 2628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hgt2(const __half2 a, const __half2 b) 
#line 2629
{int volatile ___ = 1;(void)a;(void)b;
#line 2631
::exit(___);}
#if 0
#line 2629
{ 
#line 2630
do { __half2 val; __asm { set.gt.f16x2.f16x2 %0,%1,%2;
}
#line 2630
return val; } while (0); ; 
#line 2631
} 
#endif
#line 2632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hequ2(const __half2 a, const __half2 b) 
#line 2633
{int volatile ___ = 1;(void)a;(void)b;
#line 2635
::exit(___);}
#if 0
#line 2633
{ 
#line 2634
do { __half2 val; __asm { set.equ.f16x2.f16x2 %0,%1,%2;
}
#line 2634
return val; } while (0); ; 
#line 2635
} 
#endif
#line 2636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hneu2(const __half2 a, const __half2 b) 
#line 2637
{int volatile ___ = 1;(void)a;(void)b;
#line 2639
::exit(___);}
#if 0
#line 2637
{ 
#line 2638
do { __half2 val; __asm { set.neu.f16x2.f16x2 %0,%1,%2;
}
#line 2638
return val; } while (0); ; 
#line 2639
} 
#endif
#line 2640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hleu2(const __half2 a, const __half2 b) 
#line 2641
{int volatile ___ = 1;(void)a;(void)b;
#line 2643
::exit(___);}
#if 0
#line 2641
{ 
#line 2642
do { __half2 val; __asm { set.leu.f16x2.f16x2 %0,%1,%2;
}
#line 2642
return val; } while (0); ; 
#line 2643
} 
#endif
#line 2644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hgeu2(const __half2 a, const __half2 b) 
#line 2645
{int volatile ___ = 1;(void)a;(void)b;
#line 2647
::exit(___);}
#if 0
#line 2645
{ 
#line 2646
do { __half2 val; __asm { set.geu.f16x2.f16x2 %0,%1,%2;
}
#line 2646
return val; } while (0); ; 
#line 2647
} 
#endif
#line 2648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hltu2(const __half2 a, const __half2 b) 
#line 2649
{int volatile ___ = 1;(void)a;(void)b;
#line 2651
::exit(___);}
#if 0
#line 2649
{ 
#line 2650
do { __half2 val; __asm { set.ltu.f16x2.f16x2 %0,%1,%2;
}
#line 2650
return val; } while (0); ; 
#line 2651
} 
#endif
#line 2652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hgtu2(const __half2 a, const __half2 b) 
#line 2653
{int volatile ___ = 1;(void)a;(void)b;
#line 2655
::exit(___);}
#if 0
#line 2653
{ 
#line 2654
do { __half2 val; __asm { set.gtu.f16x2.f16x2 %0,%1,%2;
}
#line 2654
return val; } while (0); ; 
#line 2655
} 
#endif
#line 2666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbeq2(const __half2 a, const __half2 b) 
#line 2667
{int volatile ___ = 1;(void)a;(void)b;
#line 2669
::exit(___);}
#if 0
#line 2667
{ 
#line 2668
do { __half2 val; __asm { set.eq.f16x2.f16x2 %0,%1,%2;
}
#line 2668
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2669
} 
#endif
#line 2670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbne2(const __half2 a, const __half2 b) 
#line 2671
{int volatile ___ = 1;(void)a;(void)b;
#line 2673
::exit(___);}
#if 0
#line 2671
{ 
#line 2672
do { __half2 val; __asm { set.ne.f16x2.f16x2 %0,%1,%2;
}
#line 2672
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2673
} 
#endif
#line 2674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hble2(const __half2 a, const __half2 b) 
#line 2675
{int volatile ___ = 1;(void)a;(void)b;
#line 2677
::exit(___);}
#if 0
#line 2675
{ 
#line 2676
do { __half2 val; __asm { set.le.f16x2.f16x2 %0,%1,%2;
}
#line 2676
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2677
} 
#endif
#line 2678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbge2(const __half2 a, const __half2 b) 
#line 2679
{int volatile ___ = 1;(void)a;(void)b;
#line 2681
::exit(___);}
#if 0
#line 2679
{ 
#line 2680
do { __half2 val; __asm { set.ge.f16x2.f16x2 %0,%1,%2;
}
#line 2680
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2681
} 
#endif
#line 2682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hblt2(const __half2 a, const __half2 b) 
#line 2683
{int volatile ___ = 1;(void)a;(void)b;
#line 2685
::exit(___);}
#if 0
#line 2683
{ 
#line 2684
do { __half2 val; __asm { set.lt.f16x2.f16x2 %0,%1,%2;
}
#line 2684
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2685
} 
#endif
#line 2686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbgt2(const __half2 a, const __half2 b) 
#line 2687
{int volatile ___ = 1;(void)a;(void)b;
#line 2689
::exit(___);}
#if 0
#line 2687
{ 
#line 2688
do { __half2 val; __asm { set.gt.f16x2.f16x2 %0,%1,%2;
}
#line 2688
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2689
} 
#endif
#line 2690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbequ2(const __half2 a, const __half2 b) 
#line 2691
{int volatile ___ = 1;(void)a;(void)b;
#line 2693
::exit(___);}
#if 0
#line 2691
{ 
#line 2692
do { __half2 val; __asm { set.equ.f16x2.f16x2 %0,%1,%2;
}
#line 2692
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2693
} 
#endif
#line 2694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbneu2(const __half2 a, const __half2 b) 
#line 2695
{int volatile ___ = 1;(void)a;(void)b;
#line 2697
::exit(___);}
#if 0
#line 2695
{ 
#line 2696
do { __half2 val; __asm { set.neu.f16x2.f16x2 %0,%1,%2;
}
#line 2696
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2697
} 
#endif
#line 2698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbleu2(const __half2 a, const __half2 b) 
#line 2699
{int volatile ___ = 1;(void)a;(void)b;
#line 2701
::exit(___);}
#if 0
#line 2699
{ 
#line 2700
do { __half2 val; __asm { set.leu.f16x2.f16x2 %0,%1,%2;
}
#line 2700
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2701
} 
#endif
#line 2702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbgeu2(const __half2 a, const __half2 b) 
#line 2703
{int volatile ___ = 1;(void)a;(void)b;
#line 2705
::exit(___);}
#if 0
#line 2703
{ 
#line 2704
do { __half2 val; __asm { set.geu.f16x2.f16x2 %0,%1,%2;
}
#line 2704
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2705
} 
#endif
#line 2706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbltu2(const __half2 a, const __half2 b) 
#line 2707
{int volatile ___ = 1;(void)a;(void)b;
#line 2709
::exit(___);}
#if 0
#line 2707
{ 
#line 2708
do { __half2 val; __asm { set.ltu.f16x2.f16x2 %0,%1,%2;
}
#line 2708
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2709
} 
#endif
#line 2710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hbgtu2(const __half2 a, const __half2 b) 
#line 2711
{int volatile ___ = 1;(void)a;(void)b;
#line 2713
::exit(___);}
#if 0
#line 2711
{ 
#line 2712
do { __half2 val; __asm { set.gtu.f16x2.f16x2 %0,%1,%2;
}
#line 2712
if ((val.x) == (1006648320)) { return true; } else { return false; }  } while (0); ; 
#line 2713
} 
#endif
#line 2726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __heq(const __half a, const __half b) 
#line 2727
{int volatile ___ = 1;(void)a;(void)b;
#line 2729
::exit(___);}
#if 0
#line 2727
{ 
#line 2728
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.eq.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2728
return (val) ? true : false; } while (0); ; 
#line 2729
} 
#endif
#line 2730 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hne(const __half a, const __half b) 
#line 2731
{int volatile ___ = 1;(void)a;(void)b;
#line 2733
::exit(___);}
#if 0
#line 2731
{ 
#line 2732
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.ne.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2732
return (val) ? true : false; } while (0); ; 
#line 2733
} 
#endif
#line 2734 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hle(const __half a, const __half b) 
#line 2735
{int volatile ___ = 1;(void)a;(void)b;
#line 2737
::exit(___);}
#if 0
#line 2735
{ 
#line 2736
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.le.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2736
return (val) ? true : false; } while (0); ; 
#line 2737
} 
#endif
#line 2738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hge(const __half a, const __half b) 
#line 2739
{int volatile ___ = 1;(void)a;(void)b;
#line 2741
::exit(___);}
#if 0
#line 2739
{ 
#line 2740
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.ge.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2740
return (val) ? true : false; } while (0); ; 
#line 2741
} 
#endif
#line 2742 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hlt(const __half a, const __half b) 
#line 2743
{int volatile ___ = 1;(void)a;(void)b;
#line 2745
::exit(___);}
#if 0
#line 2743
{ 
#line 2744
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.lt.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2744
return (val) ? true : false; } while (0); ; 
#line 2745
} 
#endif
#line 2746 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hgt(const __half a, const __half b) 
#line 2747
{int volatile ___ = 1;(void)a;(void)b;
#line 2749
::exit(___);}
#if 0
#line 2747
{ 
#line 2748
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.gt.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2748
return (val) ? true : false; } while (0); ; 
#line 2749
} 
#endif
#line 2750 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hequ(const __half a, const __half b) 
#line 2751
{int volatile ___ = 1;(void)a;(void)b;
#line 2753
::exit(___);}
#if 0
#line 2751
{ 
#line 2752
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.equ.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2752
return (val) ? true : false; } while (0); ; 
#line 2753
} 
#endif
#line 2754 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hneu(const __half a, const __half b) 
#line 2755
{int volatile ___ = 1;(void)a;(void)b;
#line 2757
::exit(___);}
#if 0
#line 2755
{ 
#line 2756
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.neu.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2756
return (val) ? true : false; } while (0); ; 
#line 2757
} 
#endif
#line 2758 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hleu(const __half a, const __half b) 
#line 2759
{int volatile ___ = 1;(void)a;(void)b;
#line 2761
::exit(___);}
#if 0
#line 2759
{ 
#line 2760
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.leu.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2760
return (val) ? true : false; } while (0); ; 
#line 2761
} 
#endif
#line 2762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hgeu(const __half a, const __half b) 
#line 2763
{int volatile ___ = 1;(void)a;(void)b;
#line 2765
::exit(___);}
#if 0
#line 2763
{ 
#line 2764
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.geu.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2764
return (val) ? true : false; } while (0); ; 
#line 2765
} 
#endif
#line 2766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hltu(const __half a, const __half b) 
#line 2767
{int volatile ___ = 1;(void)a;(void)b;
#line 2769
::exit(___);}
#if 0
#line 2767
{ 
#line 2768
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.ltu.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2768
return (val) ? true : false; } while (0); ; 
#line 2769
} 
#endif
#line 2770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hgtu(const __half a, const __half b) 
#line 2771
{int volatile ___ = 1;(void)a;(void)b;
#line 2773
::exit(___);}
#if 0
#line 2771
{ 
#line 2772
do { unsigned short val; __asm { .reg .pred __$temp3;
  setp.gtu.f16  __$temp3, %1, %2;
  selp.u16 %0, 1, 0, __$temp3;}
#line 2772
return (val) ? true : false; } while (0); ; 
#line 2773
} 
#endif
#line 2784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hadd2(const __half2 a, const __half2 b) 
#line 2785
{int volatile ___ = 1;(void)a;(void)b;
#line 2787
::exit(___);}
#if 0
#line 2785
{ 
#line 2786
do { __half2 val; __asm {add.f16x2 %0,%1,%2;
}
#line 2786
return val; } while (0); ; 
#line 2787
} 
#endif
#line 2788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hsub2(const __half2 a, const __half2 b) 
#line 2789
{int volatile ___ = 1;(void)a;(void)b;
#line 2791
::exit(___);}
#if 0
#line 2789
{ 
#line 2790
do { __half2 val; __asm {sub.f16x2 %0,%1,%2;
}
#line 2790
return val; } while (0); ; 
#line 2791
} 
#endif
#line 2792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hmul2(const __half2 a, const __half2 b) 
#line 2793
{int volatile ___ = 1;(void)a;(void)b;
#line 2795
::exit(___);}
#if 0
#line 2793
{ 
#line 2794
do { __half2 val; __asm {mul.f16x2 %0,%1,%2;
}
#line 2794
return val; } while (0); ; 
#line 2795
} 
#endif
#line 2796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hadd2_sat(const __half2 a, const __half2 b) 
#line 2797
{int volatile ___ = 1;(void)a;(void)b;
#line 2799
::exit(___);}
#if 0
#line 2797
{ 
#line 2798
do { __half2 val; __asm {add.sat.f16x2 %0,%1,%2;
}
#line 2798
return val; } while (0); ; 
#line 2799
} 
#endif
#line 2800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hsub2_sat(const __half2 a, const __half2 b) 
#line 2801
{int volatile ___ = 1;(void)a;(void)b;
#line 2803
::exit(___);}
#if 0
#line 2801
{ 
#line 2802
do { __half2 val; __asm {sub.sat.f16x2 %0,%1,%2;
}
#line 2802
return val; } while (0); ; 
#line 2803
} 
#endif
#line 2804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hmul2_sat(const __half2 a, const __half2 b) 
#line 2805
{int volatile ___ = 1;(void)a;(void)b;
#line 2807
::exit(___);}
#if 0
#line 2805
{ 
#line 2806
do { __half2 val; __asm {mul.sat.f16x2 %0,%1,%2;
}
#line 2806
return val; } while (0); ; 
#line 2807
} 
#endif
#line 2815 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c) 
#line 2816
{int volatile ___ = 1;(void)a;(void)b;(void)c;
#line 2818
::exit(___);}
#if 0
#line 2816
{ 
#line 2817
do { __half2 val; __asm {fma.rn.f16x2 %0,%1,%2,%3;
}
#line 2817
return val; } while (0); ; 
#line 2818
} 
#endif
#line 2819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c) 
#line 2820
{int volatile ___ = 1;(void)a;(void)b;(void)c;
#line 2822
::exit(___);}
#if 0
#line 2820
{ 
#line 2821
do { __half2 val; __asm {fma.rn.sat.f16x2 %0,%1,%2,%3;
}
#line 2821
return val; } while (0); ; 
#line 2822
} 
#endif
#line 2824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2div(__half2 a, __half2 b) {int volatile ___ = 1;(void)a;(void)b;
#line 2838
::exit(___);}
#if 0
#line 2824
{ 
#line 2825
__half ha, hb; 
#line 2827
ha = __low2half(a); 
#line 2828
hb = __low2half(b); 
#line 2830
__half v1 = hdiv(ha, hb); 
#line 2832
ha = __high2half(a); 
#line 2833
hb = __high2half(b); 
#line 2835
__half v2 = hdiv(ha, hb); 
#line 2837
return __halves2half2(v1, v2); 
#line 2838
} 
#endif
#line 2848 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hadd(const __half a, const __half b) 
#line 2849
{int volatile ___ = 1;(void)a;(void)b;
#line 2851
::exit(___);}
#if 0
#line 2849
{ 
#line 2850
do { __half val; __asm {add.f16 %0,%1,%2;
}
#line 2850
return val; } while (0); ; 
#line 2851
} 
#endif
#line 2852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hsub(const __half a, const __half b) 
#line 2853
{int volatile ___ = 1;(void)a;(void)b;
#line 2855
::exit(___);}
#if 0
#line 2853
{ 
#line 2854
do { __half val; __asm {sub.f16 %0,%1,%2;
}
#line 2854
return val; } while (0); ; 
#line 2855
} 
#endif
#line 2856 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hmul(const __half a, const __half b) 
#line 2857
{int volatile ___ = 1;(void)a;(void)b;
#line 2859
::exit(___);}
#if 0
#line 2857
{ 
#line 2858
do { __half val; __asm {mul.f16 %0,%1,%2;
}
#line 2858
return val; } while (0); ; 
#line 2859
} 
#endif
#line 2860 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hadd_sat(const __half a, const __half b) 
#line 2861
{int volatile ___ = 1;(void)a;(void)b;
#line 2863
::exit(___);}
#if 0
#line 2861
{ 
#line 2862
do { __half val; __asm {add.sat.f16 %0,%1,%2;
}
#line 2862
return val; } while (0); ; 
#line 2863
} 
#endif
#line 2864 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hsub_sat(const __half a, const __half b) 
#line 2865
{int volatile ___ = 1;(void)a;(void)b;
#line 2867
::exit(___);}
#if 0
#line 2865
{ 
#line 2866
do { __half val; __asm {sub.sat.f16 %0,%1,%2;
}
#line 2866
return val; } while (0); ; 
#line 2867
} 
#endif
#line 2868 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hmul_sat(const __half a, const __half b) 
#line 2869
{int volatile ___ = 1;(void)a;(void)b;
#line 2871
::exit(___);}
#if 0
#line 2869
{ 
#line 2870
do { __half val; __asm {mul.sat.f16 %0,%1,%2;
}
#line 2870
return val; } while (0); ; 
#line 2871
} 
#endif
#line 2879 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hfma(const __half a, const __half b, const __half c) 
#line 2880
{int volatile ___ = 1;(void)a;(void)b;(void)c;
#line 2882
::exit(___);}
#if 0
#line 2880
{ 
#line 2881
do { __half val; __asm {fma.rn.f16 %0,%1,%2,%3;
}
#line 2881
return val; } while (0); ; 
#line 2882
} 
#endif
#line 2883 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hfma_sat(const __half a, const __half b, const __half c) 
#line 2884
{int volatile ___ = 1;(void)a;(void)b;(void)c;
#line 2886
::exit(___);}
#if 0
#line 2884
{ 
#line 2885
do { __half val; __asm {fma.rn.sat.f16 %0,%1,%2,%3;
}
#line 2885
return val; } while (0); ; 
#line 2886
} 
#endif
#line 2888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hdiv(__half a, __half b) {int volatile ___ = 1;(void)a;(void)b;
#line 2908
::exit(___);}
#if 0
#line 2888
{ 
#line 2889
__half v, abs, den; 
#line 2890
(den.x) = (143); 
#line 2891
float fa, fb, fv, rcp; 
#line 2893
fa = __half2float(a); 
#line 2894
fb = __half2float(b); 
#line 2896
__asm {rcp.approx.f32 %0, %1;
}
fv = (rcp * fa); 
#line 2900
v = __float2half(fv); 
#line 2901
(abs.x) = ((unsigned short)(((unsigned)(v.x)) & (32767))); 
#line 2902
if (__hlt(abs, den) && (!((abs.x) == 0))) { 
#line 2903
float err = __fmaf_rn(-fb, fv, fa); 
#line 2904
fv = __fmaf_rn(rcp, err, fv); 
#line 2905
v = __float2half(fv); 
#line 2906
}  
#line 2907
return v; 
#line 2908
} 
#endif
#line 2952 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __forceinline float __float_simpl_sinf(float); 
#line 2953
static __forceinline float __float_simpl_cosf(float); 
#line 2954
static __inline __half __hsin_internal(const __half a) {int volatile ___ = 1;(void)a;
#line 2960
::exit(___);}
#if 0
#line 2954
{ 
#line 2955
__half val; 
#line 2956
float f = __half2float(a); 
#line 2957
f = __float_simpl_sinf(f); 
#line 2958
(val.x) = __float2half_rn(f); 
#line 2959
return val; 
#line 2960
} 
#endif
#line 2961 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hsin(const __half a) {int volatile ___ = 1;(void)a;
#line 2977
::exit(___);}
#if 0
#line 2961
{ 
#line 2962
__half r = __hsin_internal(a); 
#line 2963
__asm {
	  .reg.b16 i,r,t;     
	  mov.b16 r, %0;      
	  mov.b16 i, %1;      
	  mov.b16 t, 0x8000;  
	  and.b16 t,r,t;      
	{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X32B3;
  mov.b16 ulp,0x0800;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X5CB0;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0XB2B3;
  mov.b16 ulp,0x8800;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0XDCB0;
  mov.b16 ulp,0x9000;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
  or.b16  r,r,t;      
	  mov.b16 %0, r;      
}

#line 2976
return r; 
#line 2977
} 
#endif
#line 2978 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2sin(const __half2 a) {int volatile ___ = 1;(void)a;
#line 2995
::exit(___);}
#if 0
#line 2978
{ 
#line 2979
__half l = __low2half(a); 
#line 2980
__half h = __high2half(a); 
#line 2981
__half2 r = __halves2half2(__hsin_internal(l), __hsin_internal(h)); 
#line 2982
__asm {
	  .reg.b32 i,r,t;             
	  mov.b32 r, %0;              
	  mov.b32 i, %1;              
	  and.b32 t, r, 0x80008000;   
	{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X32B332B3;
  mov.b32 ulp,0x08000800;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X5CB05CB0;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0XB2B3B2B3;
  mov.b32 ulp,0x88008800;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0XDCB0DCB0;
  mov.b32 ulp,0x90009000;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  or.b32  r, r, t;            
	  mov.b32 %0, r;              
}

#line 2994
return r; 
#line 2995
} 
#endif
#line 2996 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hcos_internal(const __half a) {int volatile ___ = 1;(void)a;
#line 3002
::exit(___);}
#if 0
#line 2996
{ 
#line 2997
__half val; 
#line 2998
float f = __half2float(a); 
#line 2999
f = __float_simpl_cosf(f); 
#line 3000
(val.x) = __float2half_rn(f); 
#line 3001
return val; 
#line 3002
} 
#endif
#line 3003 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hcos(const __half a) {int volatile ___ = 1;(void)a;
#line 3014
::exit(___);}
#if 0
#line 3003
{ 
#line 3004
__half r = __hcos_internal(a); 
#line 3005
__asm {
	  .reg.b16 i,r;        
	  mov.b16 r, %0;       
	  mov.b16 i, %1;       
	{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X2B7C;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0XAB7C;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,i, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16 %0, r;       
}

#line 3013
return r; 
#line 3014
} 
#endif
#line 3015 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2cos(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3028
::exit(___);}
#if 0
#line 3015
{ 
#line 3016
__half l = __low2half(a); 
#line 3017
__half h = __high2half(a); 
#line 3018
__half2 r = __halves2half2(__hcos_internal(l), __hcos_internal(h)); 
#line 3019
__asm {
	  .reg.b32 i,r;   
	  mov.b32 r, %0;  
	  mov.b32 i, %1;  
	{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X2B7C2B7C;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0XAB7CAB7C;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,i, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32 %0, r;  
}

#line 3027
return r; 
#line 3028
} 
#endif
#line 3029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __forceinline float __internal_trig_reduction_kernel(float a, int *quadrant) 
#line 3030
{int volatile ___ = 1;(void)a;(void)quadrant;
#line 3039
::exit(___);}
#if 0
#line 3030
{ 
#line 3031
float j, t; 
#line 3032
int q; 
#line 3033
q = __float2int_rn(a * (0.6366197467F)); 
#line 3034
j = ((float)q); 
#line 3035
t = __fmaf_rn(-j, (1.570796251F), a); 
#line 3036
t = __fmaf_rn(-j, (7.549789416e-008F), t); 
#line 3037
(*quadrant) = q; 
#line 3038
return t; 
#line 3039
} 
#endif
#line 3040 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __forceinline float __internal_sin_cos_kernel(float x, int i) 
#line 3041
{int volatile ___ = 1;(void)x;(void)i;
#line 3063
::exit(___);}
#if 0
#line 3041
{ 
#line 3042
float x2, z; 
#line 3043
x2 = (x * x); 
#line 3045
if (i & 1) { 
#line 3046
z = (2.44331568e-005F); 
#line 3047
z = __fmaf_rn(z, x2, -(0.001388731645F)); 
#line 3048
} else { 
#line 3049
z = (-(0.0001951529557F)); 
#line 3050
z = __fmaf_rn(z, x2, (0.008332161233F)); 
#line 3051
}  
#line 3052
if (i & 1) { 
#line 3053
z = __fmaf_rn(z, x2, (0.04166664556F)); 
#line 3054
z = __fmaf_rn(z, x2, -(0.5F)); 
#line 3055
} else { 
#line 3056
z = __fmaf_rn(z, x2, -(0.1666665524F)); 
#line 3057
z = __fmaf_rn(z, x2, (0.0F)); 
#line 3058
}  
#line 3059
x = __fmaf_rn(z, x, x); 
#line 3060
if (i & 1) { x = __fmaf_rn(z, x2, (1.0F)); }  
#line 3061
if (i & 2) { x = __fmaf_rn(x, -(1.0F), (0.0F)); }  
#line 3062
return x; 
#line 3063
} 
#endif
#line 3064 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __forceinline float __float_simpl_sinf(float a) 
#line 3065
{int volatile ___ = 1;(void)a;
#line 3074
::exit(___);}
#if 0
#line 3065
{ 
#line 3066
float z; 
#line 3067
int i; 
#line 3068
if (isinf(a)) { 
#line 3069
a = (a * (0.0F)); 
#line 3070
}  
#line 3071
a = __internal_trig_reduction_kernel(a, &i); 
#line 3072
z = __internal_sin_cos_kernel(a, i); 
#line 3073
return z; 
#line 3074
} 
#endif
#line 3075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __forceinline float __float_simpl_cosf(float a) 
#line 3076
{int volatile ___ = 1;(void)a;
#line 3086
::exit(___);}
#if 0
#line 3076
{ 
#line 3077
float z; 
#line 3078
int i; 
#line 3079
if (isinf(a)) { 
#line 3080
a = (a * (0.0F)); 
#line 3081
}  
#line 3082
a = __internal_trig_reduction_kernel(a, &i); 
#line 3083
i++; 
#line 3084
z = __internal_sin_cos_kernel(a, i); 
#line 3085
return z; 
#line 3086
} 
#endif
#line 3087 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hexp(const __half a) {int volatile ___ = 1;(void)a;
#line 3104
::exit(___);}
#if 0
#line 3087
{ 
#line 3088
__half val; 
#line 3089
__asm {.reg.b32         f, C;           
 .reg.b16         h,r;            
  mov.b16         h,%1;           
  cvt.f32.f16     f,h;            
  mov.b32         C, 0x3fb8aa3b;  
  mul.f32         f,f,C;          
  ex2.approx.f32      f,f;        
  cvt.rn.f16.f32      r,f;        
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X1F79;
  mov.b16 ulp,0x9400;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X25CF;
  mov.b16 ulp,0x9400;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0XC13B;
  mov.b16 ulp,0x0400;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0XC1EF;
  mov.b16 ulp,0x0200;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16         %0,r;           
}
#line 3103
return val; 
#line 3104
} 
#endif
#line 3105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2exp(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3128
::exit(___);}
#if 0
#line 3105
{ 
#line 3106
__half2 val; 
#line 3107
__asm {.reg.b16         hl, hu;         
 .reg.b32         h,r,fl,fu, C;   
  mov.b32         {hl, hu}, %1;   
  mov.b32         h, %1;          
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  mov.b32         C, 0x3fb8aa3b;  
  mul.f32         fl,fl,C;        
  mul.f32         fu,fu,C;        
  ex2.approx.f32      fl, fl;     
  ex2.approx.f32      fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         r, {hl, hu};    
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X1F791F79;
  mov.b32 ulp,0x94009400;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X25CF25CF;
  mov.b32 ulp,0x94009400;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0XC13BC13B;
  mov.b32 ulp,0x04000400;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0XC1EFC1EF;
  mov.b32 ulp,0x02000200;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32         %0, r;  
}
#line 3127
return val; 
#line 3128
} 
#endif
#line 3129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hexp2(const __half a) {int volatile ___ = 1;(void)a;
#line 3142
::exit(___);}
#if 0
#line 3129
{ 
#line 3130
__half val; 
#line 3131
__asm {.reg.b32         f, ULP;         
 .reg.b16         r;              
  mov.b16         r,%1;           
  cvt.f32.f16     f,r;            
  ex2.approx.f32      f,f;        
  mov.b32         ULP, 0x33800000;
  fma.rn.f32      f,f,ULP,f;      
  cvt.rn.f16.f32      r,f;        
  mov.b16         %0,r;           
}
return val; 
#line 3142
} 
#endif
#line 3143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2exp2(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3160
::exit(___);}
#if 0
#line 3143
{ 
#line 3144
__half2 val; 
#line 3145
__asm {.reg.b16         hl, hu;         
 .reg.b32         fl, fu, ULP;    
  mov.b32         {hl, hu}, %1;   
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  ex2.approx.f32      fl, fl;     
  ex2.approx.f32      fu, fu;     
  mov.b32         ULP, 0x33800000;
  fma.rn.f32      fl,fl,ULP,fl;   
  fma.rn.f32      fu,fu,ULP,fu;   
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         %0, {hl, hu};   
}
return val; 
#line 3160
} 
#endif
#line 3161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hexp10(const __half a) {int volatile ___ = 1;(void)a;
#line 3179
::exit(___);}
#if 0
#line 3161
{ 
#line 3162
__half val; 
#line 3163
__asm {.reg.b16         h,r;            
 .reg.b32         f, C;           
  mov.b16         h, %1;          
  cvt.f32.f16     f, h;           
  mov.b32         C, 0x40549A78;  
  mul.f32         f,f,C;          
  ex2.approx.f32      f, f;       
  cvt.rn.f16.f32      r, f;       
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x34DE;
  mov.b16 ulp,0x9800;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x9766;
  mov.b16 ulp,0x9000;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x9972;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0xA5C4;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0xBF0A;
  mov.b16 ulp,0x8100;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16         %0, r;          
}
#line 3178
return val; 
#line 3179
} 
#endif
#line 3180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2exp10(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3204
::exit(___);}
#if 0
#line 3180
{ 
#line 3181
__half2 val; 
#line 3182
__asm {.reg.b16         hl, hu;         
 .reg.b32         h,r,fl,fu, C;   
  mov.b32         {hl, hu}, %1;   
  mov.b32         h, %1;          
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  mov.b32         C, 0x40549A78;  
  mul.f32         fl,fl,C;        
  mul.f32         fu,fu,C;        
  ex2.approx.f32      fl, fl;     
  ex2.approx.f32      fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         r, {hl, hu};    
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x34DE34DE;
  mov.b32 ulp,0x98009800;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x97669766;
  mov.b32 ulp,0x90009000;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x99729972;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0xA5C4A5C4;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0xBF0ABF0A;
  mov.b32 ulp,0x81008100;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32         %0, r;  
}
#line 3203
return val; 
#line 3204
} 
#endif
#line 3205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hlog2(const __half a) {int volatile ___ = 1;(void)a;
#line 3218
::exit(___);}
#if 0
#line 3205
{ 
#line 3206
__half val; 
#line 3207
__asm {.reg.b16         h, r;           
 .reg.b32         f;              
  mov.b16         h, %1;          
  cvt.f32.f16     f, h;           
  lg2.approx.f32      f, f;       
  cvt.rn.f16.f32      r, f;       
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0xA2E2;
  mov.b16 ulp,0x8080;
  set.eq.f16.f16 p,r, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0xBF46;
  mov.b16 ulp,0x9400;
  set.eq.f16.f16 p,r, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16         %0, r;          
}
#line 3217
return val; 
#line 3218
} 
#endif
#line 3219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2log2(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3236
::exit(___);}
#if 0
#line 3219
{ 
#line 3220
__half2 val; 
#line 3221
__asm {.reg.b16         hl, hu;         
 .reg.b32         fl, fu, r, p;   
  mov.b32         {hl, hu}, %1;   
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  lg2.approx.f32      fl, fl;     
  lg2.approx.f32      fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         r, {hl, hu};    
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0xA2E2A2E2;
  mov.b32 ulp,0x80808080;
  set.eq.f16x2.f16x2 p,r, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0xBF46BF46;
  mov.b32 ulp,0x94009400;
  set.eq.f16x2.f16x2 p,r, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32         %0, r;          
}
#line 3235
return val; 
#line 3236
} 
#endif
#line 3237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hlog(const __half a) {int volatile ___ = 1;(void)a;
#line 3254
::exit(___);}
#if 0
#line 3237
{ 
#line 3238
__half val; 
#line 3239
__asm {.reg.b32         f, C;           
 .reg.b16         r,h;            
  mov.b16         h,%1;           
  cvt.f32.f16     f,h;            
  lg2.approx.f32      f,f;        
  mov.b32         C, 0x3f317218;  
  mul.f32         f,f,C;          
  cvt.rn.f16.f32      r,f;        
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X160D;
  mov.b16 ulp,0x9C00;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X3BFE;
  mov.b16 ulp,0x8010;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X3C0B;
  mov.b16 ulp,0x8080;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0X6051;
  mov.b16 ulp,0x1C00;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16         %0,r;           
}
#line 3253
return val; 
#line 3254
} 
#endif
#line 3255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2log(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3278
::exit(___);}
#if 0
#line 3255
{ 
#line 3256
__half2 val; 
#line 3257
__asm {.reg.b16         hl, hu;             
 .reg.b32         r, fl, fu, C, h;    
  mov.b32         {hl, hu}, %1;       
  mov.b32         h, %1;              
  cvt.f32.f16     fl, hl;             
  cvt.f32.f16     fu, hu;             
  lg2.approx.f32      fl, fl;         
  lg2.approx.f32      fu, fu;         
  mov.b32         C, 0x3f317218;      
  mul.f32         fl,fl,C;            
  mul.f32         fu,fu,C;            
  cvt.rn.f16.f32      hl, fl;         
  cvt.rn.f16.f32      hu, fu;         
  mov.b32         r, {hl, hu};        
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X160D160D;
  mov.b32 ulp,0x9C009C00;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X3BFE3BFE;
  mov.b32 ulp,0x80108010;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X3C0B3C0B;
  mov.b32 ulp,0x80808080;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0X60516051;
  mov.b32 ulp,0x1C001C00;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32         %0, r;              
}
#line 3277
return val; 
#line 3278
} 
#endif
#line 3279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hlog10(const __half a) {int volatile ___ = 1;(void)a;
#line 3296
::exit(___);}
#if 0
#line 3279
{ 
#line 3280
__half val; 
#line 3281
__asm {.reg.b16         h, r;           
 .reg.b32         f, C;           
  mov.b16         h, %1;          
  cvt.f32.f16     f, h;           
  lg2.approx.f32      f, f;       
  mov.b32         C, 0x3E9A209B;  
  mul.f32         f,f,C;          
  cvt.rn.f16.f32      r, f;       
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x338F;
  mov.b16 ulp,0x1000;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x33F8;
  mov.b16 ulp,0x9000;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x57E1;
  mov.b16 ulp,0x9800;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
{.reg.b16 spc, ulp, p;
  mov.b16 spc,0x719D;
  mov.b16 ulp,0x9C00;
  set.eq.f16.f16 p,h, spc;
  fma.rn.f16 r,p,ulp,r;
}
  mov.b16         %0, r;          
}
#line 3295
return val; 
#line 3296
} 
#endif
#line 3297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2log10(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3320
::exit(___);}
#if 0
#line 3297
{ 
#line 3298
__half2 val; 
#line 3299
__asm {.reg.b16         hl, hu;             
 .reg.b32         r, fl, fu, C, h;    
  mov.b32         {hl, hu}, %1;       
  mov.b32         h, %1;              
  cvt.f32.f16     fl, hl;             
  cvt.f32.f16     fu, hu;             
  lg2.approx.f32      fl, fl;         
  lg2.approx.f32      fu, fu;         
  mov.b32         C, 0x3E9A209B;      
  mul.f32         fl,fl,C;            
  mul.f32         fu,fu,C;            
  cvt.rn.f16.f32      hl, fl;         
  cvt.rn.f16.f32      hu, fu;         
  mov.b32         r, {hl, hu};        
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x338F338F;
  mov.b32 ulp,0x10001000;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x33F833F8;
  mov.b32 ulp,0x90009000;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x57E157E1;
  mov.b32 ulp,0x98009800;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
{.reg.b32 spc, ulp, p;
  mov.b32 spc,0x719D719D;
  mov.b32 ulp,0x9C009C00;
  set.eq.f16x2.f16x2 p,h, spc;
  fma.rn.f16x2 r,p,ulp,r;
}
  mov.b32         %0, r;              
}
#line 3319
return val; 
#line 3320
} 
#endif
#line 3323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2rcp(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3325
::exit(___);}
#if 0
#line 3323
{ 
#line 3324
do { __half2 val; __asm {.reg.b16         hl, hu;         
 .reg.b32         fl, fu;         
  mov.b32         {hl, hu}, %1;   
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  rcp.approx.f32   fl, fl;     
  rcp.approx.f32   fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         %0, {hl, hu};   
}
#line 3324
return val; } while (0); ; 
#line 3325
} 
#endif
#line 3326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hrcp(const __half a) {int volatile ___ = 1;(void)a;
#line 3328
::exit(___);}
#if 0
#line 3326
{ 
#line 3327
do { __half val; __asm {.reg.b32         f;        
 .reg.b16         r;        
  mov.b16         r,%1;     
  cvt.f32.f16     f,r;      
  rcp.approx.f32   f,f;  
  cvt.rn.f16.f32      r,f;  
  mov.b16         %0,r;     
}
#line 3327
return val; } while (0); ; 
#line 3328
} 
#endif
#line 3329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2rsqrt(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3331
::exit(___);}
#if 0
#line 3329
{ 
#line 3330
do { __half2 val; __asm {.reg.b16         hl, hu;         
 .reg.b32         fl, fu;         
  mov.b32         {hl, hu}, %1;   
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  rsqrt.approx.f32   fl, fl;     
  rsqrt.approx.f32   fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         %0, {hl, hu};   
}
#line 3330
return val; } while (0); ; 
#line 3331
} 
#endif
#line 3332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hrsqrt(const __half a) {int volatile ___ = 1;(void)a;
#line 3334
::exit(___);}
#if 0
#line 3332
{ 
#line 3333
do { __half val; __asm {.reg.b32         f;        
 .reg.b16         r;        
  mov.b16         r,%1;     
  cvt.f32.f16     f,r;      
  rsqrt.approx.f32   f,f;  
  cvt.rn.f16.f32      r,f;  
  mov.b16         %0,r;     
}
#line 3333
return val; } while (0); ; 
#line 3334
} 
#endif
#line 3335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 h2sqrt(const __half2 a) {int volatile ___ = 1;(void)a;
#line 3337
::exit(___);}
#if 0
#line 3335
{ 
#line 3336
do { __half2 val; __asm {.reg.b16         hl, hu;         
 .reg.b32         fl, fu;         
  mov.b32         {hl, hu}, %1;   
  cvt.f32.f16     fl, hl;         
  cvt.f32.f16     fu, hu;         
  sqrt.approx.f32   fl, fl;     
  sqrt.approx.f32   fu, fu;     
  cvt.rn.f16.f32      hl, fl;     
  cvt.rn.f16.f32      hu, fu;     
  mov.b32         %0, {hl, hu};   
}
#line 3336
return val; } while (0); ; 
#line 3337
} 
#endif
#line 3338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half hsqrt(const __half a) {int volatile ___ = 1;(void)a;
#line 3340
::exit(___);}
#if 0
#line 3338
{ 
#line 3339
do { __half val; __asm {.reg.b32         f;        
 .reg.b16         r;        
  mov.b16         r,%1;     
  cvt.f32.f16     f,r;      
  sqrt.approx.f32   f,f;  
  cvt.rn.f16.f32      r,f;  
  mov.b16         %0,r;     
}
#line 3339
return val; } while (0); ; 
#line 3340
} 
#endif
#line 3343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hisnan2(const __half2 a) 
#line 3344
{int volatile ___ = 1;(void)a;
#line 3349
::exit(___);}
#if 0
#line 3344
{ 
#line 3345
__half2 r; 
#line 3346
__asm {set.nan.f16x2.f16x2 %0,%1,%2;
}
return r; 
#line 3349
} 
#endif
#line 3350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline bool __hisnan(const __half a) 
#line 3351
{int volatile ___ = 1;(void)a;
#line 3358
::exit(___);}
#if 0
#line 3351
{ 
#line 3352
__half r; 
#line 3353
__asm {set.nan.f16.f16 %0,%1,%2;
}
if ((r.x) == 0) { 
#line 3356
return false; } else { 
#line 3357
return true; }  
#line 3358
} 
#endif
#line 3359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half2 __hneg2(const __half2 a) 
#line 3360
{int volatile ___ = 1;(void)a;
#line 3363
::exit(___);}
#if 0
#line 3360
{ 
#line 3361
__half2 zero = __float2half2_rn((0.0)); 
#line 3362
return __hsub2(zero, a); 
#line 3363
} 
#endif
#line 3364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_fp16.h"
static __inline __half __hneg(const __half a) 
#line 3365
{int volatile ___ = 1;(void)a;
#line 3369
::exit(___);}
#if 0
#line 3365
{ 
#line 3366
__half zero; 
#line 3367
zero = __float2half((0.0)); 
#line 3368
return __hsub(zero, a); 
#line 3369
} 
#endif
#line 78 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cublas_api.h"
extern "C" {
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cublas_api.h"
typedef 
#line 82
enum { 
#line 83
CUBLAS_STATUS_SUCCESS, 
#line 84
CUBLAS_STATUS_NOT_INITIALIZED, 
#line 85
CUBLAS_STATUS_ALLOC_FAILED = 3, 
#line 86
CUBLAS_STATUS_INVALID_VALUE = 7, 
#line 87
CUBLAS_STATUS_ARCH_MISMATCH, 
#line 88
CUBLAS_STATUS_MAPPING_ERROR = 11, 
#line 89
CUBLAS_STATUS_EXECUTION_FAILED = 13, 
#line 90
CUBLAS_STATUS_INTERNAL_ERROR, 
#line 91
CUBLAS_STATUS_NOT_SUPPORTED, 
#line 92
CUBLAS_STATUS_LICENSE_ERROR
#line 93
} cublasStatus_t; 
#line 99
typedef 
#line 96
enum { 
#line 97
CUBLAS_FILL_MODE_LOWER, 
#line 98
CUBLAS_FILL_MODE_UPPER
#line 99
} cublasFillMode_t; 
#line 104
typedef 
#line 101
enum { 
#line 102
CUBLAS_DIAG_NON_UNIT, 
#line 103
CUBLAS_DIAG_UNIT
#line 104
} cublasDiagType_t; 
#line 109
typedef 
#line 106
enum { 
#line 107
CUBLAS_SIDE_LEFT, 
#line 108
CUBLAS_SIDE_RIGHT
#line 109
} cublasSideMode_t; 
#line 116
typedef 
#line 112
enum { 
#line 113
CUBLAS_OP_N, 
#line 114
CUBLAS_OP_T, 
#line 115
CUBLAS_OP_C
#line 116
} cublasOperation_t; 
#line 122
typedef 
#line 119
enum { 
#line 120
CUBLAS_POINTER_MODE_HOST, 
#line 121
CUBLAS_POINTER_MODE_DEVICE
#line 122
} cublasPointerMode_t; 
#line 127
typedef 
#line 124
enum { 
#line 125
CUBLAS_ATOMICS_NOT_ALLOWED, 
#line 126
CUBLAS_ATOMICS_ALLOWED
#line 127
} cublasAtomicsMode_t; 
#line 140
typedef 
#line 130
enum { 
#line 131
CUBLAS_GEMM_DFALT = (-1), 
#line 132
CUBLAS_GEMM_ALGO0 = 0, 
#line 133
CUBLAS_GEMM_ALGO1, 
#line 134
CUBLAS_GEMM_ALGO2, 
#line 135
CUBLAS_GEMM_ALGO3, 
#line 136
CUBLAS_GEMM_ALGO4, 
#line 137
CUBLAS_GEMM_ALGO5, 
#line 138
CUBLAS_GEMM_ALGO6, 
#line 139
CUBLAS_GEMM_ALGO7
#line 140
} cublasGemmAlgo_t; 
#line 143
typedef cudaDataType cublasDataType_t; 
#line 146
struct cublasContext; 
#line 147
typedef cublasContext *cublasHandle_t; 
#line 149
cublasStatus_t __stdcall cublasCreate_v2(cublasHandle_t * handle); 
#line 150
cublasStatus_t __stdcall cublasDestroy_v2(cublasHandle_t handle); 
#line 152
cublasStatus_t __stdcall cublasGetVersion_v2(cublasHandle_t handle, int * version); 
#line 153
cublasStatus_t __stdcall cublasGetProperty(libraryPropertyType type, int * value); 
#line 155
cublasStatus_t __stdcall cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId); 
#line 156
cublasStatus_t __stdcall cublasGetStream_v2(cublasHandle_t handle, cudaStream_t * streamId); 
#line 158
cublasStatus_t __stdcall cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t * mode); 
#line 159
cublasStatus_t __stdcall cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode); 
#line 161
cublasStatus_t __stdcall cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t * mode); 
#line 162
cublasStatus_t __stdcall cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode); 
#line 187
cublasStatus_t __stdcall cublasSetVector(int n, int elemSize, const void * x, int incx, void * devicePtr, int incy); 
#line 213
cublasStatus_t __stdcall cublasGetVector(int n, int elemSize, const void * x, int incx, void * y, int incy); 
#line 237
cublasStatus_t __stdcall cublasSetMatrix(int rows, int cols, int elemSize, const void * A, int lda, void * B, int ldb); 
#line 261
cublasStatus_t __stdcall cublasGetMatrix(int rows, int cols, int elemSize, const void * A, int lda, void * B, int ldb); 
#line 281
cublasStatus_t __stdcall cublasSetVectorAsync(int n, int elemSize, const void * hostPtr, int incx, void * devicePtr, int incy, cudaStream_t stream); 
#line 301
cublasStatus_t __stdcall cublasGetVectorAsync(int n, int elemSize, const void * devicePtr, int incx, void * hostPtr, int incy, cudaStream_t stream); 
#line 323
cublasStatus_t __stdcall cublasSetMatrixAsync(int rows, int cols, int elemSize, const void * A, int lda, void * B, int ldb, cudaStream_t stream); 
#line 343
cublasStatus_t __stdcall cublasGetMatrixAsync(int rows, int cols, int elemSize, const void * A, int lda, void * B, int ldb, cudaStream_t stream); 
#line 348
void __stdcall cublasXerbla(const char * srName, int info); 
#line 350
cublasStatus_t __stdcall cublasNrm2Ex(cublasHandle_t handle, int n, const void * x, cudaDataType xType, int incx, void * result, cudaDataType resultType, cudaDataType executionType); 
#line 358
cublasStatus_t __stdcall cublasSnrm2_v2(cublasHandle_t handle, int n, const float * x, int incx, float * result); 
#line 364
cublasStatus_t __stdcall cublasDnrm2_v2(cublasHandle_t handle, int n, const double * x, int incx, double * result); 
#line 370
cublasStatus_t __stdcall cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, float * result); 
#line 376
cublasStatus_t __stdcall cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, double * result); 
#line 382
cublasStatus_t __stdcall cublasDotEx(cublasHandle_t handle, int n, const void * x, cudaDataType xType, int incx, const void * y, cudaDataType yType, int incy, void * result, cudaDataType resultType, cudaDataType executionType); 
#line 394
cublasStatus_t __stdcall cublasDotcEx(cublasHandle_t handle, int n, const void * x, cudaDataType xType, int incx, const void * y, cudaDataType yType, int incy, void * result, cudaDataType resultType, cudaDataType executionType); 
#line 406
cublasStatus_t __stdcall cublasSdot_v2(cublasHandle_t handle, int n, const float * x, int incx, const float * y, int incy, float * result); 
#line 414
cublasStatus_t __stdcall cublasDdot_v2(cublasHandle_t handle, int n, const double * x, int incx, const double * y, int incy, double * result); 
#line 422
cublasStatus_t __stdcall cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * result); 
#line 430
cublasStatus_t __stdcall cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * result); 
#line 438
cublasStatus_t __stdcall cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * result); 
#line 446
cublasStatus_t __stdcall cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * result); 
#line 454
cublasStatus_t __stdcall cublasScalEx(cublasHandle_t handle, int n, const void * alpha, cudaDataType alphaType, void * x, cudaDataType xType, int incx, cudaDataType executionType); 
#line 462
cublasStatus_t __stdcall cublasSscal_v2(cublasHandle_t handle, int n, const float * alpha, float * x, int incx); 
#line 468
cublasStatus_t __stdcall cublasDscal_v2(cublasHandle_t handle, int n, const double * alpha, double * x, int incx); 
#line 474
cublasStatus_t __stdcall cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex * alpha, cuComplex * x, int incx); 
#line 480
cublasStatus_t __stdcall cublasCsscal_v2(cublasHandle_t handle, int n, const float * alpha, cuComplex * x, int incx); 
#line 486
cublasStatus_t __stdcall cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex * alpha, cuDoubleComplex * x, int incx); 
#line 492
cublasStatus_t __stdcall cublasZdscal_v2(cublasHandle_t handle, int n, const double * alpha, cuDoubleComplex * x, int incx); 
#line 498
cublasStatus_t __stdcall cublasAxpyEx(cublasHandle_t handle, int n, const void * alpha, cudaDataType alphaType, const void * x, cudaDataType xType, int incx, void * y, cudaDataType yType, int incy, cudaDataType executiontype); 
#line 510
cublasStatus_t __stdcall cublasSaxpy_v2(cublasHandle_t handle, int n, const float * alpha, const float * x, int incx, float * y, int incy); 
#line 518
cublasStatus_t __stdcall cublasDaxpy_v2(cublasHandle_t handle, int n, const double * alpha, const double * x, int incx, double * y, int incy); 
#line 526
cublasStatus_t __stdcall cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex * alpha, const cuComplex * x, int incx, cuComplex * y, int incy); 
#line 534
cublasStatus_t __stdcall cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy); 
#line 542
cublasStatus_t __stdcall cublasScopy_v2(cublasHandle_t handle, int n, const float * x, int incx, float * y, int incy); 
#line 549
cublasStatus_t __stdcall cublasDcopy_v2(cublasHandle_t handle, int n, const double * x, int incx, double * y, int incy); 
#line 556
cublasStatus_t __stdcall cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, cuComplex * y, int incy); 
#line 563
cublasStatus_t __stdcall cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy); 
#line 570
cublasStatus_t __stdcall cublasSswap_v2(cublasHandle_t handle, int n, float * x, int incx, float * y, int incy); 
#line 577
cublasStatus_t __stdcall cublasDswap_v2(cublasHandle_t handle, int n, double * x, int incx, double * y, int incy); 
#line 584
cublasStatus_t __stdcall cublasCswap_v2(cublasHandle_t handle, int n, cuComplex * x, int incx, cuComplex * y, int incy); 
#line 591
cublasStatus_t __stdcall cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy); 
#line 598
cublasStatus_t __stdcall cublasIsamax_v2(cublasHandle_t handle, int n, const float * x, int incx, int * result); 
#line 604
cublasStatus_t __stdcall cublasIdamax_v2(cublasHandle_t handle, int n, const double * x, int incx, int * result); 
#line 610
cublasStatus_t __stdcall cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, int * result); 
#line 616
cublasStatus_t __stdcall cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, int * result); 
#line 622
cublasStatus_t __stdcall cublasIsamin_v2(cublasHandle_t handle, int n, const float * x, int incx, int * result); 
#line 628
cublasStatus_t __stdcall cublasIdamin_v2(cublasHandle_t handle, int n, const double * x, int incx, int * result); 
#line 634
cublasStatus_t __stdcall cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, int * result); 
#line 640
cublasStatus_t __stdcall cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, int * result); 
#line 646
cublasStatus_t __stdcall cublasSasum_v2(cublasHandle_t handle, int n, const float * x, int incx, float * result); 
#line 652
cublasStatus_t __stdcall cublasDasum_v2(cublasHandle_t handle, int n, const double * x, int incx, double * result); 
#line 658
cublasStatus_t __stdcall cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex * x, int incx, float * result); 
#line 664
cublasStatus_t __stdcall cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex * x, int incx, double * result); 
#line 670
cublasStatus_t __stdcall cublasSrot_v2(cublasHandle_t handle, int n, float * x, int incx, float * y, int incy, const float * c, const float * s); 
#line 679
cublasStatus_t __stdcall cublasDrot_v2(cublasHandle_t handle, int n, double * x, int incx, double * y, int incy, const double * c, const double * s); 
#line 688
cublasStatus_t __stdcall cublasCrot_v2(cublasHandle_t handle, int n, cuComplex * x, int incx, cuComplex * y, int incy, const float * c, const cuComplex * s); 
#line 697
cublasStatus_t __stdcall cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex * x, int incx, cuComplex * y, int incy, const float * c, const float * s); 
#line 706
cublasStatus_t __stdcall cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy, const double * c, const cuDoubleComplex * s); 
#line 715
cublasStatus_t __stdcall cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy, const double * c, const double * s); 
#line 724
cublasStatus_t __stdcall cublasSrotg_v2(cublasHandle_t handle, float * a, float * b, float * c, float * s); 
#line 730
cublasStatus_t __stdcall cublasDrotg_v2(cublasHandle_t handle, double * a, double * b, double * c, double * s); 
#line 736
cublasStatus_t __stdcall cublasCrotg_v2(cublasHandle_t handle, cuComplex * a, cuComplex * b, float * c, cuComplex * s); 
#line 742
cublasStatus_t __stdcall cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex * a, cuDoubleComplex * b, double * c, cuDoubleComplex * s); 
#line 748
cublasStatus_t __stdcall cublasSrotm_v2(cublasHandle_t handle, int n, float * x, int incx, float * y, int incy, const float * param); 
#line 756
cublasStatus_t __stdcall cublasDrotm_v2(cublasHandle_t handle, int n, double * x, int incx, double * y, int incy, const double * param); 
#line 764
cublasStatus_t __stdcall cublasSrotmg_v2(cublasHandle_t handle, float * d1, float * d2, float * x1, const float * y1, float * param); 
#line 771
cublasStatus_t __stdcall cublasDrotmg_v2(cublasHandle_t handle, double * d1, double * d2, double * x1, const double * y1, double * param); 
#line 781
cublasStatus_t __stdcall cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy); 
#line 794
cublasStatus_t __stdcall cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double * alpha, const double * A, int lda, const double * x, int incx, const double * beta, double * y, int incy); 
#line 807
cublasStatus_t __stdcall cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 820
cublasStatus_t __stdcall cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 833
cublasStatus_t __stdcall cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy); 
#line 848
cublasStatus_t __stdcall cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double * alpha, const double * A, int lda, const double * x, int incx, const double * beta, double * y, int incy); 
#line 863
cublasStatus_t __stdcall cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 878
cublasStatus_t __stdcall cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 894
cublasStatus_t __stdcall cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float * A, int lda, float * x, int incx); 
#line 904
cublasStatus_t __stdcall cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double * A, int lda, double * x, int incx); 
#line 914
cublasStatus_t __stdcall cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex * A, int lda, cuComplex * x, int incx); 
#line 924
cublasStatus_t __stdcall cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx); 
#line 935
cublasStatus_t __stdcall cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float * A, int lda, float * x, int incx); 
#line 946
cublasStatus_t __stdcall cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double * A, int lda, double * x, int incx); 
#line 957
cublasStatus_t __stdcall cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex * A, int lda, cuComplex * x, int incx); 
#line 968
cublasStatus_t __stdcall cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx); 
#line 980
cublasStatus_t __stdcall cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float * AP, float * x, int incx); 
#line 989
cublasStatus_t __stdcall cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double * AP, double * x, int incx); 
#line 998
cublasStatus_t __stdcall cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex * AP, cuComplex * x, int incx); 
#line 1007
cublasStatus_t __stdcall cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex * AP, cuDoubleComplex * x, int incx); 
#line 1017
cublasStatus_t __stdcall cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float * A, int lda, float * x, int incx); 
#line 1027
cublasStatus_t __stdcall cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double * A, int lda, double * x, int incx); 
#line 1037
cublasStatus_t __stdcall cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex * A, int lda, cuComplex * x, int incx); 
#line 1047
cublasStatus_t __stdcall cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx); 
#line 1058
cublasStatus_t __stdcall cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float * AP, float * x, int incx); 
#line 1067
cublasStatus_t __stdcall cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double * AP, double * x, int incx); 
#line 1076
cublasStatus_t __stdcall cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex * AP, cuComplex * x, int incx); 
#line 1085
cublasStatus_t __stdcall cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex * AP, cuDoubleComplex * x, int incx); 
#line 1094
cublasStatus_t __stdcall cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float * A, int lda, float * x, int incx); 
#line 1105
cublasStatus_t __stdcall cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double * A, int lda, double * x, int incx); 
#line 1116
cublasStatus_t __stdcall cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex * A, int lda, cuComplex * x, int incx); 
#line 1127
cublasStatus_t __stdcall cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx); 
#line 1139
cublasStatus_t __stdcall cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy); 
#line 1151
cublasStatus_t __stdcall cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * A, int lda, const double * x, int incx, const double * beta, double * y, int incy); 
#line 1163
cublasStatus_t __stdcall cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 1175
cublasStatus_t __stdcall cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 1187
cublasStatus_t __stdcall cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 1199
cublasStatus_t __stdcall cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 1212
cublasStatus_t __stdcall cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float * alpha, const float * A, int lda, const float * x, int incx, const float * beta, float * y, int incy); 
#line 1225
cublasStatus_t __stdcall cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double * alpha, const double * A, int lda, const double * x, int incx, const double * beta, double * y, int incy); 
#line 1238
cublasStatus_t __stdcall cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 1251
cublasStatus_t __stdcall cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 1265
cublasStatus_t __stdcall cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * AP, const float * x, int incx, const float * beta, float * y, int incy); 
#line 1276
cublasStatus_t __stdcall cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * AP, const double * x, int incx, const double * beta, double * y, int incy); 
#line 1287
cublasStatus_t __stdcall cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * AP, const cuComplex * x, int incx, const cuComplex * beta, cuComplex * y, int incy); 
#line 1298
cublasStatus_t __stdcall cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * AP, const cuDoubleComplex * x, int incx, const cuDoubleComplex * beta, cuDoubleComplex * y, int incy); 
#line 1310
cublasStatus_t __stdcall cublasSger_v2(cublasHandle_t handle, int m, int n, const float * alpha, const float * x, int incx, const float * y, int incy, float * A, int lda); 
#line 1321
cublasStatus_t __stdcall cublasDger_v2(cublasHandle_t handle, int m, int n, const double * alpha, const double * x, int incx, const double * y, int incy, double * A, int lda); 
#line 1332
cublasStatus_t __stdcall cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex * alpha, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * A, int lda); 
#line 1343
cublasStatus_t __stdcall cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex * alpha, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * A, int lda); 
#line 1354
cublasStatus_t __stdcall cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * A, int lda); 
#line 1365
cublasStatus_t __stdcall cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * A, int lda); 
#line 1377
cublasStatus_t __stdcall cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * x, int incx, float * A, int lda); 
#line 1386
cublasStatus_t __stdcall cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * x, int incx, double * A, int lda); 
#line 1395
cublasStatus_t __stdcall cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * x, int incx, cuComplex * A, int lda); 
#line 1404
cublasStatus_t __stdcall cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * A, int lda); 
#line 1413
cublasStatus_t __stdcall cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const cuComplex * x, int incx, cuComplex * A, int lda); 
#line 1422
cublasStatus_t __stdcall cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * A, int lda); 
#line 1432
cublasStatus_t __stdcall cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * x, int incx, float * AP); 
#line 1440
cublasStatus_t __stdcall cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * x, int incx, double * AP); 
#line 1448
cublasStatus_t __stdcall cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const cuComplex * x, int incx, cuComplex * AP); 
#line 1456
cublasStatus_t __stdcall cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const cuDoubleComplex * x, int incx, cuDoubleComplex * AP); 
#line 1465
cublasStatus_t __stdcall cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * x, int incx, const float * y, int incy, float * A, int lda); 
#line 1476
cublasStatus_t __stdcall cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * x, int incx, const double * y, int incy, double * A, int lda); 
#line 1487
cublasStatus_t __stdcall cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * A, int lda); 
#line 1497
cublasStatus_t __stdcall cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * A, int lda); 
#line 1509
cublasStatus_t __stdcall cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * A, int lda); 
#line 1519
cublasStatus_t __stdcall cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * A, int lda); 
#line 1531
cublasStatus_t __stdcall cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * alpha, const float * x, int incx, const float * y, int incy, float * AP); 
#line 1541
cublasStatus_t __stdcall cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * alpha, const double * x, int incx, const double * y, int incy, double * AP); 
#line 1552
cublasStatus_t __stdcall cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * alpha, const cuComplex * x, int incx, const cuComplex * y, int incy, cuComplex * AP); 
#line 1562
cublasStatus_t __stdcall cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * AP); 
#line 1575
cublasStatus_t __stdcall cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float * alpha, const float * A, int lda, const float * B, int ldb, const float * beta, float * C, int ldc); 
#line 1590
cublasStatus_t __stdcall cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double * alpha, const double * A, int lda, const double * B, int ldb, const double * beta, double * C, int ldc); 
#line 1605
cublasStatus_t __stdcall cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 1620
cublasStatus_t __stdcall cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 1634
cublasStatus_t __stdcall cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const void * A, cudaDataType Atype, int lda, const void * B, cudaDataType Btype, int ldb, const cuComplex * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1650
cublasStatus_t __stdcall cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 1665
cublasStatus_t __stdcall cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 1680
cublasStatus_t __stdcall cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half * alpha, const __half * A, int lda, const __half * B, int ldb, const __half * beta, __half * C, int ldc); 
#line 1695
cublasStatus_t __stdcall cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float * alpha, const void * A, cudaDataType Atype, int lda, const void * B, cudaDataType Btype, int ldb, const float * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1713
cublasStatus_t __stdcall cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void * alpha, const void * A, cudaDataType Atype, int lda, const void * B, cudaDataType Btype, int ldb, const void * beta, void * C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo); 
#line 1734
cublasStatus_t __stdcall cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const void * A, cudaDataType Atype, int lda, const void * B, cudaDataType Btype, int ldb, const cuComplex * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1749
cublasStatus_t __stdcall cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char * A, int A_bias, int lda, const unsigned char * B, int B_bias, int ldb, unsigned char * C, int C_bias, int ldc, int C_mult, int C_shift); 
#line 1758
cublasStatus_t __stdcall cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const float * A, int lda, const float * beta, float * C, int ldc); 
#line 1770
cublasStatus_t __stdcall cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double * alpha, const double * A, int lda, const double * beta, double * C, int ldc); 
#line 1782
cublasStatus_t __stdcall cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * beta, cuComplex * C, int ldc); 
#line 1794
cublasStatus_t __stdcall cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 1806
cublasStatus_t __stdcall cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const void * A, cudaDataType Atype, int lda, const cuComplex * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1821
cublasStatus_t __stdcall cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const void * A, cudaDataType Atype, int lda, const cuComplex * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1836
cublasStatus_t __stdcall cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const cuComplex * A, int lda, const float * beta, cuComplex * C, int ldc); 
#line 1848
cublasStatus_t __stdcall cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double * alpha, const cuDoubleComplex * A, int lda, const double * beta, cuDoubleComplex * C, int ldc); 
#line 1861
cublasStatus_t __stdcall cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const void * A, cudaDataType Atype, int lda, const float * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1876
cublasStatus_t __stdcall cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const void * A, cudaDataType Atype, int lda, const float * beta, void * C, cudaDataType Ctype, int ldc); 
#line 1892
cublasStatus_t __stdcall cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const float * A, int lda, const float * B, int ldb, const float * beta, float * C, int ldc); 
#line 1906
cublasStatus_t __stdcall cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double * alpha, const double * A, int lda, const double * B, int ldb, const double * beta, double * C, int ldc); 
#line 1920
cublasStatus_t __stdcall cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 1934
cublasStatus_t __stdcall cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 1948
cublasStatus_t __stdcall cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const float * beta, cuComplex * C, int ldc); 
#line 1962
cublasStatus_t __stdcall cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const double * beta, cuDoubleComplex * C, int ldc); 
#line 1976
cublasStatus_t __stdcall cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float * alpha, const float * A, int lda, const float * B, int ldb, const float * beta, float * C, int ldc); 
#line 1990
cublasStatus_t __stdcall cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double * alpha, const double * A, int lda, const double * B, int ldb, const double * beta, double * C, int ldc); 
#line 2004
cublasStatus_t __stdcall cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 2018
cublasStatus_t __stdcall cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 2032
cublasStatus_t __stdcall cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const float * beta, cuComplex * C, int ldc); 
#line 2046
cublasStatus_t __stdcall cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const double * beta, cuDoubleComplex * C, int ldc); 
#line 2060
cublasStatus_t __stdcall cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float * alpha, const float * A, int lda, const float * B, int ldb, const float * beta, float * C, int ldc); 
#line 2074
cublasStatus_t __stdcall cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double * alpha, const double * A, int lda, const double * B, int ldb, const double * beta, double * C, int ldc); 
#line 2088
cublasStatus_t __stdcall cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 2102
cublasStatus_t __stdcall cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 2117
cublasStatus_t __stdcall cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, const cuComplex * beta, cuComplex * C, int ldc); 
#line 2131
cublasStatus_t __stdcall cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc); 
#line 2146
cublasStatus_t __stdcall cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float * alpha, const float * A, int lda, float * B, int ldb); 
#line 2160
cublasStatus_t __stdcall cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double * alpha, const double * A, int lda, double * B, int ldb); 
#line 2173
cublasStatus_t __stdcall cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, cuComplex * B, int ldb); 
#line 2186
cublasStatus_t __stdcall cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, cuDoubleComplex * B, int ldb); 
#line 2200
cublasStatus_t __stdcall cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float * alpha, const float * A, int lda, const float * B, int ldb, float * C, int ldc); 
#line 2215
cublasStatus_t __stdcall cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double * alpha, const double * A, int lda, const double * B, int ldb, double * C, int ldc); 
#line 2230
cublasStatus_t __stdcall cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * B, int ldb, cuComplex * C, int ldc); 
#line 2245
cublasStatus_t __stdcall cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb, cuDoubleComplex * C, int ldc); 
#line 2259
cublasStatus_t __stdcall cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float * alpha, const float * Aarray[], int lda, const float * Barray[], int ldb, const float * beta, float * Carray[], int ldc, int batchCount); 
#line 2275
cublasStatus_t __stdcall cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double * alpha, const double * Aarray[], int lda, const double * Barray[], int ldb, const double * beta, double * Carray[], int ldc, int batchCount); 
#line 2291
cublasStatus_t __stdcall cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * Aarray[], int lda, const cuComplex * Barray[], int ldb, const cuComplex * beta, cuComplex * Carray[], int ldc, int batchCount); 
#line 2307
cublasStatus_t __stdcall cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * Aarray[], int lda, const cuComplex * Barray[], int ldb, const cuComplex * beta, cuComplex * Carray[], int ldc, int batchCount); 
#line 2323
cublasStatus_t __stdcall cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * Aarray[], int lda, const cuDoubleComplex * Barray[], int ldb, const cuDoubleComplex * beta, cuDoubleComplex * Carray[], int ldc, int batchCount); 
#line 2339
cublasStatus_t __stdcall cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float * alpha, const float * A, int lda, __int64 strideA, const float * B, int ldb, __int64 strideB, const float * beta, float * C, int ldc, __int64 strideC, int batchCount); 
#line 2358
cublasStatus_t __stdcall cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double * alpha, const double * A, int lda, __int64 strideA, const double * B, int ldb, __int64 strideB, const double * beta, double * C, int ldc, __int64 strideC, int batchCount); 
#line 2377
cublasStatus_t __stdcall cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, __int64 strideA, const cuComplex * B, int ldb, __int64 strideB, const cuComplex * beta, cuComplex * C, int ldc, __int64 strideC, int batchCount); 
#line 2396
cublasStatus_t __stdcall cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex * alpha, const cuComplex * A, int lda, __int64 strideA, const cuComplex * B, int ldb, __int64 strideB, const cuComplex * beta, cuComplex * C, int ldc, __int64 strideC, int batchCount); 
#line 2416
cublasStatus_t __stdcall cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, __int64 strideA, const cuDoubleComplex * B, int ldb, __int64 strideB, const cuDoubleComplex * beta, cuDoubleComplex * C, int ldc, __int64 strideC, int batchCount); 
#line 2434
cublasStatus_t __stdcall cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half * alpha, const __half * A, int lda, __int64 strideA, const __half * B, int ldb, __int64 strideB, const __half * beta, __half * C, int ldc, __int64 strideC, int batchCount); 
#line 2455
cublasStatus_t __stdcall cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float * alpha, const float * A, int lda, const float * beta, const float * B, int ldb, float * C, int ldc); 
#line 2469
cublasStatus_t __stdcall cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double * alpha, const double * A, int lda, const double * beta, const double * B, int ldb, double * C, int ldc); 
#line 2483
cublasStatus_t __stdcall cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex * alpha, const cuComplex * A, int lda, const cuComplex * beta, const cuComplex * B, int ldb, cuComplex * C, int ldc); 
#line 2497
cublasStatus_t __stdcall cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A, int lda, const cuDoubleComplex * beta, const cuDoubleComplex * B, int ldb, cuDoubleComplex * C, int ldc); 
#line 2512
cublasStatus_t __stdcall cublasSgetrfBatched(cublasHandle_t handle, int n, float * A[], int lda, int * P, int * info, int batchSize); 
#line 2520
cublasStatus_t __stdcall cublasDgetrfBatched(cublasHandle_t handle, int n, double * A[], int lda, int * P, int * info, int batchSize); 
#line 2528
cublasStatus_t __stdcall cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex * A[], int lda, int * P, int * info, int batchSize); 
#line 2536
cublasStatus_t __stdcall cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex * A[], int lda, int * P, int * info, int batchSize); 
#line 2545
cublasStatus_t __stdcall cublasSgetriBatched(cublasHandle_t handle, int n, const float * A[], int lda, const int * P, float * C[], int ldc, int * info, int batchSize); 
#line 2555
cublasStatus_t __stdcall cublasDgetriBatched(cublasHandle_t handle, int n, const double * A[], int lda, const int * P, double * C[], int ldc, int * info, int batchSize); 
#line 2565
cublasStatus_t __stdcall cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex * A[], int lda, const int * P, cuComplex * C[], int ldc, int * info, int batchSize); 
#line 2575
cublasStatus_t __stdcall cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex * A[], int lda, const int * P, cuDoubleComplex * C[], int ldc, int * info, int batchSize); 
#line 2587
cublasStatus_t __stdcall cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float * Aarray[], int lda, const int * devIpiv, float * Barray[], int ldb, int * info, int batchSize); 
#line 2599
cublasStatus_t __stdcall cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double * Aarray[], int lda, const int * devIpiv, double * Barray[], int ldb, int * info, int batchSize); 
#line 2611
cublasStatus_t __stdcall cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex * Aarray[], int lda, const int * devIpiv, cuComplex * Barray[], int ldb, int * info, int batchSize); 
#line 2624
cublasStatus_t __stdcall cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex * Aarray[], int lda, const int * devIpiv, cuDoubleComplex * Barray[], int ldb, int * info, int batchSize); 
#line 2639
cublasStatus_t __stdcall cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float * alpha, const float * A[], int lda, float * B[], int ldb, int batchCount); 
#line 2653
cublasStatus_t __stdcall cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double * alpha, const double * A[], int lda, double * B[], int ldb, int batchCount); 
#line 2667
cublasStatus_t __stdcall cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex * alpha, const cuComplex * A[], int lda, cuComplex * B[], int ldb, int batchCount); 
#line 2681
cublasStatus_t __stdcall cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex * alpha, const cuDoubleComplex * A[], int lda, cuDoubleComplex * B[], int ldb, int batchCount); 
#line 2696
cublasStatus_t __stdcall cublasSmatinvBatched(cublasHandle_t handle, int n, const float * A[], int lda, float * Ainv[], int lda_inv, int * info, int batchSize); 
#line 2705
cublasStatus_t __stdcall cublasDmatinvBatched(cublasHandle_t handle, int n, const double * A[], int lda, double * Ainv[], int lda_inv, int * info, int batchSize); 
#line 2714
cublasStatus_t __stdcall cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex * A[], int lda, cuComplex * Ainv[], int lda_inv, int * info, int batchSize); 
#line 2723
cublasStatus_t __stdcall cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex * A[], int lda, cuDoubleComplex * Ainv[], int lda_inv, int * info, int batchSize); 
#line 2733
cublasStatus_t __stdcall cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float * Aarray[], int lda, float * TauArray[], int * info, int batchSize); 
#line 2742
cublasStatus_t __stdcall cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double * Aarray[], int lda, double * TauArray[], int * info, int batchSize); 
#line 2751
cublasStatus_t __stdcall cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex * Aarray[], int lda, cuComplex * TauArray[], int * info, int batchSize); 
#line 2760
cublasStatus_t __stdcall cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex * Aarray[], int lda, cuDoubleComplex * TauArray[], int * info, int batchSize); 
#line 2769
cublasStatus_t __stdcall cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float * Aarray[], int lda, float * Carray[], int ldc, int * info, int * devInfoArray, int batchSize); 
#line 2782
cublasStatus_t __stdcall cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double * Aarray[], int lda, double * Carray[], int ldc, int * info, int * devInfoArray, int batchSize); 
#line 2795
cublasStatus_t __stdcall cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex * Aarray[], int lda, cuComplex * Carray[], int ldc, int * info, int * devInfoArray, int batchSize); 
#line 2808
cublasStatus_t __stdcall cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex * Aarray[], int lda, cuDoubleComplex * Carray[], int ldc, int * info, int * devInfoArray, int batchSize); 
#line 2821
cublasStatus_t __stdcall cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float * A, int lda, const float * x, int incx, float * C, int ldc); 
#line 2832
cublasStatus_t __stdcall cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double * A, int lda, const double * x, int incx, double * C, int ldc); 
#line 2843
cublasStatus_t __stdcall cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex * A, int lda, const cuComplex * x, int incx, cuComplex * C, int ldc); 
#line 2854
cublasStatus_t __stdcall cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx, cuDoubleComplex * C, int ldc); 
#line 2866
cublasStatus_t __stdcall cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * AP, float * A, int lda); 
#line 2873
cublasStatus_t __stdcall cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * AP, double * A, int lda); 
#line 2880
cublasStatus_t __stdcall cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * AP, cuComplex * A, int lda); 
#line 2887
cublasStatus_t __stdcall cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * AP, cuDoubleComplex * A, int lda); 
#line 2894
cublasStatus_t __stdcall cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float * A, int lda, float * AP); 
#line 2901
cublasStatus_t __stdcall cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double * A, int lda, double * AP); 
#line 2908
cublasStatus_t __stdcall cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, cuComplex * AP); 
#line 2915
cublasStatus_t __stdcall cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex * A, int lda, cuDoubleComplex * AP); 
#line 2923
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_stdio_config.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 65 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_stdio_config.h"
#pragma detect_mismatch("_CRT_STDIO_ISO_WIDE_SPECIFIERS", "0")
#line 72 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_stdio_config.h"
__declspec(noinline) __inline unsigned __int64 *__cdecl __local_stdio_printf_options() 
#line 73
{ 
#line 74
static unsigned __int64 _OptionsStorage; 
#line 75
return &_OptionsStorage; 
#line 76
} 
#line 81
__declspec(noinline) __inline unsigned __int64 *__cdecl __local_stdio_scanf_options() 
#line 82
{ 
#line 83
static unsigned __int64 _OptionsStorage; 
#line 84
return &_OptionsStorage; 
#line 85
} 
#line 105
}__pragma( pack ( pop )) 
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
typedef 
#line 25
struct _iobuf { 
#line 27
void *_Placeholder; 
#line 28
} FILE; 
#line 31 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) FILE *__cdecl __acrt_iob_func(unsigned); 
#line 47
__declspec(dllimport) wint_t __cdecl fgetwc(FILE * _Stream); 
#line 52
__declspec(dllimport) wint_t __cdecl _fgetwchar(); 
#line 55
__declspec(dllimport) wint_t __cdecl fputwc(__wchar_t _Character, FILE * _Stream); 
#line 60
__declspec(dllimport) wint_t __cdecl _fputwchar(__wchar_t _Character); 
#line 65
__declspec(dllimport) wint_t __cdecl getwc(FILE * _Stream); 
#line 70
__declspec(dllimport) wint_t __cdecl getwchar(); 
#line 75
__declspec(dllimport) __wchar_t *__cdecl fgetws(__wchar_t * _Buffer, int _BufferCount, FILE * _Stream); 
#line 82
__declspec(dllimport) int __cdecl fputws(const __wchar_t * _Buffer, FILE * _Stream); 
#line 89
__declspec(dllimport) __wchar_t *__cdecl _getws_s(__wchar_t * _Buffer, size_t _BufferCount); 
#line 94
extern "C++" {template< size_t _Size> inline __wchar_t *__cdecl _getws_s(__wchar_t (&_Buffer)[_Size]) throw() { return _getws_s(_Buffer, _Size); } }
#line 101 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) wint_t __cdecl putwc(__wchar_t _Character, FILE * _Stream); 
#line 107
__declspec(dllimport) wint_t __cdecl putwchar(__wchar_t _Character); 
#line 112
__declspec(dllimport) int __cdecl _putws(const __wchar_t * _Buffer); 
#line 117
__declspec(dllimport) wint_t __cdecl ungetwc(wint_t _Character, FILE * _Stream); 
#line 123
__declspec(dllimport) FILE *__cdecl _wfdopen(int _FileHandle, const __wchar_t * _Mode); 
#line 129
__declspec(dllimport) FILE *__cdecl _wfopen(const __wchar_t * _FileName, const __wchar_t * _Mode); 
#line 135
__declspec(dllimport) errno_t __cdecl _wfopen_s(FILE ** _Stream, const __wchar_t * _FileName, const __wchar_t * _Mode); 
#line 143
__declspec(dllimport) FILE *__cdecl _wfreopen(const __wchar_t * _FileName, const __wchar_t * _Mode, FILE * _OldStream); 
#line 150
__declspec(dllimport) errno_t __cdecl _wfreopen_s(FILE ** _Stream, const __wchar_t * _FileName, const __wchar_t * _Mode, FILE * _OldStream); 
#line 158
__declspec(dllimport) FILE *__cdecl _wfsopen(const __wchar_t * _FileName, const __wchar_t * _Mode, int _ShFlag); 
#line 164
__declspec(dllimport) void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 171
__declspec(dllimport) FILE *__cdecl _wpopen(const __wchar_t * _Command, const __wchar_t * _Mode); 
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) int __cdecl _wremove(const __wchar_t * _FileName); 
#line 186
__declspec(dllimport) __declspec(allocator) __wchar_t *__cdecl _wtempnam(const __wchar_t * _Directory, const __wchar_t * _FilePrefix); 
#line 195
__declspec(dllimport) errno_t __cdecl _wtmpnam_s(__wchar_t * _Buffer, size_t _BufferCount); 
#line 200
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl _wtmpnam_s(__wchar_t (&_Buffer)[_Size]) throw() { return _wtmpnam_s(_Buffer, _Size); } }
#line 205 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) __wchar_t *__cdecl _wtmpnam(__wchar_t * _Buffer); 
#line 219 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) wint_t __cdecl _fgetwc_nolock(FILE * _Stream); 
#line 224
__declspec(dllimport) wint_t __cdecl _fputwc_nolock(__wchar_t _Character, FILE * _Stream); 
#line 230
__declspec(dllimport) wint_t __cdecl _getwc_nolock(FILE * _Stream); 
#line 235
__declspec(dllimport) wint_t __cdecl _putwc_nolock(__wchar_t _Character, FILE * _Stream); 
#line 241
__declspec(dllimport) wint_t __cdecl _ungetwc_nolock(wint_t _Character, FILE * _Stream); 
#line 267 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vfwprintf(unsigned __int64 _Options, FILE * _Stream, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 276
__declspec(dllimport) int __cdecl __stdio_common_vfwprintf_s(unsigned __int64 _Options, FILE * _Stream, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 285
__declspec(dllimport) int __cdecl __stdio_common_vfwprintf_p(unsigned __int64 _Options, FILE * _Stream, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 294
__inline int __cdecl _vfwprintf_l(FILE *const 
#line 295
_Stream, const __wchar_t *const 
#line 296
_Format, const _locale_t 
#line 297
_Locale, va_list 
#line 298
_ArgList) 
#line 303 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 304
return __stdio_common_vfwprintf(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 305
} 
#line 309 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vfwprintf(FILE *const 
#line 310
_Stream, const __wchar_t *const 
#line 311
_Format, va_list 
#line 312
_ArgList) 
#line 317 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 318
return _vfwprintf_l(_Stream, _Format, 0, _ArgList); 
#line 319
} 
#line 323 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vfwprintf_s_l(FILE *const 
#line 324
_Stream, const __wchar_t *const 
#line 325
_Format, const _locale_t 
#line 326
_Locale, va_list 
#line 327
_ArgList) 
#line 332 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 333
return __stdio_common_vfwprintf_s(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 334
} 
#line 340 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vfwprintf_s(FILE *const 
#line 341
_Stream, const __wchar_t *const 
#line 342
_Format, va_list 
#line 343
_ArgList) 
#line 348 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 349
return _vfwprintf_s_l(_Stream, _Format, 0, _ArgList); 
#line 350
} 
#line 356 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vfwprintf_p_l(FILE *const 
#line 357
_Stream, const __wchar_t *const 
#line 358
_Format, const _locale_t 
#line 359
_Locale, va_list 
#line 360
_ArgList) 
#line 365 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 366
return __stdio_common_vfwprintf_p(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 367
} 
#line 371 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vfwprintf_p(FILE *const 
#line 372
_Stream, const __wchar_t *const 
#line 373
_Format, va_list 
#line 374
_ArgList) 
#line 379 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 380
return _vfwprintf_p_l(_Stream, _Format, 0, _ArgList); 
#line 381
} 
#line 385 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwprintf_l(const __wchar_t *const 
#line 386
_Format, const _locale_t 
#line 387
_Locale, va_list 
#line 388
_ArgList) 
#line 393 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 394
return _vfwprintf_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 395
} 
#line 399 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vwprintf(const __wchar_t *const 
#line 400
_Format, va_list 
#line 401
_ArgList) 
#line 406 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 407
return _vfwprintf_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 408
} 
#line 412 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwprintf_s_l(const __wchar_t *const 
#line 413
_Format, const _locale_t 
#line 414
_Locale, va_list 
#line 415
_ArgList) 
#line 420 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 421
return _vfwprintf_s_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 422
} 
#line 428 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vwprintf_s(const __wchar_t *const 
#line 429
_Format, va_list 
#line 430
_ArgList) 
#line 435 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 436
return _vfwprintf_s_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 437
} 
#line 443 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwprintf_p_l(const __wchar_t *const 
#line 444
_Format, const _locale_t 
#line 445
_Locale, va_list 
#line 446
_ArgList) 
#line 451 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 452
return _vfwprintf_p_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 453
} 
#line 457 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwprintf_p(const __wchar_t *const 
#line 458
_Format, va_list 
#line 459
_ArgList) 
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 465
return _vfwprintf_p_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 466
} 
#line 470 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwprintf_l(FILE *const 
#line 471
_Stream, const __wchar_t *const 
#line 472
_Format, const _locale_t 
#line 473
_Locale, ...) 
#line 478 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 479
int _Result; 
#line 480
va_list _ArgList; 
#line 481
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 482
_Result = _vfwprintf_l(_Stream, _Format, _Locale, _ArgList); 
#line 483
(void)(_ArgList = ((va_list)0)); 
#line 484
return _Result; 
#line 485
} 
#line 489 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl fwprintf(FILE *const 
#line 490
_Stream, const __wchar_t *const 
#line 491
_Format, ...) 
#line 496 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 497
int _Result; 
#line 498
va_list _ArgList; 
#line 499
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 500
_Result = _vfwprintf_l(_Stream, _Format, 0, _ArgList); 
#line 501
(void)(_ArgList = ((va_list)0)); 
#line 502
return _Result; 
#line 503
} 
#line 507 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwprintf_s_l(FILE *const 
#line 508
_Stream, const __wchar_t *const 
#line 509
_Format, const _locale_t 
#line 510
_Locale, ...) 
#line 515 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 516
int _Result; 
#line 517
va_list _ArgList; 
#line 518
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 519
_Result = _vfwprintf_s_l(_Stream, _Format, _Locale, _ArgList); 
#line 520
(void)(_ArgList = ((va_list)0)); 
#line 521
return _Result; 
#line 522
} 
#line 528 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl fwprintf_s(FILE *const 
#line 529
_Stream, const __wchar_t *const 
#line 530
_Format, ...) 
#line 535 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 536
int _Result; 
#line 537
va_list _ArgList; 
#line 538
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 539
_Result = _vfwprintf_s_l(_Stream, _Format, 0, _ArgList); 
#line 540
(void)(_ArgList = ((va_list)0)); 
#line 541
return _Result; 
#line 542
} 
#line 548 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwprintf_p_l(FILE *const 
#line 549
_Stream, const __wchar_t *const 
#line 550
_Format, const _locale_t 
#line 551
_Locale, ...) 
#line 556 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 557
int _Result; 
#line 558
va_list _ArgList; 
#line 559
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 560
_Result = _vfwprintf_p_l(_Stream, _Format, _Locale, _ArgList); 
#line 561
(void)(_ArgList = ((va_list)0)); 
#line 562
return _Result; 
#line 563
} 
#line 567 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwprintf_p(FILE *const 
#line 568
_Stream, const __wchar_t *const 
#line 569
_Format, ...) 
#line 574 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 575
int _Result; 
#line 576
va_list _ArgList; 
#line 577
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 578
_Result = _vfwprintf_p_l(_Stream, _Format, 0, _ArgList); 
#line 579
(void)(_ArgList = ((va_list)0)); 
#line 580
return _Result; 
#line 581
} 
#line 585 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wprintf_l(const __wchar_t *const 
#line 586
_Format, const _locale_t 
#line 587
_Locale, ...) 
#line 592 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 593
int _Result; 
#line 594
va_list _ArgList; 
#line 595
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 596
_Result = _vfwprintf_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 597
(void)(_ArgList = ((va_list)0)); 
#line 598
return _Result; 
#line 599
} 
#line 603 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl wprintf(const __wchar_t *const 
#line 604
_Format, ...) 
#line 609 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 610
int _Result; 
#line 611
va_list _ArgList; 
#line 612
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 613
_Result = _vfwprintf_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 614
(void)(_ArgList = ((va_list)0)); 
#line 615
return _Result; 
#line 616
} 
#line 620 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wprintf_s_l(const __wchar_t *const 
#line 621
_Format, const _locale_t 
#line 622
_Locale, ...) 
#line 627 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 628
int _Result; 
#line 629
va_list _ArgList; 
#line 630
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 631
_Result = _vfwprintf_s_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 632
(void)(_ArgList = ((va_list)0)); 
#line 633
return _Result; 
#line 634
} 
#line 640 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl wprintf_s(const __wchar_t *const 
#line 641
_Format, ...) 
#line 646 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 647
int _Result; 
#line 648
va_list _ArgList; 
#line 649
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 650
_Result = _vfwprintf_s_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 651
(void)(_ArgList = ((va_list)0)); 
#line 652
return _Result; 
#line 653
} 
#line 659 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wprintf_p_l(const __wchar_t *const 
#line 660
_Format, const _locale_t 
#line 661
_Locale, ...) 
#line 666 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 667
int _Result; 
#line 668
va_list _ArgList; 
#line 669
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 670
_Result = _vfwprintf_p_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 671
(void)(_ArgList = ((va_list)0)); 
#line 672
return _Result; 
#line 673
} 
#line 677 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wprintf_p(const __wchar_t *const 
#line 678
_Format, ...) 
#line 683 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 684
int _Result; 
#line 685
va_list _ArgList; 
#line 686
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 687
_Result = _vfwprintf_p_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 688
(void)(_ArgList = ((va_list)0)); 
#line 689
return _Result; 
#line 690
} 
#line 700 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vfwscanf(unsigned __int64 _Options, FILE * _Stream, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 709
__inline int __cdecl _vfwscanf_l(FILE *const 
#line 710
_Stream, const __wchar_t *const 
#line 711
_Format, const _locale_t 
#line 712
_Locale, va_list 
#line 713
_ArgList) 
#line 718 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 719
return __stdio_common_vfwscanf(*__local_stdio_scanf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 722
} 
#line 726 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vfwscanf(FILE *const 
#line 727
_Stream, const __wchar_t *const 
#line 728
_Format, va_list 
#line 729
_ArgList) 
#line 734 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 735
return _vfwscanf_l(_Stream, _Format, 0, _ArgList); 
#line 736
} 
#line 740 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vfwscanf_s_l(FILE *const 
#line 741
_Stream, const __wchar_t *const 
#line 742
_Format, const _locale_t 
#line 743
_Locale, va_list 
#line 744
_ArgList) 
#line 749 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 750
return __stdio_common_vfwscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Stream, _Format, _Locale, _ArgList); 
#line 753
} 
#line 759 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vfwscanf_s(FILE *const 
#line 760
_Stream, const __wchar_t *const 
#line 761
_Format, va_list 
#line 762
_ArgList) 
#line 767 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 768
return _vfwscanf_s_l(_Stream, _Format, 0, _ArgList); 
#line 769
} 
#line 774 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwscanf_l(const __wchar_t *const 
#line 775
_Format, const _locale_t 
#line 776
_Locale, va_list 
#line 777
_ArgList) 
#line 782 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 783
return _vfwscanf_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 784
} 
#line 788 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vwscanf(const __wchar_t *const 
#line 789
_Format, va_list 
#line 790
_ArgList) 
#line 795 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 796
return _vfwscanf_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 797
} 
#line 801 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vwscanf_s_l(const __wchar_t *const 
#line 802
_Format, const _locale_t 
#line 803
_Locale, va_list 
#line 804
_ArgList) 
#line 809 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 810
return _vfwscanf_s_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 811
} 
#line 817 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vwscanf_s(const __wchar_t *const 
#line 818
_Format, va_list 
#line 819
_ArgList) 
#line 824 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 825
return _vfwscanf_s_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 826
} 
#line 832 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwscanf_l(FILE *const 
#line 833
_Stream, const __wchar_t *const 
#line 834
_Format, const _locale_t 
#line 835
_Locale, ...) 
#line 840 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 841
int _Result; 
#line 842
va_list _ArgList; 
#line 843
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 844
_Result = _vfwscanf_l(_Stream, _Format, _Locale, _ArgList); 
#line 845
(void)(_ArgList = ((va_list)0)); 
#line 846
return _Result; 
#line 847
} 
#line 851 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl fwscanf(FILE *const 
#line 852
_Stream, const __wchar_t *const 
#line 853
_Format, ...) 
#line 858 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 859
int _Result; 
#line 860
va_list _ArgList; 
#line 861
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 862
_Result = _vfwscanf_l(_Stream, _Format, 0, _ArgList); 
#line 863
(void)(_ArgList = ((va_list)0)); 
#line 864
return _Result; 
#line 865
} 
#line 869 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _fwscanf_s_l(FILE *const 
#line 870
_Stream, const __wchar_t *const 
#line 871
_Format, const _locale_t 
#line 872
_Locale, ...) 
#line 877 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 878
int _Result; 
#line 879
va_list _ArgList; 
#line 880
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 881
_Result = _vfwscanf_s_l(_Stream, _Format, _Locale, _ArgList); 
#line 882
(void)(_ArgList = ((va_list)0)); 
#line 883
return _Result; 
#line 884
} 
#line 890 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl fwscanf_s(FILE *const 
#line 891
_Stream, const __wchar_t *const 
#line 892
_Format, ...) 
#line 897 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 898
int _Result; 
#line 899
va_list _ArgList; 
#line 900
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 901
_Result = _vfwscanf_s_l(_Stream, _Format, 0, _ArgList); 
#line 902
(void)(_ArgList = ((va_list)0)); 
#line 903
return _Result; 
#line 904
} 
#line 910 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wscanf_l(const __wchar_t *const 
#line 911
_Format, const _locale_t 
#line 912
_Locale, ...) 
#line 917 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 918
int _Result; 
#line 919
va_list _ArgList; 
#line 920
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 921
_Result = _vfwscanf_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 922
(void)(_ArgList = ((va_list)0)); 
#line 923
return _Result; 
#line 924
} 
#line 928 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl wscanf(const __wchar_t *const 
#line 929
_Format, ...) 
#line 934 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 935
int _Result; 
#line 936
va_list _ArgList; 
#line 937
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 938
_Result = _vfwscanf_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 939
(void)(_ArgList = ((va_list)0)); 
#line 940
return _Result; 
#line 941
} 
#line 945 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _wscanf_s_l(const __wchar_t *const 
#line 946
_Format, const _locale_t 
#line 947
_Locale, ...) 
#line 952 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 953
int _Result; 
#line 954
va_list _ArgList; 
#line 955
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 956
_Result = _vfwscanf_s_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 957
(void)(_ArgList = ((va_list)0)); 
#line 958
return _Result; 
#line 959
} 
#line 965 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl wscanf_s(const __wchar_t *const 
#line 966
_Format, ...) 
#line 971 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 972
int _Result; 
#line 973
va_list _ArgList; 
#line 974
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 975
_Result = _vfwscanf_s_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 976
(void)(_ArgList = ((va_list)0)); 
#line 977
return _Result; 
#line 978
} 
#line 1001 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vswprintf(unsigned __int64 _Options, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1012
__declspec(dllimport) int __cdecl __stdio_common_vswprintf_s(unsigned __int64 _Options, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1023
__declspec(dllimport) int __cdecl __stdio_common_vsnwprintf_s(unsigned __int64 _Options, __wchar_t * _Buffer, size_t _BufferCount, size_t _MaxCount, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1035
__declspec(dllimport) int __cdecl __stdio_common_vswprintf_p(unsigned __int64 _Options, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1046
__inline int __cdecl _vsnwprintf_l(__wchar_t *const 
#line 1047
_Buffer, const size_t 
#line 1048
_BufferCount, const __wchar_t *const 
#line 1049
_Format, const _locale_t 
#line 1050
_Locale, va_list 
#line 1051
_ArgList) 
#line 1056 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1057
const int _Result = __stdio_common_vswprintf((*__local_stdio_printf_options()) | (1Ui64 << 0), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1061
return (_Result < 0) ? -1 : _Result; 
#line 1062
} 
#line 1065 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(push)
#pragma warning(disable: 4793)
#line 1070
__inline int __cdecl _vsnwprintf_s_l(__wchar_t *const 
#line 1071
_Buffer, const size_t 
#line 1072
_BufferCount, const size_t 
#line 1073
_MaxCount, const __wchar_t *const 
#line 1074
_Format, const _locale_t 
#line 1075
_Locale, va_list 
#line 1076
_ArgList) 
#line 1081 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1082
const int _Result = __stdio_common_vsnwprintf_s(*__local_stdio_printf_options(), _Buffer, _BufferCount, _MaxCount, _Format, _Locale, _ArgList); 
#line 1086
return (_Result < 0) ? -1 : _Result; 
#line 1087
} 
#line 1092 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vsnwprintf_s(__wchar_t *const 
#line 1093
_Buffer, const size_t 
#line 1094
_BufferCount, const size_t 
#line 1095
_MaxCount, const __wchar_t *const 
#line 1096
_Format, va_list 
#line 1097
_ArgList) 
#line 1102 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1103
return _vsnwprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, 0, _ArgList); 
#line 1104
} 
#line 1107 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwprintf(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, ...); __inline int __cdecl _vsnwprintf(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, va_list _Args); 
#line 1116 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(pop)
#line 1120
__inline int __cdecl _vsnwprintf(__wchar_t *
#line 1121
_Buffer, size_t 
#line 1122
_BufferCount, const __wchar_t *
#line 1123
_Format, va_list 
#line 1124
_ArgList) 
#line 1129 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _vsnwprintf_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#pragma warning(pop)
} 
#line 1137 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl _vsnwprintf_s(__wchar_t (&_Buffer)[_Size], ::size_t _BufferCount, const __wchar_t *_Format, ::va_list _ArgList) throw() { return _vsnwprintf_s(_Buffer, _Size, _BufferCount, _Format, _ArgList); } }
#line 1148 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_c_l(__wchar_t *const 
#line 1149
_Buffer, const size_t 
#line 1150
_BufferCount, const __wchar_t *const 
#line 1151
_Format, const _locale_t 
#line 1152
_Locale, va_list 
#line 1153
_ArgList) 
#line 1158 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1159
const int _Result = __stdio_common_vswprintf(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1163
return (_Result < 0) ? -1 : _Result; 
#line 1164
} 
#line 1169 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_c(__wchar_t *const 
#line 1170
_Buffer, const size_t 
#line 1171
_BufferCount, const __wchar_t *const 
#line 1172
_Format, va_list 
#line 1173
_ArgList) 
#line 1178 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1179
return _vswprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1180
} 
#line 1185 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_l(__wchar_t *const 
#line 1186
_Buffer, const size_t 
#line 1187
_BufferCount, const __wchar_t *const 
#line 1188
_Format, const _locale_t 
#line 1189
_Locale, va_list 
#line 1190
_ArgList) 
#line 1195 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _vswprintf_c_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#pragma warning(pop)
} 
#line 1205 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl __vswprintf_l(__wchar_t *const 
#line 1206
_Buffer, const __wchar_t *const 
#line 1207
_Format, const _locale_t 
#line 1208
_Locale, va_list 
#line 1209
_ArgList) 
#line 1214 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1215
return _vswprintf_l(_Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 1216
} 
#line 1221 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf(__wchar_t *const 
#line 1222
_Buffer, const __wchar_t *const 
#line 1223
_Format, va_list 
#line 1224
_ArgList) 
#line 1229 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1230
return _vswprintf_l(_Buffer, (size_t)(-1), _Format, 0, _ArgList); 
#line 1231
} 
#line 1236 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vswprintf(__wchar_t *const 
#line 1237
_Buffer, const size_t 
#line 1238
_BufferCount, const __wchar_t *const 
#line 1239
_Format, va_list 
#line 1240
_ArgList) 
#line 1245 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1246
return _vswprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1247
} 
#line 1252 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_s_l(__wchar_t *const 
#line 1253
_Buffer, const size_t 
#line 1254
_BufferCount, const __wchar_t *const 
#line 1255
_Format, const _locale_t 
#line 1256
_Locale, va_list 
#line 1257
_ArgList) 
#line 1262 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1263
const int _Result = __stdio_common_vswprintf_s(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1267
return (_Result < 0) ? -1 : _Result; 
#line 1268
} 
#line 1274 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vswprintf_s(__wchar_t *const 
#line 1275
_Buffer, const size_t 
#line 1276
_BufferCount, const __wchar_t *const 
#line 1277
_Format, va_list 
#line 1278
_ArgList) 
#line 1283 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1284
return _vswprintf_s_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1285
} 
#line 1290 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl vswprintf_s(__wchar_t (&_Buffer)[_Size], const __wchar_t *_Format, ::va_list _ArgList) throw() { return vswprintf_s(_Buffer, _Size, _Format, _ArgList); } }
#line 1300 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_p_l(__wchar_t *const 
#line 1301
_Buffer, const size_t 
#line 1302
_BufferCount, const __wchar_t *const 
#line 1303
_Format, const _locale_t 
#line 1304
_Locale, va_list 
#line 1305
_ArgList) 
#line 1310 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1311
const int _Result = __stdio_common_vswprintf_p(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1315
return (_Result < 0) ? -1 : _Result; 
#line 1316
} 
#line 1321 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswprintf_p(__wchar_t *const 
#line 1322
_Buffer, const size_t 
#line 1323
_BufferCount, const __wchar_t *const 
#line 1324
_Format, va_list 
#line 1325
_ArgList) 
#line 1330 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1331
return _vswprintf_p_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1332
} 
#line 1337 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vscwprintf_l(const __wchar_t *const 
#line 1338
_Format, const _locale_t 
#line 1339
_Locale, va_list 
#line 1340
_ArgList) 
#line 1345 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1346
const int _Result = __stdio_common_vswprintf((*__local_stdio_printf_options()) | (1Ui64 << 1), 0, 0, _Format, _Locale, _ArgList); 
#line 1350
return (_Result < 0) ? -1 : _Result; 
#line 1351
} 
#line 1356 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vscwprintf(const __wchar_t *const 
#line 1357
_Format, va_list 
#line 1358
_ArgList) 
#line 1363 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1364
return _vscwprintf_l(_Format, 0, _ArgList); 
#line 1365
} 
#line 1370 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vscwprintf_p_l(const __wchar_t *const 
#line 1371
_Format, const _locale_t 
#line 1372
_Locale, va_list 
#line 1373
_ArgList) 
#line 1378 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1379
const int _Result = __stdio_common_vswprintf_p((*__local_stdio_printf_options()) | (1Ui64 << 1), 0, 0, _Format, _Locale, _ArgList); 
#line 1383
return (_Result < 0) ? -1 : _Result; 
#line 1384
} 
#line 1389 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vscwprintf_p(const __wchar_t *const 
#line 1390
_Format, va_list 
#line 1391
_ArgList) 
#line 1396 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1397
return _vscwprintf_p_l(_Format, 0, _ArgList); 
#line 1398
} 
#line 1403 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl __swprintf_l(__wchar_t *const 
#line 1404
_Buffer, const __wchar_t *const 
#line 1405
_Format, const _locale_t 
#line 1406
_Locale, ...) 
#line 1411 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1412
int _Result; 
#line 1413
va_list _ArgList; 
#line 1414
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1415
_Result = __vswprintf_l(_Buffer, _Format, _Locale, _ArgList); 
#line 1416
(void)(_ArgList = ((va_list)0)); 
#line 1417
return _Result; 
#line 1418
} 
#line 1423 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf_l(__wchar_t *const 
#line 1424
_Buffer, const size_t 
#line 1425
_BufferCount, const __wchar_t *const 
#line 1426
_Format, const _locale_t 
#line 1427
_Locale, ...) 
#line 1432 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1433
int _Result; 
#line 1434
va_list _ArgList; 
#line 1435
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1436
_Result = _vswprintf_c_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1437
(void)(_ArgList = ((va_list)0)); 
#line 1438
return _Result; 
#line 1439
} 
#line 1444 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf(__wchar_t *const 
#line 1445
_Buffer, const __wchar_t *const 
#line 1446
_Format, ...) 
#line 1451 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1452
int _Result; 
#line 1453
va_list _ArgList; 
#line 1454
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1455
_Result = __vswprintf_l(_Buffer, _Format, 0, _ArgList); 
#line 1456
(void)(_ArgList = ((va_list)0)); 
#line 1457
return _Result; 
#line 1458
} 
#line 1463 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl swprintf(__wchar_t *const 
#line 1464
_Buffer, const size_t 
#line 1465
_BufferCount, const __wchar_t *const 
#line 1466
_Format, ...) 
#line 1471 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1472
int _Result; 
#line 1473
va_list _ArgList; 
#line 1474
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1475
_Result = _vswprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1476
(void)(_ArgList = ((va_list)0)); 
#line 1477
return _Result; 
#line 1478
} 
#line 1481 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(push)
#line 1484
#pragma warning(disable:4793 4996)
#line 1486
__inline int __cdecl __swprintf_l(__wchar_t * _Buffer, const __wchar_t * _Format, _locale_t _Locale, ...); __inline int __cdecl __vswprintf_l(__wchar_t * _Buffer, const __wchar_t * _Format, _locale_t _Locale, va_list _Args); 
#line 1495 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf(__wchar_t * _Buffer, const __wchar_t * _Format, ...); __inline int __cdecl _vswprintf(__wchar_t * _Buffer, const __wchar_t * _Format, va_list _Args); 
#line 1502 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(pop)
#line 1506
__inline int __cdecl _swprintf_s_l(__wchar_t *const 
#line 1507
_Buffer, const size_t 
#line 1508
_BufferCount, const __wchar_t *const 
#line 1509
_Format, const _locale_t 
#line 1510
_Locale, ...) 
#line 1515 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1516
int _Result; 
#line 1517
va_list _ArgList; 
#line 1518
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1519
_Result = _vswprintf_s_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1520
(void)(_ArgList = ((va_list)0)); 
#line 1521
return _Result; 
#line 1522
} 
#line 1528 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl swprintf_s(__wchar_t *const 
#line 1529
_Buffer, const size_t 
#line 1530
_BufferCount, const __wchar_t *const 
#line 1531
_Format, ...) 
#line 1536 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1537
int _Result; 
#line 1538
va_list _ArgList; 
#line 1539
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1540
_Result = _vswprintf_s_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1541
(void)(_ArgList = ((va_list)0)); 
#line 1542
return _Result; 
#line 1543
} 
#line 1548 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
extern "C++" {__pragma( warning(push)) __pragma( warning(disable: 4793)) template< size_t _Size> inline int __cdecl swprintf_s(__wchar_t (&_Buffer)[_Size], const __wchar_t *_Format, ...) throw() { ::va_list _ArgList; (void)(__vcrt_va_start_verify_argument_type< const __wchar_t *> (), ((void)__va_start(&_ArgList, _Format))); return vswprintf_s(_Buffer, _Size, _Format, _ArgList); } __pragma( warning(pop)) }
#line 1557 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf_p_l(__wchar_t *const 
#line 1558
_Buffer, const size_t 
#line 1559
_BufferCount, const __wchar_t *const 
#line 1560
_Format, const _locale_t 
#line 1561
_Locale, ...) 
#line 1566 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1567
int _Result; 
#line 1568
va_list _ArgList; 
#line 1569
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1570
_Result = _vswprintf_p_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1571
(void)(_ArgList = ((va_list)0)); 
#line 1572
return _Result; 
#line 1573
} 
#line 1578 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf_p(__wchar_t *const 
#line 1579
_Buffer, const size_t 
#line 1580
_BufferCount, const __wchar_t *const 
#line 1581
_Format, ...) 
#line 1586 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1587
int _Result; 
#line 1588
va_list _ArgList; 
#line 1589
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1590
_Result = _vswprintf_p_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1591
(void)(_ArgList = ((va_list)0)); 
#line 1592
return _Result; 
#line 1593
} 
#line 1598 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf_c_l(__wchar_t *const 
#line 1599
_Buffer, const size_t 
#line 1600
_BufferCount, const __wchar_t *const 
#line 1601
_Format, const _locale_t 
#line 1602
_Locale, ...) 
#line 1607 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1608
int _Result; 
#line 1609
va_list _ArgList; 
#line 1610
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1611
_Result = _vswprintf_c_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1612
(void)(_ArgList = ((va_list)0)); 
#line 1613
return _Result; 
#line 1614
} 
#line 1619 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swprintf_c(__wchar_t *const 
#line 1620
_Buffer, const size_t 
#line 1621
_BufferCount, const __wchar_t *const 
#line 1622
_Format, ...) 
#line 1627 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1628
int _Result; 
#line 1629
va_list _ArgList; 
#line 1630
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1631
_Result = _vswprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1632
(void)(_ArgList = ((va_list)0)); 
#line 1633
return _Result; 
#line 1634
} 
#line 1639 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwprintf_l(__wchar_t *const 
#line 1640
_Buffer, const size_t 
#line 1641
_BufferCount, const __wchar_t *const 
#line 1642
_Format, const _locale_t 
#line 1643
_Locale, ...) 
#line 1648 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1649
int _Result; 
#line 1650
va_list _ArgList; 
#line 1651
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1653
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsnwprintf_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#pragma warning(pop)
#line 1658
(void)(_ArgList = ((va_list)0)); 
#line 1659
return _Result; 
#line 1660
} 
#line 1665 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwprintf(__wchar_t *
#line 1666
_Buffer, size_t 
#line 1667
_BufferCount, const __wchar_t *
#line 1668
_Format, ...) 
#line 1673 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1674
int _Result; 
#line 1675
va_list _ArgList; 
#line 1676
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1678
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsnwprintf_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#pragma warning(pop)
#line 1683
(void)(_ArgList = ((va_list)0)); 
#line 1684
return _Result; 
#line 1685
} 
#line 1690 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwprintf_s_l(__wchar_t *const 
#line 1691
_Buffer, const size_t 
#line 1692
_BufferCount, const size_t 
#line 1693
_MaxCount, const __wchar_t *const 
#line 1694
_Format, const _locale_t 
#line 1695
_Locale, ...) 
#line 1700 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1701
int _Result; 
#line 1702
va_list _ArgList; 
#line 1703
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1704
_Result = _vsnwprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, _Locale, _ArgList); 
#line 1705
(void)(_ArgList = ((va_list)0)); 
#line 1706
return _Result; 
#line 1707
} 
#line 1712 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwprintf_s(__wchar_t *const 
#line 1713
_Buffer, const size_t 
#line 1714
_BufferCount, const size_t 
#line 1715
_MaxCount, const __wchar_t *const 
#line 1716
_Format, ...) 
#line 1721 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1722
int _Result; 
#line 1723
va_list _ArgList; 
#line 1724
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1725
_Result = _vsnwprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, 0, _ArgList); 
#line 1726
(void)(_ArgList = ((va_list)0)); 
#line 1727
return _Result; 
#line 1728
} 
#line 1731 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
extern "C++" {__pragma( warning(push)) __pragma( warning(disable: 4793)) template< size_t _Size> inline int __cdecl _snwprintf_s(__wchar_t (&_Buffer)[_Size], ::size_t _BufferCount, const __wchar_t *_Format, ...) throw() { ::va_list _ArgList; (void)(__vcrt_va_start_verify_argument_type< const __wchar_t *> (), ((void)__va_start(&_ArgList, _Format))); return _vsnwprintf_s(_Buffer, _Size, _BufferCount, _Format, _ArgList); } __pragma( warning(pop)) }
#line 1740 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _scwprintf_l(const __wchar_t *const 
#line 1741
_Format, const _locale_t 
#line 1742
_Locale, ...) 
#line 1747 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1748
int _Result; 
#line 1749
va_list _ArgList; 
#line 1750
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1751
_Result = _vscwprintf_l(_Format, _Locale, _ArgList); 
#line 1752
(void)(_ArgList = ((va_list)0)); 
#line 1753
return _Result; 
#line 1754
} 
#line 1759 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _scwprintf(const __wchar_t *const 
#line 1760
_Format, ...) 
#line 1765 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1766
int _Result; 
#line 1767
va_list _ArgList; 
#line 1768
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1769
_Result = _vscwprintf_l(_Format, 0, _ArgList); 
#line 1770
(void)(_ArgList = ((va_list)0)); 
#line 1771
return _Result; 
#line 1772
} 
#line 1777 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _scwprintf_p_l(const __wchar_t *const 
#line 1778
_Format, const _locale_t 
#line 1779
_Locale, ...) 
#line 1784 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1785
int _Result; 
#line 1786
va_list _ArgList; 
#line 1787
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1788
_Result = _vscwprintf_p_l(_Format, _Locale, _ArgList); 
#line 1789
(void)(_ArgList = ((va_list)0)); 
#line 1790
return _Result; 
#line 1791
} 
#line 1796 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _scwprintf_p(const __wchar_t *const 
#line 1797
_Format, ...) 
#line 1802 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1803
int _Result; 
#line 1804
va_list _ArgList; 
#line 1805
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1806
_Result = _vscwprintf_p_l(_Format, 0, _ArgList); 
#line 1807
(void)(_ArgList = ((va_list)0)); 
#line 1808
return _Result; 
#line 1809
} 
#line 1814 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(push)
#pragma warning(disable: 4141 4412 4793 4996 6054)
#line 1820
extern "C++" 
#line 1819
__declspec(deprecated("function has been changed to conform with the ISO C standard, adding an extra character count parameter. To use the traditional " "Microsoft version, set _CRT_NON_CONFORMING_SWPRINTFS.")) inline int 
#line 1820
swprintf(__wchar_t *const 
#line 1821
_Buffer, const __wchar_t *const 
#line 1822
_Format, ...) throw() 
#line 1824
{ 
#line 1825
int _Result; 
#line 1826
va_list _ArgList; 
#line 1827
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#pragma warning(suppress: 28719)
_Result = vswprintf(_Buffer, 2147483647, _Format, _ArgList); 
#line 1830
(void)(_ArgList = ((va_list)0)); 
#line 1831
return _Result; 
#line 1832
} 
#line 1835
extern "C++" 
#line 1834
__declspec(deprecated("function has been changed to conform with the ISO C standard, adding an extra character count parameter. To use the traditional " "Microsoft version, set _CRT_NON_CONFORMING_SWPRINTFS.")) inline int __cdecl 
#line 1835
vswprintf(__wchar_t *const 
#line 1836
_Buffer, const __wchar_t *const 
#line 1837
_Format, va_list 
#line 1838
_ArgList) throw() 
#line 1840
{ 
#pragma warning(suppress: 28719)
return vswprintf(_Buffer, 2147483647, _Format, _ArgList); 
#line 1843
} 
#line 1846
extern "C++" 
#line 1845
__declspec(deprecated("function has been changed to conform with the ISO C standard, adding an extra character count parameter. To use the traditional " "Microsoft version, set _CRT_NON_CONFORMING_SWPRINTFS.")) inline int 
#line 1846
_swprintf_l(__wchar_t *const 
#line 1847
_Buffer, const __wchar_t *const 
#line 1848
_Format, const _locale_t 
#line 1849
_Locale, ...) throw() 
#line 1851
{ 
#line 1852
int _Result; 
#line 1853
va_list _ArgList; 
#line 1854
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1855
_Result = _vswprintf_l(_Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 1856
(void)(_ArgList = ((va_list)0)); 
#line 1857
return _Result; 
#line 1858
} 
#line 1861
extern "C++" 
#line 1860
__declspec(deprecated("function has been changed to conform with the ISO C standard, adding an extra character count parameter. To use the traditional " "Microsoft version, set _CRT_NON_CONFORMING_SWPRINTFS.")) inline int __cdecl 
#line 1861
_vswprintf_l(__wchar_t *const 
#line 1862
_Buffer, const __wchar_t *const 
#line 1863
_Format, const _locale_t 
#line 1864
_Locale, va_list 
#line 1865
_ArgList) throw() 
#line 1867
{ 
#line 1868
return _vswprintf_l(_Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 1869
} 
#line 1873 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
#pragma warning(pop)
#line 1890 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vswscanf(unsigned __int64 _Options, const __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1901
__inline int __cdecl _vswscanf_l(const __wchar_t *const 
#line 1902
_Buffer, const __wchar_t *const 
#line 1903
_Format, const _locale_t 
#line 1904
_Locale, va_list 
#line 1905
_ArgList) 
#line 1910 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1911
return __stdio_common_vswscanf(*__local_stdio_scanf_options(), _Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 1914
} 
#line 1919 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vswscanf(const __wchar_t *
#line 1920
_Buffer, const __wchar_t *
#line 1921
_Format, va_list 
#line 1922
_ArgList) 
#line 1927 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1928
return _vswscanf_l(_Buffer, _Format, 0, _ArgList); 
#line 1929
} 
#line 1934 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vswscanf_s_l(const __wchar_t *const 
#line 1935
_Buffer, const __wchar_t *const 
#line 1936
_Format, const _locale_t 
#line 1937
_Locale, va_list 
#line 1938
_ArgList) 
#line 1943 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1944
return __stdio_common_vswscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 1947
} 
#line 1954 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl vswscanf_s(const __wchar_t *const 
#line 1955
_Buffer, const __wchar_t *const 
#line 1956
_Format, va_list 
#line 1957
_ArgList) 
#line 1962 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1963
return _vswscanf_s_l(_Buffer, _Format, 0, _ArgList); 
#line 1964
} 
#line 1969 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl vswscanf_s(__wchar_t (&_Buffer)[_Size], const __wchar_t *_Format, ::va_list _Args) throw() { return vswscanf_s(_Buffer, _Size, _Format, _Args); } }
#line 1979 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vsnwscanf_l(const __wchar_t *const 
#line 1980
_Buffer, const size_t 
#line 1981
_BufferCount, const __wchar_t *const 
#line 1982
_Format, const _locale_t 
#line 1983
_Locale, va_list 
#line 1984
_ArgList) 
#line 1989 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 1990
return __stdio_common_vswscanf(*__local_stdio_scanf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1993
} 
#line 1998 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _vsnwscanf_s_l(const __wchar_t *const 
#line 1999
_Buffer, const size_t 
#line 2000
_BufferCount, const __wchar_t *const 
#line 2001
_Format, const _locale_t 
#line 2002
_Locale, va_list 
#line 2003
_ArgList) 
#line 2008 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2009
return __stdio_common_vswscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 2012
} 
#line 2017 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swscanf_l(const __wchar_t *const 
#line 2018
_Buffer, const __wchar_t *const 
#line 2019
_Format, _locale_t 
#line 2020
_Locale, ...) 
#line 2025 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2026
int _Result; 
#line 2027
va_list _ArgList; 
#line 2028
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2029
_Result = _vswscanf_l(_Buffer, _Format, _Locale, _ArgList); 
#line 2030
(void)(_ArgList = ((va_list)0)); 
#line 2031
return _Result; 
#line 2032
} 
#line 2037 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl swscanf(const __wchar_t *const 
#line 2038
_Buffer, const __wchar_t *const 
#line 2039
_Format, ...) 
#line 2044 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2045
int _Result; 
#line 2046
va_list _ArgList; 
#line 2047
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2048
_Result = _vswscanf_l(_Buffer, _Format, 0, _ArgList); 
#line 2049
(void)(_ArgList = ((va_list)0)); 
#line 2050
return _Result; 
#line 2051
} 
#line 2056 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _swscanf_s_l(const __wchar_t *const 
#line 2057
_Buffer, const __wchar_t *const 
#line 2058
_Format, const _locale_t 
#line 2059
_Locale, ...) 
#line 2064 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2065
int _Result; 
#line 2066
va_list _ArgList; 
#line 2067
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2068
_Result = _vswscanf_s_l(_Buffer, _Format, _Locale, _ArgList); 
#line 2069
(void)(_ArgList = ((va_list)0)); 
#line 2070
return _Result; 
#line 2071
} 
#line 2078 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl swscanf_s(const __wchar_t *const 
#line 2079
_Buffer, const __wchar_t *const 
#line 2080
_Format, ...) 
#line 2085 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2086
int _Result; 
#line 2087
va_list _ArgList; 
#line 2088
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2089
_Result = _vswscanf_s_l(_Buffer, _Format, 0, _ArgList); 
#line 2090
(void)(_ArgList = ((va_list)0)); 
#line 2091
return _Result; 
#line 2092
} 
#line 2099 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwscanf_l(const __wchar_t *const 
#line 2100
_Buffer, const size_t 
#line 2101
_BufferCount, const __wchar_t *const 
#line 2102
_Format, const _locale_t 
#line 2103
_Locale, ...) 
#line 2108 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2109
int _Result; 
#line 2110
va_list _ArgList; 
#line 2111
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2113
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsnwscanf_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#pragma warning(pop)
#line 2118
(void)(_ArgList = ((va_list)0)); 
#line 2119
return _Result; 
#line 2120
} 
#line 2125 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwscanf(const __wchar_t *const 
#line 2126
_Buffer, const size_t 
#line 2127
_BufferCount, const __wchar_t *const 
#line 2128
_Format, ...) 
#line 2133 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2134
int _Result; 
#line 2135
va_list _ArgList; 
#line 2136
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2138
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsnwscanf_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#pragma warning(pop)
#line 2143
(void)(_ArgList = ((va_list)0)); 
#line 2144
return _Result; 
#line 2145
} 
#line 2150 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwscanf_s_l(const __wchar_t *const 
#line 2151
_Buffer, const size_t 
#line 2152
_BufferCount, const __wchar_t *const 
#line 2153
_Format, const _locale_t 
#line 2154
_Locale, ...) 
#line 2159 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2160
int _Result; 
#line 2161
va_list _ArgList; 
#line 2162
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2163
_Result = _vsnwscanf_s_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 2164
(void)(_ArgList = ((va_list)0)); 
#line 2165
return _Result; 
#line 2166
} 
#line 2171 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
__inline int __cdecl _snwscanf_s(const __wchar_t *const 
#line 2172
_Buffer, const size_t 
#line 2173
_BufferCount, const __wchar_t *const 
#line 2174
_Format, ...) 
#line 2179 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
{ 
#line 2180
int _Result; 
#line 2181
va_list _ArgList; 
#line 2182
(void)(__vcrt_va_start_verify_argument_type< const __wchar_t *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2183
_Result = _vsnwscanf_s_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 2184
(void)(_ArgList = ((va_list)0)); 
#line 2185
return _Result; 
#line 2186
} 
#line 2195 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\corecrt_wstdio.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
typedef __int64 fpos_t; 
#line 73
__declspec(dllimport) errno_t __cdecl _get_stream_buffer_pointers(FILE * _Stream, char *** _Base, char *** _Pointer, int ** _Count); 
#line 89
__declspec(dllimport) errno_t __cdecl clearerr_s(FILE * _Stream); 
#line 94
__declspec(dllimport) errno_t __cdecl fopen_s(FILE ** _Stream, const char * _FileName, const char * _Mode); 
#line 102
__declspec(dllimport) size_t __cdecl fread_s(void * _Buffer, size_t _BufferSize, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 111
__declspec(dllimport) errno_t __cdecl freopen_s(FILE ** _Stream, const char * _FileName, const char * _Mode, FILE * _OldStream); 
#line 119
__declspec(dllimport) char *__cdecl gets_s(char * _Buffer, rsize_t _Size); 
#line 125
__declspec(dllimport) errno_t __cdecl tmpfile_s(FILE ** _Stream); 
#line 131
__declspec(dllimport) errno_t __cdecl tmpnam_s(char * _Buffer, rsize_t _Size); 
#line 138 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) void __cdecl clearerr(FILE * _Stream); 
#line 144
__declspec(dllimport) int __cdecl fclose(FILE * _Stream); 
#line 149
__declspec(dllimport) int __cdecl _fcloseall(); 
#line 152
__declspec(dllimport) FILE *__cdecl _fdopen(int _FileHandle, const char * _Mode); 
#line 158
__declspec(dllimport) int __cdecl feof(FILE * _Stream); 
#line 163
__declspec(dllimport) int __cdecl ferror(FILE * _Stream); 
#line 168
__declspec(dllimport) int __cdecl fflush(FILE * _Stream); 
#line 174
__declspec(dllimport) int __cdecl fgetc(FILE * _Stream); 
#line 179
__declspec(dllimport) int __cdecl _fgetchar(); 
#line 183
__declspec(dllimport) int __cdecl fgetpos(FILE * _Stream, fpos_t * _Position); 
#line 190
__declspec(dllimport) char *__cdecl fgets(char * _Buffer, int _MaxCount, FILE * _Stream); 
#line 197
__declspec(dllimport) int __cdecl _fileno(FILE * _Stream); 
#line 202
__declspec(dllimport) int __cdecl _flushall(); 
#line 205
__declspec(dllimport) FILE *__cdecl fopen(const char * _FileName, const char * _Mode); 
#line 213
__declspec(dllimport) int __cdecl fputc(int _Character, FILE * _Stream); 
#line 219
__declspec(dllimport) int __cdecl _fputchar(int _Character); 
#line 225
__declspec(dllimport) int __cdecl fputs(const char * _Buffer, FILE * _Stream); 
#line 231
__declspec(dllimport) size_t __cdecl fread(void * _Buffer, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 240
__declspec(dllimport) FILE *__cdecl freopen(const char * _FileName, const char * _Mode, FILE * _Stream); 
#line 247
__declspec(dllimport) FILE *__cdecl _fsopen(const char * _FileName, const char * _Mode, int _ShFlag); 
#line 255
__declspec(dllimport) int __cdecl fsetpos(FILE * _Stream, const fpos_t * _Position); 
#line 262
__declspec(dllimport) int __cdecl fseek(FILE * _Stream, long _Offset, int _Origin); 
#line 270
__declspec(dllimport) int __cdecl _fseeki64(FILE * _Stream, __int64 _Offset, int _Origin); 
#line 278
__declspec(dllimport) long __cdecl ftell(FILE * _Stream); 
#line 284
__declspec(dllimport) __int64 __cdecl _ftelli64(FILE * _Stream); 
#line 289
__declspec(dllimport) size_t __cdecl fwrite(const void * _Buffer, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 298
__declspec(dllimport) int __cdecl getc(FILE * _Stream); 
#line 303
__declspec(dllimport) int __cdecl getchar(); 
#line 306
__declspec(dllimport) int __cdecl _getmaxstdio(); 
#line 308
extern "C++" {template< size_t _Size> inline char *__cdecl gets_s(char (&_Buffer)[_Size]) throw() { return gets_s(_Buffer, _Size); } }
#line 313 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl _getw(FILE * _Stream); 
#line 317
__declspec(dllimport) void __cdecl perror(const char * _ErrorMessage); 
#line 325
__declspec(dllimport) int __cdecl _pclose(FILE * _Stream); 
#line 330
__declspec(dllimport) FILE *__cdecl _popen(const char * _Command, const char * _Mode); 
#line 339 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl putc(int _Character, FILE * _Stream); 
#line 345
__declspec(dllimport) int __cdecl putchar(int _Character); 
#line 350
__declspec(dllimport) int __cdecl puts(const char * _Buffer); 
#line 356
__declspec(dllimport) int __cdecl _putw(int _Word, FILE * _Stream); 
#line 363
__declspec(dllimport) int __cdecl remove(const char * _FileName); 
#line 368
__declspec(dllimport) int __cdecl rename(const char * _OldFileName, const char * _NewFileName); 
#line 373
__declspec(dllimport) int __cdecl _unlink(const char * _FileName); 
#line 380
__declspec(dllimport) int __cdecl unlink(const char * _FileName); 
#line 388 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) void __cdecl rewind(FILE * _Stream); 
#line 393
__declspec(dllimport) int __cdecl _rmtmp(); 
#line 396
__declspec(dllimport) void __cdecl setbuf(FILE * _Stream, char * _Buffer); 
#line 402
__declspec(dllimport) int __cdecl _setmaxstdio(int _Maximum); 
#line 408
__declspec(dllimport) int __cdecl setvbuf(FILE * _Stream, char * _Buffer, int _Mode, size_t _Size); 
#line 421 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) __declspec(allocator) char *__cdecl _tempnam(const char * _DirectoryName, const char * _FilePrefix); 
#line 431 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) FILE *__cdecl tmpfile(); 
#line 433
extern "C++" {template< size_t _Size> inline ::errno_t __cdecl tmpnam_s(char (&_Buffer)[_Size]) throw() { return tmpnam_s(_Buffer, _Size); } }
#line 438 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) char *__cdecl tmpnam(char * _Buffer); 
#line 446 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl ungetc(int _Character, FILE * _Stream); 
#line 458
__declspec(dllimport) void __cdecl _lock_file(FILE * _Stream); 
#line 462
__declspec(dllimport) void __cdecl _unlock_file(FILE * _Stream); 
#line 468
__declspec(dllimport) int __cdecl _fclose_nolock(FILE * _Stream); 
#line 474
__declspec(dllimport) int __cdecl _fflush_nolock(FILE * _Stream); 
#line 480
__declspec(dllimport) int __cdecl _fgetc_nolock(FILE * _Stream); 
#line 486
__declspec(dllimport) int __cdecl _fputc_nolock(int _Character, FILE * _Stream); 
#line 492
__declspec(dllimport) size_t __cdecl _fread_nolock(void * _Buffer, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 501
__declspec(dllimport) size_t __cdecl _fread_nolock_s(void * _Buffer, size_t _BufferSize, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 510
__declspec(dllimport) int __cdecl _fseek_nolock(FILE * _Stream, long _Offset, int _Origin); 
#line 517
__declspec(dllimport) int __cdecl _fseeki64_nolock(FILE * _Stream, __int64 _Offset, int _Origin); 
#line 524
__declspec(dllimport) long __cdecl _ftell_nolock(FILE * _Stream); 
#line 529
__declspec(dllimport) __int64 __cdecl _ftelli64_nolock(FILE * _Stream); 
#line 534
__declspec(dllimport) size_t __cdecl _fwrite_nolock(const void * _Buffer, size_t _ElementSize, size_t _ElementCount, FILE * _Stream); 
#line 542
__declspec(dllimport) int __cdecl _getc_nolock(FILE * _Stream); 
#line 547
__declspec(dllimport) int __cdecl _putc_nolock(int _Character, FILE * _Stream); 
#line 553
__declspec(dllimport) int __cdecl _ungetc_nolock(int _Character, FILE * _Stream); 
#line 583 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int *__cdecl __p__commode(); 
#line 603 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vfprintf(unsigned __int64 _Options, FILE * _Stream, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 611
__declspec(dllimport) int __cdecl __stdio_common_vfprintf_s(unsigned __int64 _Options, FILE * _Stream, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 620
__declspec(dllimport) int __cdecl __stdio_common_vfprintf_p(unsigned __int64 _Options, FILE * _Stream, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 629
__inline int __cdecl _vfprintf_l(FILE *const 
#line 630
_Stream, const char *const 
#line 631
_Format, const _locale_t 
#line 632
_Locale, va_list 
#line 633
_ArgList) 
#line 638 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 639
return __stdio_common_vfprintf(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 640
} 
#line 644 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vfprintf(FILE *const 
#line 645
_Stream, const char *const 
#line 646
_Format, va_list 
#line 647
_ArgList) 
#line 652 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 653
return _vfprintf_l(_Stream, _Format, 0, _ArgList); 
#line 654
} 
#line 658 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vfprintf_s_l(FILE *const 
#line 659
_Stream, const char *const 
#line 660
_Format, const _locale_t 
#line 661
_Locale, va_list 
#line 662
_ArgList) 
#line 667 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 668
return __stdio_common_vfprintf_s(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 669
} 
#line 675 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vfprintf_s(FILE *const 
#line 676
_Stream, const char *const 
#line 677
_Format, va_list 
#line 678
_ArgList) 
#line 683 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 684
return _vfprintf_s_l(_Stream, _Format, 0, _ArgList); 
#line 685
} 
#line 691 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vfprintf_p_l(FILE *const 
#line 692
_Stream, const char *const 
#line 693
_Format, const _locale_t 
#line 694
_Locale, va_list 
#line 695
_ArgList) 
#line 700 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 701
return __stdio_common_vfprintf_p(*__local_stdio_printf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 702
} 
#line 706 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vfprintf_p(FILE *const 
#line 707
_Stream, const char *const 
#line 708
_Format, va_list 
#line 709
_ArgList) 
#line 714 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 715
return _vfprintf_p_l(_Stream, _Format, 0, _ArgList); 
#line 716
} 
#line 720 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vprintf_l(const char *const 
#line 721
_Format, const _locale_t 
#line 722
_Locale, va_list 
#line 723
_ArgList) 
#line 728 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 729
return _vfprintf_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 730
} 
#line 734 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vprintf(const char *const 
#line 735
_Format, va_list 
#line 736
_ArgList) 
#line 741 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 742
return _vfprintf_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 743
} 
#line 747 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vprintf_s_l(const char *const 
#line 748
_Format, const _locale_t 
#line 749
_Locale, va_list 
#line 750
_ArgList) 
#line 755 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 756
return _vfprintf_s_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 757
} 
#line 763 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vprintf_s(const char *const 
#line 764
_Format, va_list 
#line 765
_ArgList) 
#line 770 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 771
return _vfprintf_s_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 772
} 
#line 778 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vprintf_p_l(const char *const 
#line 779
_Format, const _locale_t 
#line 780
_Locale, va_list 
#line 781
_ArgList) 
#line 786 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 787
return _vfprintf_p_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 788
} 
#line 792 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vprintf_p(const char *const 
#line 793
_Format, va_list 
#line 794
_ArgList) 
#line 799 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 800
return _vfprintf_p_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 801
} 
#line 805 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _fprintf_l(FILE *const 
#line 806
_Stream, const char *const 
#line 807
_Format, const _locale_t 
#line 808
_Locale, ...) 
#line 813 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 814
int _Result; 
#line 815
va_list _ArgList; 
#line 816
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 817
_Result = _vfprintf_l(_Stream, _Format, _Locale, _ArgList); 
#line 818
(void)(_ArgList = ((va_list)0)); 
#line 819
return _Result; 
#line 820
} 
#line 824 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl fprintf(FILE *const 
#line 825
_Stream, const char *const 
#line 826
_Format, ...) 
#line 831 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 832
int _Result; 
#line 833
va_list _ArgList; 
#line 834
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 835
_Result = _vfprintf_l(_Stream, _Format, 0, _ArgList); 
#line 836
(void)(_ArgList = ((va_list)0)); 
#line 837
return _Result; 
#line 838
} 
#line 841 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl _set_printf_count_output(int _Value); 
#line 845
__declspec(dllimport) int __cdecl _get_printf_count_output(); 
#line 848
__inline int __cdecl _fprintf_s_l(FILE *const 
#line 849
_Stream, const char *const 
#line 850
_Format, const _locale_t 
#line 851
_Locale, ...) 
#line 856 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 857
int _Result; 
#line 858
va_list _ArgList; 
#line 859
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 860
_Result = _vfprintf_s_l(_Stream, _Format, _Locale, _ArgList); 
#line 861
(void)(_ArgList = ((va_list)0)); 
#line 862
return _Result; 
#line 863
} 
#line 869 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl fprintf_s(FILE *const 
#line 870
_Stream, const char *const 
#line 871
_Format, ...) 
#line 876 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 877
int _Result; 
#line 878
va_list _ArgList; 
#line 879
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 880
_Result = _vfprintf_s_l(_Stream, _Format, 0, _ArgList); 
#line 881
(void)(_ArgList = ((va_list)0)); 
#line 882
return _Result; 
#line 883
} 
#line 889 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _fprintf_p_l(FILE *const 
#line 890
_Stream, const char *const 
#line 891
_Format, const _locale_t 
#line 892
_Locale, ...) 
#line 897 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 898
int _Result; 
#line 899
va_list _ArgList; 
#line 900
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 901
_Result = _vfprintf_p_l(_Stream, _Format, _Locale, _ArgList); 
#line 902
(void)(_ArgList = ((va_list)0)); 
#line 903
return _Result; 
#line 904
} 
#line 908 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _fprintf_p(FILE *const 
#line 909
_Stream, const char *const 
#line 910
_Format, ...) 
#line 915 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 916
int _Result; 
#line 917
va_list _ArgList; 
#line 918
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 919
_Result = _vfprintf_p_l(_Stream, _Format, 0, _ArgList); 
#line 920
(void)(_ArgList = ((va_list)0)); 
#line 921
return _Result; 
#line 922
} 
#line 926 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _printf_l(const char *const 
#line 927
_Format, const _locale_t 
#line 928
_Locale, ...) 
#line 933 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 934
int _Result; 
#line 935
va_list _ArgList; 
#line 936
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 937
_Result = _vfprintf_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 938
(void)(_ArgList = ((va_list)0)); 
#line 939
return _Result; 
#line 940
} 
#line 944 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl printf(const char *const 
#line 945
_Format, ...) 
#line 950 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 951
int _Result; 
#line 952
va_list _ArgList; 
#line 953
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 954
_Result = _vfprintf_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 955
(void)(_ArgList = ((va_list)0)); 
#line 956
return _Result; 
#line 957
} 
#line 961 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _printf_s_l(const char *const 
#line 962
_Format, const _locale_t 
#line 963
_Locale, ...) 
#line 968 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 969
int _Result; 
#line 970
va_list _ArgList; 
#line 971
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 972
_Result = _vfprintf_s_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 973
(void)(_ArgList = ((va_list)0)); 
#line 974
return _Result; 
#line 975
} 
#line 981 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl printf_s(const char *const 
#line 982
_Format, ...) 
#line 987 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 988
int _Result; 
#line 989
va_list _ArgList; 
#line 990
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 991
_Result = _vfprintf_s_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 992
(void)(_ArgList = ((va_list)0)); 
#line 993
return _Result; 
#line 994
} 
#line 1000 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _printf_p_l(const char *const 
#line 1001
_Format, const _locale_t 
#line 1002
_Locale, ...) 
#line 1007 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1008
int _Result; 
#line 1009
va_list _ArgList; 
#line 1010
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1011
_Result = _vfprintf_p_l(__acrt_iob_func(1), _Format, _Locale, _ArgList); 
#line 1012
(void)(_ArgList = ((va_list)0)); 
#line 1013
return _Result; 
#line 1014
} 
#line 1018 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _printf_p(const char *const 
#line 1019
_Format, ...) 
#line 1024 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1025
int _Result; 
#line 1026
va_list _ArgList; 
#line 1027
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1028
_Result = _vfprintf_p_l(__acrt_iob_func(1), _Format, 0, _ArgList); 
#line 1029
(void)(_ArgList = ((va_list)0)); 
#line 1030
return _Result; 
#line 1031
} 
#line 1040 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vfscanf(unsigned __int64 _Options, FILE * _Stream, const char * _Format, _locale_t _Locale, va_list _Arglist); 
#line 1049
__inline int __cdecl _vfscanf_l(FILE *const 
#line 1050
_Stream, const char *const 
#line 1051
_Format, const _locale_t 
#line 1052
_Locale, va_list 
#line 1053
_ArgList) 
#line 1058 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1059
return __stdio_common_vfscanf(*__local_stdio_scanf_options(), _Stream, _Format, _Locale, _ArgList); 
#line 1062
} 
#line 1066 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vfscanf(FILE *const 
#line 1067
_Stream, const char *const 
#line 1068
_Format, va_list 
#line 1069
_ArgList) 
#line 1074 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1075
return _vfscanf_l(_Stream, _Format, 0, _ArgList); 
#line 1076
} 
#line 1080 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vfscanf_s_l(FILE *const 
#line 1081
_Stream, const char *const 
#line 1082
_Format, const _locale_t 
#line 1083
_Locale, va_list 
#line 1084
_ArgList) 
#line 1089 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1090
return __stdio_common_vfscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Stream, _Format, _Locale, _ArgList); 
#line 1093
} 
#line 1100 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vfscanf_s(FILE *const 
#line 1101
_Stream, const char *const 
#line 1102
_Format, va_list 
#line 1103
_ArgList) 
#line 1108 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1109
return _vfscanf_s_l(_Stream, _Format, 0, _ArgList); 
#line 1110
} 
#line 1116 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscanf_l(const char *const 
#line 1117
_Format, const _locale_t 
#line 1118
_Locale, va_list 
#line 1119
_ArgList) 
#line 1124 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1125
return _vfscanf_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 1126
} 
#line 1130 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vscanf(const char *const 
#line 1131
_Format, va_list 
#line 1132
_ArgList) 
#line 1137 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1138
return _vfscanf_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 1139
} 
#line 1143 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscanf_s_l(const char *const 
#line 1144
_Format, const _locale_t 
#line 1145
_Locale, va_list 
#line 1146
_ArgList) 
#line 1151 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1152
return _vfscanf_s_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 1153
} 
#line 1159 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vscanf_s(const char *const 
#line 1160
_Format, va_list 
#line 1161
_ArgList) 
#line 1166 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1167
return _vfscanf_s_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 1168
} 
#line 1174 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _fscanf_l(FILE *const 
#line 1175
_Stream, const char *const 
#line 1176
_Format, const _locale_t 
#line 1177
_Locale, ...) 
#line 1182 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1183
int _Result; 
#line 1184
va_list _ArgList; 
#line 1185
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1186
_Result = _vfscanf_l(_Stream, _Format, _Locale, _ArgList); 
#line 1187
(void)(_ArgList = ((va_list)0)); 
#line 1188
return _Result; 
#line 1189
} 
#line 1193 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl fscanf(FILE *const 
#line 1194
_Stream, const char *const 
#line 1195
_Format, ...) 
#line 1200 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1201
int _Result; 
#line 1202
va_list _ArgList; 
#line 1203
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1204
_Result = _vfscanf_l(_Stream, _Format, 0, _ArgList); 
#line 1205
(void)(_ArgList = ((va_list)0)); 
#line 1206
return _Result; 
#line 1207
} 
#line 1211 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _fscanf_s_l(FILE *const 
#line 1212
_Stream, const char *const 
#line 1213
_Format, const _locale_t 
#line 1214
_Locale, ...) 
#line 1219 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1220
int _Result; 
#line 1221
va_list _ArgList; 
#line 1222
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1223
_Result = _vfscanf_s_l(_Stream, _Format, _Locale, _ArgList); 
#line 1224
(void)(_ArgList = ((va_list)0)); 
#line 1225
return _Result; 
#line 1226
} 
#line 1232 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl fscanf_s(FILE *const 
#line 1233
_Stream, const char *const 
#line 1234
_Format, ...) 
#line 1239 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1240
int _Result; 
#line 1241
va_list _ArgList; 
#line 1242
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1243
_Result = _vfscanf_s_l(_Stream, _Format, 0, _ArgList); 
#line 1244
(void)(_ArgList = ((va_list)0)); 
#line 1245
return _Result; 
#line 1246
} 
#line 1252 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scanf_l(const char *const 
#line 1253
_Format, const _locale_t 
#line 1254
_Locale, ...) 
#line 1259 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1260
int _Result; 
#line 1261
va_list _ArgList; 
#line 1262
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1263
_Result = _vfscanf_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 1264
(void)(_ArgList = ((va_list)0)); 
#line 1265
return _Result; 
#line 1266
} 
#line 1270 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl scanf(const char *const 
#line 1271
_Format, ...) 
#line 1276 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1277
int _Result; 
#line 1278
va_list _ArgList; 
#line 1279
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1280
_Result = _vfscanf_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 1281
(void)(_ArgList = ((va_list)0)); 
#line 1282
return _Result; 
#line 1283
} 
#line 1287 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scanf_s_l(const char *const 
#line 1288
_Format, const _locale_t 
#line 1289
_Locale, ...) 
#line 1294 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1295
int _Result; 
#line 1296
va_list _ArgList; 
#line 1297
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1298
_Result = _vfscanf_s_l(__acrt_iob_func(0), _Format, _Locale, _ArgList); 
#line 1299
(void)(_ArgList = ((va_list)0)); 
#line 1300
return _Result; 
#line 1301
} 
#line 1307 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl scanf_s(const char *const 
#line 1308
_Format, ...) 
#line 1313 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1314
int _Result; 
#line 1315
va_list _ArgList; 
#line 1316
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1317
_Result = _vfscanf_s_l(__acrt_iob_func(0), _Format, 0, _ArgList); 
#line 1318
(void)(_ArgList = ((va_list)0)); 
#line 1319
return _Result; 
#line 1320
} 
#line 1333 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vsprintf(unsigned __int64 _Options, char * _Buffer, size_t _BufferCount, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1343
__declspec(dllimport) int __cdecl __stdio_common_vsprintf_s(unsigned __int64 _Options, char * _Buffer, size_t _BufferCount, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1353
__declspec(dllimport) int __cdecl __stdio_common_vsnprintf_s(unsigned __int64 _Options, char * _Buffer, size_t _BufferCount, size_t _MaxCount, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1364
__declspec(dllimport) int __cdecl __stdio_common_vsprintf_p(unsigned __int64 _Options, char * _Buffer, size_t _BufferCount, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 1375
__inline int __cdecl _vsnprintf_l(char *const 
#line 1376
_Buffer, const size_t 
#line 1377
_BufferCount, const char *const 
#line 1378
_Format, const _locale_t 
#line 1379
_Locale, va_list 
#line 1380
_ArgList) 
#line 1385 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1386
const int _Result = __stdio_common_vsprintf((*__local_stdio_printf_options()) | (1Ui64 << 0), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1390
return (_Result < 0) ? -1 : _Result; 
#line 1391
} 
#line 1396 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsnprintf(char *const 
#line 1397
_Buffer, const size_t 
#line 1398
_BufferCount, const char *const 
#line 1399
_Format, va_list 
#line 1400
_ArgList) 
#line 1405 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _vsnprintf_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#pragma warning(pop)
} 
#line 1426 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vsnprintf(char *const 
#line 1427
_Buffer, const size_t 
#line 1428
_BufferCount, const char *const 
#line 1429
_Format, va_list 
#line 1430
_ArgList) 
#line 1435 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1436
const int _Result = __stdio_common_vsprintf((*__local_stdio_printf_options()) | (1Ui64 << 1), _Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1440
return (_Result < 0) ? -1 : _Result; 
#line 1441
} 
#line 1446 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsprintf_l(char *const 
#line 1447
_Buffer, const char *const 
#line 1448
_Format, const _locale_t 
#line 1449
_Locale, va_list 
#line 1450
_ArgList) 
#line 1455 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _vsnprintf_l(_Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#pragma warning(pop)
} 
#line 1465 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vsprintf(char *const 
#line 1466
_Buffer, const char *const 
#line 1467
_Format, va_list 
#line 1468
_ArgList) 
#line 1473 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _vsnprintf_l(_Buffer, (size_t)(-1), _Format, 0, _ArgList); 
#pragma warning(pop)
} 
#line 1483 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsprintf_s_l(char *const 
#line 1484
_Buffer, const size_t 
#line 1485
_BufferCount, const char *const 
#line 1486
_Format, const _locale_t 
#line 1487
_Locale, va_list 
#line 1488
_ArgList) 
#line 1493 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1494
const int _Result = __stdio_common_vsprintf_s(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1498
return (_Result < 0) ? -1 : _Result; 
#line 1499
} 
#line 1506 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vsprintf_s(char *const 
#line 1507
_Buffer, const size_t 
#line 1508
_BufferCount, const char *const 
#line 1509
_Format, va_list 
#line 1510
_ArgList) 
#line 1515 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1516
return _vsprintf_s_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1517
} 
#line 1520 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl vsprintf_s(char (&_Buffer)[_Size], const char *_Format, ::va_list _ArgList) throw() { return vsprintf_s(_Buffer, _Size, _Format, _ArgList); } }
#line 1532 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsprintf_p_l(char *const 
#line 1533
_Buffer, const size_t 
#line 1534
_BufferCount, const char *const 
#line 1535
_Format, const _locale_t 
#line 1536
_Locale, va_list 
#line 1537
_ArgList) 
#line 1542 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1543
const int _Result = __stdio_common_vsprintf_p(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1547
return (_Result < 0) ? -1 : _Result; 
#line 1548
} 
#line 1553 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsprintf_p(char *const 
#line 1554
_Buffer, const size_t 
#line 1555
_BufferCount, const char *const 
#line 1556
_Format, va_list 
#line 1557
_ArgList) 
#line 1562 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1563
return _vsprintf_p_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1564
} 
#line 1569 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsnprintf_s_l(char *const 
#line 1570
_Buffer, const size_t 
#line 1571
_BufferCount, const size_t 
#line 1572
_MaxCount, const char *const 
#line 1573
_Format, const _locale_t 
#line 1574
_Locale, va_list 
#line 1575
_ArgList) 
#line 1580 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1581
const int _Result = __stdio_common_vsnprintf_s(*__local_stdio_printf_options(), _Buffer, _BufferCount, _MaxCount, _Format, _Locale, _ArgList); 
#line 1585
return (_Result < 0) ? -1 : _Result; 
#line 1586
} 
#line 1591 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsnprintf_s(char *const 
#line 1592
_Buffer, const size_t 
#line 1593
_BufferCount, const size_t 
#line 1594
_MaxCount, const char *const 
#line 1595
_Format, va_list 
#line 1596
_ArgList) 
#line 1601 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1602
return _vsnprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, 0, _ArgList); 
#line 1603
} 
#line 1606 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl _vsnprintf_s(char (&_Buffer)[_Size], ::size_t _BufferCount, const char *_Format, ::va_list _ArgList) throw() { return _vsnprintf_s(_Buffer, _Size, _BufferCount, _Format, _ArgList); } }
#line 1619 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vsnprintf_s(char *const 
#line 1620
_Buffer, const size_t 
#line 1621
_BufferCount, const size_t 
#line 1622
_MaxCount, const char *const 
#line 1623
_Format, va_list 
#line 1624
_ArgList) 
#line 1629 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1630
return _vsnprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, 0, _ArgList); 
#line 1631
} 
#line 1634 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl vsnprintf_s(char (&_Buffer)[_Size], ::size_t _BufferCount, const char *_Format, ::va_list _ArgList) throw() { return vsnprintf_s(_Buffer, _Size, _BufferCount, _Format, _ArgList); } }
#line 1646 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscprintf_l(const char *const 
#line 1647
_Format, const _locale_t 
#line 1648
_Locale, va_list 
#line 1649
_ArgList) 
#line 1654 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1655
const int _Result = __stdio_common_vsprintf((*__local_stdio_printf_options()) | (1Ui64 << 1), 0, 0, _Format, _Locale, _ArgList); 
#line 1659
return (_Result < 0) ? -1 : _Result; 
#line 1660
} 
#line 1664 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscprintf(const char *const 
#line 1665
_Format, va_list 
#line 1666
_ArgList) 
#line 1671 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1672
return _vscprintf_l(_Format, 0, _ArgList); 
#line 1673
} 
#line 1677 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscprintf_p_l(const char *const 
#line 1678
_Format, const _locale_t 
#line 1679
_Locale, va_list 
#line 1680
_ArgList) 
#line 1685 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1686
const int _Result = __stdio_common_vsprintf_p((*__local_stdio_printf_options()) | (1Ui64 << 1), 0, 0, _Format, _Locale, _ArgList); 
#line 1690
return (_Result < 0) ? -1 : _Result; 
#line 1691
} 
#line 1695 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vscprintf_p(const char *const 
#line 1696
_Format, va_list 
#line 1697
_ArgList) 
#line 1702 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1703
return _vscprintf_p_l(_Format, 0, _ArgList); 
#line 1704
} 
#line 1708 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsnprintf_c_l(char *const 
#line 1709
_Buffer, const size_t 
#line 1710
_BufferCount, const char *const 
#line 1711
_Format, const _locale_t 
#line 1712
_Locale, va_list 
#line 1713
_ArgList) 
#line 1718 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1719
const int _Result = __stdio_common_vsprintf(*__local_stdio_printf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1723
return (_Result < 0) ? -1 : _Result; 
#line 1724
} 
#line 1729 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsnprintf_c(char *const 
#line 1730
_Buffer, const size_t 
#line 1731
_BufferCount, const char *const 
#line 1732
_Format, va_list 
#line 1733
_ArgList) 
#line 1738 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1739
return _vsnprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1740
} 
#line 1744 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _sprintf_l(char *const 
#line 1745
_Buffer, const char *const 
#line 1746
_Format, const _locale_t 
#line 1747
_Locale, ...) 
#line 1752 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1753
int _Result; 
#line 1754
va_list _ArgList; 
#line 1755
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1757
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsprintf_l(_Buffer, _Format, _Locale, _ArgList); 
#pragma warning(pop)
#line 1762
(void)(_ArgList = ((va_list)0)); 
#line 1763
return _Result; 
#line 1764
} 
#line 1769 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl sprintf(char *const 
#line 1770
_Buffer, const char *const 
#line 1771
_Format, ...) 
#line 1776 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1777
int _Result; 
#line 1778
va_list _ArgList; 
#line 1779
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1781
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsprintf_l(_Buffer, _Format, 0, _ArgList); 
#pragma warning(pop)
#line 1786
(void)(_ArgList = ((va_list)0)); 
#line 1787
return _Result; 
#line 1788
} 
#line 1791 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(push)
#pragma warning(disable: 4996)
__inline int __cdecl sprintf(char * _Buffer, const char * _Format, ...); __inline int __cdecl vsprintf(char * _Buffer, const char * _Format, va_list _Args); 
#line 1799 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(pop)
#line 1803
__inline int __cdecl _sprintf_s_l(char *const 
#line 1804
_Buffer, const size_t 
#line 1805
_BufferCount, const char *const 
#line 1806
_Format, const _locale_t 
#line 1807
_Locale, ...) 
#line 1812 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1813
int _Result; 
#line 1814
va_list _ArgList; 
#line 1815
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1816
_Result = _vsprintf_s_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1817
(void)(_ArgList = ((va_list)0)); 
#line 1818
return _Result; 
#line 1819
} 
#line 1826 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl sprintf_s(char *const 
#line 1827
_Buffer, const size_t 
#line 1828
_BufferCount, const char *const 
#line 1829
_Format, ...) 
#line 1834 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1835
int _Result; 
#line 1836
va_list _ArgList; 
#line 1837
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1838
_Result = _vsprintf_s_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1839
(void)(_ArgList = ((va_list)0)); 
#line 1840
return _Result; 
#line 1841
} 
#line 1846 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {__pragma( warning(push)) __pragma( warning(disable: 4793)) template< size_t _Size> inline int __cdecl sprintf_s(char (&_Buffer)[_Size], const char *_Format, ...) throw() { ::va_list _ArgList; (void)(__vcrt_va_start_verify_argument_type< const char *> (), ((void)__va_start(&_ArgList, _Format))); return vsprintf_s(_Buffer, _Size, _Format, _ArgList); } __pragma( warning(pop)) }
#line 1854 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _sprintf_p_l(char *const 
#line 1855
_Buffer, const size_t 
#line 1856
_BufferCount, const char *const 
#line 1857
_Format, const _locale_t 
#line 1858
_Locale, ...) 
#line 1863 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1864
int _Result; 
#line 1865
va_list _ArgList; 
#line 1866
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1867
_Result = _vsprintf_p_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1868
(void)(_ArgList = ((va_list)0)); 
#line 1869
return _Result; 
#line 1870
} 
#line 1875 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _sprintf_p(char *const 
#line 1876
_Buffer, const size_t 
#line 1877
_BufferCount, const char *const 
#line 1878
_Format, ...) 
#line 1883 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1884
int _Result; 
#line 1885
va_list _ArgList; 
#line 1886
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 1887
_Result = _vsprintf_p_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 1888
(void)(_ArgList = ((va_list)0)); 
#line 1889
return _Result; 
#line 1890
} 
#line 1895 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf_l(char *const 
#line 1896
_Buffer, const size_t 
#line 1897
_BufferCount, const char *const 
#line 1898
_Format, const _locale_t 
#line 1899
_Locale, ...) 
#line 1904 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1905
int _Result; 
#line 1906
va_list _ArgList; 
#line 1907
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1909
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = _vsnprintf_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#pragma warning(pop)
#line 1914
(void)(_ArgList = ((va_list)0)); 
#line 1915
return _Result; 
#line 1916
} 
#line 1932 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl snprintf(char *const 
#line 1933
_Buffer, const size_t 
#line 1934
_BufferCount, const char *const 
#line 1935
_Format, ...) 
#line 1940 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1941
int _Result; 
#line 1942
va_list _ArgList; 
#line 1943
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#pragma warning(suppress:28719)
_Result = vsnprintf(_Buffer, _BufferCount, _Format, _ArgList); 
#line 1946
(void)(_ArgList = ((va_list)0)); 
#line 1947
return _Result; 
#line 1948
} 
#line 1952 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf(char *const 
#line 1953
_Buffer, const size_t 
#line 1954
_BufferCount, const char *const 
#line 1955
_Format, ...) 
#line 1960 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1961
int _Result; 
#line 1962
va_list _ArgList; 
#line 1963
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#pragma warning(suppress:28719)
_Result = _vsnprintf(_Buffer, _BufferCount, _Format, _ArgList); 
#line 1966
(void)(_ArgList = ((va_list)0)); 
#line 1967
return _Result; 
#line 1968
} 
#line 1971 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf(char * _Buffer, size_t _BufferCount, const char * _Format, ...); __inline int __cdecl _vsnprintf(char * _Buffer, size_t _BufferCount, const char * _Format, va_list _Args); 
#line 1982 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf_c_l(char *const 
#line 1983
_Buffer, const size_t 
#line 1984
_BufferCount, const char *const 
#line 1985
_Format, const _locale_t 
#line 1986
_Locale, ...) 
#line 1991 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 1992
int _Result; 
#line 1993
va_list _ArgList; 
#line 1994
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 1995
_Result = _vsnprintf_c_l(_Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 1996
(void)(_ArgList = ((va_list)0)); 
#line 1997
return _Result; 
#line 1998
} 
#line 2003 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf_c(char *const 
#line 2004
_Buffer, const size_t 
#line 2005
_BufferCount, const char *const 
#line 2006
_Format, ...) 
#line 2011 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2012
int _Result; 
#line 2013
va_list _ArgList; 
#line 2014
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2015
_Result = _vsnprintf_c_l(_Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 2016
(void)(_ArgList = ((va_list)0)); 
#line 2017
return _Result; 
#line 2018
} 
#line 2023 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf_s_l(char *const 
#line 2024
_Buffer, const size_t 
#line 2025
_BufferCount, const size_t 
#line 2026
_MaxCount, const char *const 
#line 2027
_Format, const _locale_t 
#line 2028
_Locale, ...) 
#line 2033 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2034
int _Result; 
#line 2035
va_list _ArgList; 
#line 2036
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2037
_Result = _vsnprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, _Locale, _ArgList); 
#line 2038
(void)(_ArgList = ((va_list)0)); 
#line 2039
return _Result; 
#line 2040
} 
#line 2045 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snprintf_s(char *const 
#line 2046
_Buffer, const size_t 
#line 2047
_BufferCount, const size_t 
#line 2048
_MaxCount, const char *const 
#line 2049
_Format, ...) 
#line 2054 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2055
int _Result; 
#line 2056
va_list _ArgList; 
#line 2057
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2058
_Result = _vsnprintf_s_l(_Buffer, _BufferCount, _MaxCount, _Format, 0, _ArgList); 
#line 2059
(void)(_ArgList = ((va_list)0)); 
#line 2060
return _Result; 
#line 2061
} 
#line 2064 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {__pragma( warning(push)) __pragma( warning(disable: 4793)) template< size_t _Size> inline int __cdecl _snprintf_s(char (&_Buffer)[_Size], ::size_t _BufferCount, const char *_Format, ...) throw() { ::va_list _ArgList; (void)(__vcrt_va_start_verify_argument_type< const char *> (), ((void)__va_start(&_ArgList, _Format))); return _vsnprintf_s(_Buffer, _Size, _BufferCount, _Format, _ArgList); } __pragma( warning(pop)) }
#line 2073 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scprintf_l(const char *const 
#line 2074
_Format, const _locale_t 
#line 2075
_Locale, ...) 
#line 2080 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2081
int _Result; 
#line 2082
va_list _ArgList; 
#line 2083
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2084
_Result = _vscprintf_l(_Format, _Locale, _ArgList); 
#line 2085
(void)(_ArgList = ((va_list)0)); 
#line 2086
return _Result; 
#line 2087
} 
#line 2091 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scprintf(const char *const 
#line 2092
_Format, ...) 
#line 2097 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2098
int _Result; 
#line 2099
va_list _ArgList; 
#line 2100
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2101
_Result = _vscprintf_l(_Format, 0, _ArgList); 
#line 2102
(void)(_ArgList = ((va_list)0)); 
#line 2103
return _Result; 
#line 2104
} 
#line 2108 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scprintf_p_l(const char *const 
#line 2109
_Format, const _locale_t 
#line 2110
_Locale, ...) 
#line 2115 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2116
int _Result; 
#line 2117
va_list _ArgList; 
#line 2118
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2119
_Result = _vscprintf_p_l(_Format, _Locale, _ArgList); 
#line 2120
(void)(_ArgList = ((va_list)0)); 
#line 2121
return _Result; 
#line 2122
} 
#line 2126 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _scprintf_p(const char *const 
#line 2127
_Format, ...) 
#line 2132 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2133
int _Result; 
#line 2134
va_list _ArgList; 
#line 2135
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2136
_Result = _vscprintf_p(_Format, _ArgList); 
#line 2137
(void)(_ArgList = ((va_list)0)); 
#line 2138
return _Result; 
#line 2139
} 
#line 2147 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl __stdio_common_vsscanf(unsigned __int64 _Options, const char * _Buffer, size_t _BufferCount, const char * _Format, _locale_t _Locale, va_list _ArgList); 
#line 2157
__inline int __cdecl _vsscanf_l(const char *const 
#line 2158
_Buffer, const char *const 
#line 2159
_Format, const _locale_t 
#line 2160
_Locale, va_list 
#line 2161
_ArgList) 
#line 2166 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2167
return __stdio_common_vsscanf(*__local_stdio_scanf_options(), _Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 2170
} 
#line 2174 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl vsscanf(const char *const 
#line 2175
_Buffer, const char *const 
#line 2176
_Format, va_list 
#line 2177
_ArgList) 
#line 2182 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2183
return _vsscanf_l(_Buffer, _Format, 0, _ArgList); 
#line 2184
} 
#line 2188 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _vsscanf_s_l(const char *const 
#line 2189
_Buffer, const char *const 
#line 2190
_Format, const _locale_t 
#line 2191
_Locale, va_list 
#line 2192
_ArgList) 
#line 2197 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2198
return __stdio_common_vsscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Buffer, (size_t)(-1), _Format, _Locale, _ArgList); 
#line 2201
} 
#line 2206 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(push)
#pragma warning(disable:6530)
#line 2210
__inline int __cdecl vsscanf_s(const char *const 
#line 2211
_Buffer, const char *const 
#line 2212
_Format, va_list 
#line 2213
_ArgList) 
#line 2218 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2219
return _vsscanf_s_l(_Buffer, _Format, 0, _ArgList); 
#line 2220
} 
#line 2223 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
extern "C++" {template< size_t _Size> inline int __cdecl vsscanf_s(const char (&_Buffer)[_Size], const char *_Format, ::va_list _ArgList) throw() { return vsscanf_s(_Buffer, _Size, _Format, _ArgList); } }
#line 2230 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(pop)
#line 2235 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _sscanf_l(const char *const 
#line 2236
_Buffer, const char *const 
#line 2237
_Format, const _locale_t 
#line 2238
_Locale, ...) 
#line 2243 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2244
int _Result; 
#line 2245
va_list _ArgList; 
#line 2246
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2247
_Result = _vsscanf_l(_Buffer, _Format, _Locale, _ArgList); 
#line 2248
(void)(_ArgList = ((va_list)0)); 
#line 2249
return _Result; 
#line 2250
} 
#line 2254 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl sscanf(const char *const 
#line 2255
_Buffer, const char *const 
#line 2256
_Format, ...) 
#line 2261 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2262
int _Result; 
#line 2263
va_list _ArgList; 
#line 2264
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2265
_Result = _vsscanf_l(_Buffer, _Format, 0, _ArgList); 
#line 2266
(void)(_ArgList = ((va_list)0)); 
#line 2267
return _Result; 
#line 2268
} 
#line 2272 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _sscanf_s_l(const char *const 
#line 2273
_Buffer, const char *const 
#line 2274
_Format, const _locale_t 
#line 2275
_Locale, ...) 
#line 2280 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2281
int _Result; 
#line 2282
va_list _ArgList; 
#line 2283
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2284
_Result = _vsscanf_s_l(_Buffer, _Format, _Locale, _ArgList); 
#line 2285
(void)(_ArgList = ((va_list)0)); 
#line 2286
return _Result; 
#line 2287
} 
#line 2293 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl sscanf_s(const char *const 
#line 2294
_Buffer, const char *const 
#line 2295
_Format, ...) 
#line 2300 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2301
int _Result; 
#line 2302
va_list _ArgList; 
#line 2303
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2305
#pragma warning(push)
#pragma warning(disable: 4996)
_Result = vsscanf_s(_Buffer, _Format, _ArgList); 
#pragma warning(pop)
#line 2310
(void)(_ArgList = ((va_list)0)); 
#line 2311
return _Result; 
#line 2312
} 
#line 2317 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(push)
#pragma warning(disable:6530)
#line 2321
__inline int __cdecl _snscanf_l(const char *const 
#line 2322
_Buffer, const size_t 
#line 2323
_BufferCount, const char *const 
#line 2324
_Format, const _locale_t 
#line 2325
_Locale, ...) 
#line 2330 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2331
int _Result; 
#line 2332
va_list _ArgList; 
#line 2333
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2335
_Result = __stdio_common_vsscanf(*__local_stdio_scanf_options(), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 2339
(void)(_ArgList = ((va_list)0)); 
#line 2340
return _Result; 
#line 2341
} 
#line 2345 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snscanf(const char *const 
#line 2346
_Buffer, const size_t 
#line 2347
_BufferCount, const char *const 
#line 2348
_Format, ...) 
#line 2353 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2354
int _Result; 
#line 2355
va_list _ArgList; 
#line 2356
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2358
_Result = __stdio_common_vsscanf(*__local_stdio_scanf_options(), _Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 2362
(void)(_ArgList = ((va_list)0)); 
#line 2363
return _Result; 
#line 2364
} 
#line 2369 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snscanf_s_l(const char *const 
#line 2370
_Buffer, const size_t 
#line 2371
_BufferCount, const char *const 
#line 2372
_Format, const _locale_t 
#line 2373
_Locale, ...) 
#line 2378 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2379
int _Result; 
#line 2380
va_list _ArgList; 
#line 2381
(void)(__vcrt_va_start_verify_argument_type< __crt_locale_pointers *const> (), ((void)__va_start(&_ArgList, _Locale))); 
#line 2383
_Result = __stdio_common_vsscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Buffer, _BufferCount, _Format, _Locale, _ArgList); 
#line 2387
(void)(_ArgList = ((va_list)0)); 
#line 2388
return _Result; 
#line 2389
} 
#line 2393 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__inline int __cdecl _snscanf_s(const char *const 
#line 2394
_Buffer, const size_t 
#line 2395
_BufferCount, const char *const 
#line 2396
_Format, ...) 
#line 2401 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
{ 
#line 2402
int _Result; 
#line 2403
va_list _ArgList; 
#line 2404
(void)(__vcrt_va_start_verify_argument_type< const char *const> (), ((void)__va_start(&_ArgList, _Format))); 
#line 2406
_Result = __stdio_common_vsscanf((*__local_stdio_scanf_options()) | (1Ui64 << 0), _Buffer, _BufferCount, _Format, 0, _ArgList); 
#line 2410
(void)(_ArgList = ((va_list)0)); 
#line 2411
return _Result; 
#line 2412
} 
#line 2415 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
#pragma warning(pop)
#line 2438 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) char *__cdecl tempnam(const char * _Directory, const char * _FilePrefix); 
#line 2447 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
__declspec(dllimport) int __cdecl fcloseall(); 
#line 2448
__declspec(dllimport) FILE *__cdecl fdopen(int _FileHandle, const char * _Format); 
#line 2449
__declspec(dllimport) int __cdecl fgetchar(); 
#line 2450
__declspec(dllimport) int __cdecl fileno(FILE * _Stream); 
#line 2451
__declspec(dllimport) int __cdecl flushall(); 
#line 2452
__declspec(dllimport) int __cdecl fputchar(int _Ch); 
#line 2453
__declspec(dllimport) int __cdecl getw(FILE * _Stream); 
#line 2454
__declspec(dllimport) int __cdecl putw(int _Ch, FILE * _Stream); 
#line 2455
__declspec(dllimport) int __cdecl rmtmp(); 
#line 2461 "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt\\stdio.h"
}__pragma( pack ( pop )) 
#line 7 "C:/Users/Plainer/Documents/GitHub/Sigma/Sigma.Kernels/sigmakernels.cu"
int main() 
#line 8
{ 
#line 9
return 0; 
#line 10
} 
#line 1 "sigmakernels.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__20_sigmakernels_cpp1_ii_main
#line 1 "sigmakernels.cudafe1.stub.c"
#include "sigmakernels.cudafe1.stub.c"
#line 1 "sigmakernels.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
