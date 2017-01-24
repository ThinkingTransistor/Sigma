#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "sigmakernels.fatbin.c"
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_20_sigmakernels_cpp1_ii_main(void);
#pragma section(".CRT$XCU",read)
__declspec(allocate(".CRT$XCU"))static void (*__dummy_static_init__sti____cudaRegisterAll_20_sigmakernels_cpp1_ii_main[])(void) = {__sti____cudaRegisterAll_20_sigmakernels_cpp1_ii_main};
static void __nv_cudaEntityRegisterCallback(void **__T20){__nv_dummy_param_ref(__T20);__nv_save_fatbinhandle_for_managed_rt(__T20);}
static void __sti____cudaRegisterAll_20_sigmakernels_cpp1_ii_main(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}
