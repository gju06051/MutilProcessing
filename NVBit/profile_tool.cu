#include "nvbitfi.h"
#include <stdio.h>
#include <stdlib.h>

#define PROFILE_LOG_FILE "profile_log.txt"

void profileLogic(nvbit_exec_info_t *execInfo) {
    int l1_access = nvbitfi_l1_accesses(execInfo);
    int l2_access = nvbitfi_l2_accesses(execInfo);
    int gpu_mem_access = nvbitfi_memory_accesses(execInfo, GPU_MEMORY);

    FILE *logFile = fopen(PROFILE_LOG_FILE, "a");
    if (logFile) {
        fprintf(logFile, "Function: %s\n", execInfo->func_name);
        fprintf(logFile, "L1 Cache Accesses: %d\n", l1_access);
        fprintf(logFile, "L2 Cache Accesses: %d\n", l2_access);
        fprintf(logFile, "GPU Main Memory Accesses: %d\n", gpu_mem_access);
        fclose(logFile);
    }
}

int main(int argc, char **argv) {
    nvbitfi_init();
    nvbitfi_register_profile_kernel_start(profileLogic);

    int cuda_argc = argc - 1;
    char **cuda_argv = &argv[1];

    cuda_main(cuda_argc, cuda_argv);

    nvbitfi_unregister_profile_kernel_start();
    nvbitfi_finalize();

    return 0;
}
