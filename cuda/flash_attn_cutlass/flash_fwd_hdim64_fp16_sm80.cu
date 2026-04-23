/*
 * Instantiation: forward, hdim=64, fp16, non-causal, SM80 (A100)
 *
 * Just pulls in the launch template; the actual instantiation happens via the
 * inline functions in the header. This .cu file exists to give nvcc something
 * to compile per-config so distutils/ninja's dependency tracking works cleanly.
 */
#include "flash_fwd_launch_template.h"
