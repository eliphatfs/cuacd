// Device-side data structures shared across kernel modules.
// Included by common.cuh (which then exposes these to all kernel modules).
// Must stay in sync with csrc/structs.h (host-side mirrors).
#pragma once

struct DevicePool {
    char*               base;
    unsigned long long* offset;
    unsigned long long  capacity;
};
