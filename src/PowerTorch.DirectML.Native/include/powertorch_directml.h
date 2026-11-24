#pragma once

#include <cstdint>

#if defined(_WIN32)
#define PT_DML_EXPORT extern "C" __declspec(dllexport)
#else
#define PT_DML_EXPORT extern "C"
#endif

#define PT_DML_ABI_MAJOR 1u
#define PT_DML_ABI_MINOR 0u

#define PT_DML_OK 0
#define PT_DML_INVALID_ARGUMENT 1
#define PT_DML_UNSUPPORTED 2
#define PT_DML_ALLOCATION 3
#define PT_DML_SHAPE_MISMATCH 4
#define PT_DML_COMPILATION 5
#define PT_DML_DRIVER 6
#define PT_DML_SYNC 7

struct PtDmlDevice;

struct PtDmlAbiInfo
{
    std::uint32_t size;
    std::uint16_t major;
    std::uint16_t minor;
    std::uint32_t struct_version;
};

PT_DML_EXPORT void pt_dml_get_abi_info(PtDmlAbiInfo* info);
PT_DML_EXPORT std::int32_t pt_dml_adapter_count();
PT_DML_EXPORT std::int32_t pt_dml_adapter_name(
    std::int32_t adapter_index,
    char* output,
    std::uint32_t output_bytes);
PT_DML_EXPORT std::int32_t pt_dml_create_device(
    std::int32_t adapter_index,
    PtDmlDevice** device,
    char* error,
    std::uint32_t error_bytes);
PT_DML_EXPORT void pt_dml_release_device(PtDmlDevice* device);
PT_DML_EXPORT std::int32_t pt_dml_self_test(
    PtDmlDevice* device,
    char* error,
    std::uint32_t error_bytes);
