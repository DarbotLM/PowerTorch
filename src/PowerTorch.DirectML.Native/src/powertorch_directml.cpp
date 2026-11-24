#include "powertorch_directml.h"

#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <string>

using Microsoft::WRL::ComPtr;

struct PtDmlDevice
{
    ComPtr<IDXGIAdapter1> Adapter;
    ComPtr<ID3D12Device> D3D12Device;
    ComPtr<ID3D12CommandQueue> CommandQueue;
    ComPtr<IDMLDevice> DmlDevice;
    std::wstring AdapterName;
};

namespace
{
void WriteError(char* output, std::uint32_t bytes, std::string_view message)
{
    if (output == nullptr || bytes == 0)
    {
        return;
    }
    const auto count = static_cast<std::size_t>(std::min<std::uint32_t>(
        bytes - 1, static_cast<std::uint32_t>(message.size())));
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

HRESULT CreateFactory(ComPtr<IDXGIFactory6>& factory)
{
    return CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
}

bool FindHardwareAdapter(std::int32_t index, ComPtr<IDXGIAdapter1>& adapter)
{
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateFactory(factory)))
    {
        return false;
    }

    std::int32_t hardwareIndex = 0;
    for (UINT i = 0;; ++i)
    {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory->EnumAdapterByGpuPreference(
                i,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
        {
            return false;
        }

        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(candidate->GetDesc1(&description)))
        {
            continue;
        }
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
        {
            continue;
        }
        if (hardwareIndex++ == index)
        {
            adapter = candidate;
            return true;
        }
    }
}

std::string HresultMessage(HRESULT result, std::string_view action)
{
    return std::format("{} failed: HRESULT 0x{:08X}", action, static_cast<unsigned long>(result));
}
}

PT_DML_EXPORT void pt_dml_get_abi_info(PtDmlAbiInfo* info)
{
    if (info == nullptr)
    {
        return;
    }
    *info = {
        sizeof(PtDmlAbiInfo),
        PT_DML_ABI_MAJOR,
        PT_DML_ABI_MINOR,
        1
    };
}

PT_DML_EXPORT std::int32_t pt_dml_adapter_count()
{
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateFactory(factory)))
    {
        return -PT_DML_DRIVER;
    }

    std::int32_t count = 0;
    for (UINT i = 0;; ++i)
    {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT next = factory->EnumAdapterByGpuPreference(
            i,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter));
        if (next == DXGI_ERROR_NOT_FOUND)
        {
            break;
        }
        if (FAILED(next))
        {
            return -PT_DML_DRIVER;
        }
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) &&
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
        {
            ++count;
        }
    }
    return count;
}

PT_DML_EXPORT std::int32_t pt_dml_adapter_name(
    std::int32_t adapterIndex,
    char* output,
    std::uint32_t outputBytes)
{
    if (adapterIndex < 0 || output == nullptr || outputBytes == 0)
    {
        return PT_DML_INVALID_ARGUMENT;
    }

    ComPtr<IDXGIAdapter1> adapter;
    if (!FindHardwareAdapter(adapterIndex, adapter))
    {
        return PT_DML_UNSUPPORTED;
    }
    DXGI_ADAPTER_DESC1 description{};
    const HRESULT result = adapter->GetDesc1(&description);
    if (FAILED(result))
    {
        return PT_DML_DRIVER;
    }

    char narrow[256]{};
    std::size_t converted = 0;
    wcstombs_s(&converted, narrow, sizeof(narrow), description.Description, _TRUNCATE);
    if (converted == 0)
    {
        narrow[0] = '\0';
    }
    const auto count = static_cast<std::size_t>(std::min<std::uint32_t>(
        outputBytes - 1, static_cast<std::uint32_t>(std::strlen(narrow))));
    std::memcpy(output, narrow, count);
    output[count] = '\0';
    return PT_DML_OK;
}

PT_DML_EXPORT std::int32_t pt_dml_create_device(
    std::int32_t adapterIndex,
    PtDmlDevice** device,
    char* error,
    std::uint32_t errorBytes)
{
    if (adapterIndex < 0 || device == nullptr)
    {
        WriteError(error, errorBytes, "adapter index or device output is invalid");
        return PT_DML_INVALID_ARGUMENT;
    }
    *device = nullptr;

    auto native = new (std::nothrow) PtDmlDevice();
    if (native == nullptr)
    {
        WriteError(error, errorBytes, "native device allocation failed");
        return PT_DML_ALLOCATION;
    }
    if (!FindHardwareAdapter(adapterIndex, native->Adapter))
    {
        delete native;
        WriteError(error, errorBytes, std::format("hardware adapter {} was not found", adapterIndex));
        return PT_DML_UNSUPPORTED;
    }

    HRESULT result = D3D12CreateDevice(
        native->Adapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&native->D3D12Device));
    if (FAILED(result))
    {
        delete native;
        WriteError(error, errorBytes, HresultMessage(result, "D3D12CreateDevice"));
        return PT_DML_DRIVER;
    }

    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    result = native->D3D12Device->CreateCommandQueue(
        &queueDescription,
        IID_PPV_ARGS(&native->CommandQueue));
    if (FAILED(result))
    {
        delete native;
        WriteError(error, errorBytes, HresultMessage(result, "CreateCommandQueue"));
        return PT_DML_DRIVER;
    }

    result = DMLCreateDevice(
        native->D3D12Device.Get(),
        DML_CREATE_DEVICE_FLAG_NONE,
        IID_PPV_ARGS(&native->DmlDevice));
    if (FAILED(result))
    {
        delete native;
        WriteError(error, errorBytes, HresultMessage(result, "DMLCreateDevice"));
        return PT_DML_DRIVER;
    }

    DXGI_ADAPTER_DESC1 description{};
    native->Adapter->GetDesc1(&description);
    native->AdapterName = description.Description;
    *device = native;
    return PT_DML_OK;
}

PT_DML_EXPORT void pt_dml_release_device(PtDmlDevice* device)
{
    delete device;
}

PT_DML_EXPORT std::int32_t pt_dml_self_test(
    PtDmlDevice* device,
    char* error,
    std::uint32_t errorBytes)
{
    if (device == nullptr ||
        device->D3D12Device == nullptr ||
        device->CommandQueue == nullptr ||
        device->DmlDevice == nullptr)
    {
        WriteError(error, errorBytes, "native DirectML device is incomplete");
        return PT_DML_INVALID_ARGUMENT;
    }
    const auto removed = device->D3D12Device->GetDeviceRemovedReason();
    if (FAILED(removed))
    {
        WriteError(error, errorBytes, HresultMessage(removed, "GetDeviceRemovedReason"));
        return PT_DML_DRIVER;
    }
    return PT_DML_OK;
}
