#include "powertorch_directml.h"

#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

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

HRESULT CreateCommittedBuffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_STATES initialState,
    D3D12_RESOURCE_FLAGS flags,
    std::uint64_t bytes,
    ComPtr<ID3D12Resource>& resource)
{
    D3D12_HEAP_PROPERTIES heap{
        heapType,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        0,
        0
    };
    D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        bytes,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        flags
    };
    return device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &description,
        initialState,
        nullptr,
        IID_PPV_ARGS(&resource));
}

void Transition(
    ID3D12GraphicsCommandList* list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
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

PT_DML_EXPORT std::int32_t pt_dml_add_f32(
    PtDmlDevice* device,
    const float* left,
    const float* right,
    float* output,
    std::uint64_t elementCount,
    char* error,
    std::uint32_t errorBytes)
{
    if (device == nullptr ||
        left == nullptr ||
        right == nullptr ||
        output == nullptr ||
        elementCount == 0 ||
        elementCount > UINT32_MAX)
    {
        WriteError(error, errorBytes, "invalid DirectML add input");
        return PT_DML_INVALID_ARGUMENT;
    }
    const UINT64 bytes = elementCount * sizeof(float);
    if (bytes > std::numeric_limits<UINT>::max())
    {
        WriteError(error, errorBytes, "DirectML add tensor exceeds the ABI v1 buffer limit");
        return PT_DML_UNSUPPORTED;
    }

    ComPtr<IDMLDevice> dml = device->DmlDevice;
    ComPtr<IDMLOperator> op;
    ComPtr<IDMLCompiledOperator> compiled;
    ComPtr<IDMLCommandRecorder> recorder;
    ComPtr<IDMLBindingTable> bindingTable;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

    UINT dimensions[] = { static_cast<UINT>(elementCount) };
    DML_BUFFER_TENSOR_DESC bufferDesc{
        DML_TENSOR_DATA_TYPE_FLOAT32,
        DML_TENSOR_FLAG_NONE,
        1,
        dimensions,
        nullptr,
        bytes,
        0
    };
    DML_TENSOR_DESC tensorDesc{DML_TENSOR_TYPE_BUFFER, &bufferDesc};
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc{
        &tensorDesc,
        &tensorDesc,
        &tensorDesc
    };
    DML_OPERATOR_DESC operatorDesc{
        DML_OPERATOR_ELEMENT_WISE_ADD,
        &addDesc
    };
    HRESULT result = dml->CreateOperator(&operatorDesc, IID_PPV_ARGS(&op));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateOperator(element-wise add)"));
        return PT_DML_COMPILATION;
    }
    result = dml->CompileOperator(
        op.Get(),
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(&compiled));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CompileOperator(element-wise add)"));
        return PT_DML_COMPILATION;
    }
    result = dml->CreateCommandRecorder(IID_PPV_ARGS(&recorder));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateCommandRecorder"));
        return PT_DML_DRIVER;
    }

    ComPtr<ID3D12Resource> leftDefault;
    ComPtr<ID3D12Resource> rightDefault;
    ComPtr<ID3D12Resource> outputDefault;
    ComPtr<ID3D12Resource> temporary;
    ComPtr<ID3D12Resource> persistent;
    ComPtr<ID3D12Resource> leftUpload;
    ComPtr<ID3D12Resource> rightUpload;
    ComPtr<ID3D12Resource> readback;
    const D3D12_RESOURCE_FLAGS dataFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    for (ComPtr<ID3D12Resource>* destination :
         std::array<ComPtr<ID3D12Resource>*, 3>{
             &leftDefault, &rightDefault, &outputDefault})
    {
        result = CreateCommittedBuffer(
            device->D3D12Device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST,
            dataFlags,
            bytes,
            *destination);
        if (FAILED(result))
        {
            WriteError(error, errorBytes, HresultMessage(result, "CreateCommittedResource(default buffer)"));
            return PT_DML_ALLOCATION;
        }
    }
    for (ComPtr<ID3D12Resource>* destination :
         std::array<ComPtr<ID3D12Resource>*, 2>{
             &leftUpload, &rightUpload})
    {
        result = CreateCommittedBuffer(
            device->D3D12Device.Get(),
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
            bytes,
            *destination);
        if (FAILED(result))
        {
            WriteError(error, errorBytes, HresultMessage(result, "CreateCommittedResource(upload buffer)"));
            return PT_DML_ALLOCATION;
        }
    }
    result = CreateCommittedBuffer(
        device->D3D12Device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
        bytes,
        readback);
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateCommittedResource(readback buffer)"));
        return PT_DML_ALLOCATION;
    }

    const DML_BINDING_PROPERTIES bindingProperties = compiled->GetBindingProperties();
    if (bindingProperties.TemporaryResourceSize > 0)
    {
        result = CreateCommittedBuffer(
            device->D3D12Device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            dataFlags,
            bindingProperties.TemporaryResourceSize,
            temporary);
        if (FAILED(result))
        {
            WriteError(error, errorBytes, HresultMessage(result, "CreateCommittedResource(temporary)"));
            return PT_DML_ALLOCATION;
        }
    }
    if (bindingProperties.PersistentResourceSize > 0)
    {
        result = CreateCommittedBuffer(
            device->D3D12Device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            dataFlags,
            bindingProperties.PersistentResourceSize,
            persistent);
        if (FAILED(result))
        {
            WriteError(error, errorBytes, HresultMessage(result, "CreateCommittedResource(persistent)"));
            return PT_DML_ALLOCATION;
        }
    }

    D3D12_DESCRIPTOR_HEAP_DESC heapDescription{
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        bindingProperties.RequiredDescriptorCount,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        0
    };
    if (heapDescription.NumDescriptors > 0)
    {
        result = device->D3D12Device->CreateDescriptorHeap(
            &heapDescription,
            IID_PPV_ARGS(&descriptorHeap));
        if (FAILED(result))
        {
            WriteError(error, errorBytes, HresultMessage(result, "CreateDescriptorHeap"));
            return PT_DML_ALLOCATION;
        }
    }
    DML_BINDING_TABLE_DESC tableDescription{};
    tableDescription.Dispatchable = compiled.Get();
    tableDescription.SizeInDescriptors = bindingProperties.RequiredDescriptorCount;
    if (descriptorHeap)
    {
        tableDescription.CPUDescriptorHandle =
            descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        tableDescription.GPUDescriptorHandle =
            descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    }
    result = dml->CreateBindingTable(&tableDescription, IID_PPV_ARGS(&bindingTable));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateBindingTable"));
        return PT_DML_DRIVER;
    }

    DML_BUFFER_BINDING inputA{leftDefault.Get(), 0, bytes};
    DML_BUFFER_BINDING inputB{rightDefault.Get(), 0, bytes};
    DML_BUFFER_BINDING outputBinding{outputDefault.Get(), 0, bytes};
    DML_BINDING_DESC inputs[]{
        {DML_BINDING_TYPE_BUFFER, &inputA},
        {DML_BINDING_TYPE_BUFFER, &inputB}
    };
    DML_BINDING_DESC outputs[]{
        {DML_BINDING_TYPE_BUFFER, &outputBinding}
    };
    bindingTable->BindInputs(static_cast<UINT>(std::size(inputs)), inputs);
    bindingTable->BindOutputs(static_cast<UINT>(std::size(outputs)), outputs);
    if (temporary)
    {
        DML_BUFFER_BINDING buffer{temporary.Get(), 0, bindingProperties.TemporaryResourceSize};
        DML_BINDING_DESC binding{DML_BINDING_TYPE_BUFFER, &buffer};
        bindingTable->BindTemporaryResource(&binding);
    }
    if (persistent)
    {
        DML_BUFFER_BINDING buffer{persistent.Get(), 0, bindingProperties.PersistentResourceSize};
        DML_BINDING_DESC binding{DML_BINDING_TYPE_BUFFER, &buffer};
        bindingTable->BindPersistentResource(&binding);
    }

    float* mappedLeft = nullptr;
    float* mappedRight = nullptr;
    result = leftUpload->Map(0, nullptr, reinterpret_cast<void**>(&mappedLeft));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "Map(left upload)"));
        return PT_DML_DRIVER;
    }
    std::memcpy(mappedLeft, left, static_cast<std::size_t>(bytes));
    leftUpload->Unmap(0, nullptr);
    result = rightUpload->Map(0, nullptr, reinterpret_cast<void**>(&mappedRight));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "Map(right upload)"));
        return PT_DML_DRIVER;
    }
    std::memcpy(mappedRight, right, static_cast<std::size_t>(bytes));
    rightUpload->Unmap(0, nullptr);

    result = device->D3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&allocator));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateCommandAllocator"));
        return PT_DML_DRIVER;
    }
    result = device->D3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&commandList));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateCommandList"));
        return PT_DML_DRIVER;
    }
    ID3D12DescriptorHeap* heaps[] = {descriptorHeap.Get()};
    if (descriptorHeap)
    {
        commandList->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
    }
    commandList->CopyResource(leftDefault.Get(), leftUpload.Get());
    commandList->CopyResource(rightDefault.Get(), rightUpload.Get());
    Transition(
        commandList.Get(),
        leftDefault.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Transition(
        commandList.Get(),
        rightDefault.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Transition(
        commandList.Get(),
        outputDefault.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    recorder->RecordDispatch(commandList.Get(), compiled.Get(), bindingTable.Get());
    Transition(
        commandList.Get(),
        outputDefault.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    commandList->CopyResource(readback.Get(), outputDefault.Get());
    result = commandList->Close();
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "Close(command list)"));
        return PT_DML_DRIVER;
    }

    ID3D12CommandList* commandLists[] = {commandList.Get()};
    device->CommandQueue->ExecuteCommandLists(
        static_cast<UINT>(std::size(commandLists)),
        commandLists);
    result = device->D3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "CreateFence"));
        return PT_DML_DRIVER;
    }
    constexpr UINT64 fenceValue = 1;
    result = device->CommandQueue->Signal(fence.Get(), fenceValue);
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "Queue->Signal"));
        return PT_DML_SYNC;
    }
    if (fence->GetCompletedValue() < fenceValue)
    {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr)
        {
            WriteError(error, errorBytes, "CreateEventW failed");
            return PT_DML_SYNC;
        }
        result = fence->SetEventOnCompletion(fenceValue, event);
        if (FAILED(result))
        {
            CloseHandle(event);
            WriteError(error, errorBytes, HresultMessage(result, "SetEventOnCompletion"));
            return PT_DML_SYNC;
        }
        const DWORD wait = WaitForSingleObject(event, 30000);
        CloseHandle(event);
        if (wait != WAIT_OBJECT_0)
        {
            WriteError(error, errorBytes, "DirectML queue wait timed out");
            return PT_DML_SYNC;
        }
    }

    float* mappedOutput = nullptr;
    result = readback->Map(0, nullptr, reinterpret_cast<void**>(&mappedOutput));
    if (FAILED(result))
    {
        WriteError(error, errorBytes, HresultMessage(result, "Map(readback)"));
        return PT_DML_DRIVER;
    }
    std::memcpy(output, mappedOutput, static_cast<std::size_t>(bytes));
    readback->Unmap(0, nullptr);
    return PT_DML_OK;
}
