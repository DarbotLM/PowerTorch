#Requires -Version 7.4

BeforeAll {
    $projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $modulePath = Join-Path $projectRoot 'artifacts\module\PowerTorch.psd1'
    Import-Module $modulePath -Force
}

Describe 'PowerTorch binary module' {
    It 'loads a native module without requiring Python' {
        $info = Get-PowerTorchInfo
        $info.Name | Should -Be 'PowerTorch'
        $info.PythonRuntimeRequired | Should -BeFalse
    }

    It 'reports CPU and DirectML health' {
        $health = Test-PowerTorch -DirectML
        $health.Healthy | Should -BeTrue
        $health.Cpu.Backend | Should -Be 'PowerTorch.Cpu'
        $health.DirectML.Driver | Should -Not -BeNullOrEmpty
    }

    It 'executes the managed CPU tensor shape and model-math slice' {
        $left = New-PowerTorchTensor -Float32 1, 2, 3, 4 -Shape 2, 2
        $right = New-PowerTorchTensor -Float32 5, 6, 7, 8 -Shape 2, 2
        $matmul = Invoke-PowerTorchOperation -Operation Matmul -Left $left -Right $right
        $softmax = Invoke-PowerTorchOperation -Operation Softmax -Left $matmul -Dimension -1
        $normalized = Invoke-PowerTorchOperation `
            -Operation RmsNorm -Left $matmul -Shape 2
        $permuted = Invoke-PowerTorchOperation `
            -Operation Permute -Left $matmul -Permutation 1, 0

        (Get-PowerTorchTensor -Tensor $matmul).ElementCount | Should -Be 4
        (Get-PowerTorchTensor -Tensor $softmax).Shape | Should -Be @(2, 2)
        (Get-PowerTorchTensor -Tensor $normalized).IsContiguous | Should -BeTrue
        (Get-PowerTorchTensor -Tensor $permuted).IsContiguous | Should -BeTrue
        $left.Dispose()
        $right.Dispose()
        $matmul.Dispose()
        $softmax.Dispose()
        $normalized.Dispose()
        $permuted.Dispose()
    }

    It 'executes embedding convolution rotary and causal attention CPU operations' {
        $weights = New-PowerTorchTensor -Float32 1, 2, 3, 4 -Shape 2, 2
        $indices = New-PowerTorchTensor -Int64 1, 0 -Shape 2
        $embedding = Invoke-PowerTorchOperation `
            -Operation Embedding -Left $weights -Right $indices
        $positions = New-PowerTorchTensor -Int64 0, 1 -Shape 2
        $rotary = Invoke-PowerTorchOperation `
            -Operation Rotary -Left $embedding -Positions $positions

        $input = New-PowerTorchTensor -Float32 1, 2, 3, 4 -Shape 1, 1, 4
        $kernel = New-PowerTorchTensor -Float32 1, 1 -Shape 1, 1, 2
        $conv = Invoke-PowerTorchOperation `
            -Operation Conv1D -Left $input -Right $kernel -Causal

        $query = New-PowerTorchTensor -Float32 0, 0, 0, 0 -Shape 1, 2, 1, 2
        $key = New-PowerTorchTensor -Float32 0, 0 -Shape 1, 1, 1, 2
        $value = New-PowerTorchTensor -Float32 2, 4 -Shape 1, 1, 1, 2
        $attention = Invoke-PowerTorchOperation `
            -Operation CausalAttention -Left $query -Right $key -Value $value

        (Get-PowerTorchTensor -Tensor $embedding).Shape | Should -Be @(2, 2)
        (Get-PowerTorchTensor -Tensor $conv).Shape | Should -Be @(1, 1, 4)
        (Get-PowerTorchTensor -Tensor $rotary).Shape | Should -Be @(2, 2)
        (Get-PowerTorchTensor -Tensor $attention).Shape | Should -Be @(1, 2, 1, 2)
        $weights.Dispose()
        $indices.Dispose()
        $embedding.Dispose()
        $positions.Dispose()
        $rotary.Dispose()
        $input.Dispose()
        $kernel.Dispose()
        $conv.Dispose()
        $query.Dispose()
        $key.Dispose()
        $value.Dispose()
        $attention.Dispose()
    }

    It 'executes cache sampling trig random and host residency operations' {
        $cache = New-PowerTorchTensor -Float32 1, 2, 3, 4 -Shape 1, 1, 2, 2
        $update = New-PowerTorchTensor -Float32 5, 6 -Shape 1, 1, 1, 2
        $appended = Invoke-PowerTorchOperation `
            -Operation KvCacheAppend -Left $cache -Right $update
        $indices = Invoke-PowerTorchOperation -Operation ArgMax -Left $appended
        $normal = Invoke-PowerTorchOperation `
            -Operation RandomNormal -Shape 4 -DataType BFloat16 -Seed 42
        $sine = Invoke-PowerTorchOperation -Operation Sin -Left $normal
        $hostCopy = Invoke-PowerTorchOperation -Operation Transfer -Left $normal

        (Get-PowerTorchTensor -Tensor $appended).Shape | Should -Be @(1, 1, 3, 2)
        (Get-PowerTorchTensor -Tensor $indices).DataType | Should -Be 'Int64'
        (Get-PowerTorchTensor -Tensor $normal).DataType | Should -Be 'BFloat16'
        (Get-PowerTorchTensor -Tensor $sine).DataType | Should -Be 'BFloat16'
        (Get-PowerTorchTensor -Tensor $hostCopy).Residency | Should -Be 'HostManaged'
        $cache.Dispose()
        $update.Dispose()
        $appended.Dispose()
        $indices.Dispose()
        $normal.Dispose()
        $sine.Dispose()
        $hostCopy.Dispose()
    }

    It 'parses the real Florence safetensors checkpoint' {
        $checkpointPath = Join-Path (Split-Path -Parent $projectRoot) `
            'OmniParser\weights\florence31_icon_caption\model.safetensors'
        $checkpoint = Get-PowerTorchCheckpoint -Path $checkpointPath
        $checkpoint.TensorCount | Should -BeGreaterThan 600
        $checkpoint.DataBytes | Should -BeGreaterThan 880MB
    }

    It 'executes CPU operations on a real checkpoint tensor' {
        $checkpointPath = Join-Path (Split-Path -Parent $projectRoot) `
            'OmniParser\weights\florence31_icon_caption\model.safetensors'
        $tensor = Read-PowerTorchCheckpointTensor -Path $checkpointPath -MaximumBytes 100000
        $result = Invoke-PowerTorchOperation -Operation Add -Left $tensor -Right $tensor
        $info = Get-PowerTorchTensor -Tensor $result
        $info.DataType | Should -Be 'Float32'
        $info.ElementCount | Should -BeGreaterThan 0
        $tensor.Dispose()
        $result.Dispose()
    }

    It 'executes native DirectML add on a real checkpoint tensor' {
        $checkpointPath = Join-Path (Split-Path -Parent $projectRoot) `
            'OmniParser\weights\florence31_icon_caption\model.safetensors'
        $tensor = Read-PowerTorchCheckpointTensor -Path $checkpointPath -MaximumBytes 100000
        $result = Invoke-PowerTorchOperation `
            -Operation Add -Left $tensor -Right $tensor -Backend DirectML
        (Get-PowerTorchTensor -Tensor $result).ElementCount |
            Should -Be $tensor.ElementCount
        $tensor.Dispose()
        $result.Dispose()
    }

    It 'opens a real OCR ONNX model on DirectML' {
        $onnxPath = Join-Path (Split-Path -Parent (Split-Path -Parent $projectRoot)) `
            'darbot-env\Lib\site-packages\rapidocr\models\PP-OCRv6_det_small.onnx'
        $model = Get-PowerTorchOnnx -Path $onnxPath
        $model.DirectMLRequested | Should -BeTrue
        $model.RequestedProvider | Should -Be 'DmlExecutionProvider'
        $model.Inputs.Count | Should -BeGreaterThan 0
        $model.Outputs.Count | Should -BeGreaterThan 0
    }

    It 'performs backward and one AdamW step on a real tensor' {
        $checkpointPath = Join-Path (Split-Path -Parent $projectRoot) `
            'OmniParser\weights\florence31_icon_caption\model.safetensors'
        $tensor = Read-PowerTorchCheckpointTensor -Path $checkpointPath -MaximumBytes 100000
        $parameter = New-PowerTorchVariable -Tensor $tensor -RequiresGradient
        $sigmoid = Invoke-PowerTorchVariableOperation -Operation Sigmoid -Left $parameter
        $loss = Invoke-PowerTorchVariableOperation -Operation Sum -Left $sigmoid
        Invoke-PowerTorchBackward -Variable $loss | Out-Null
        $parameter.Gradient | Should -Not -BeNullOrEmpty
        $optimizer = New-PowerTorchAdamW -LearningRate 0.0001 -WeightDecay 0
        Invoke-PowerTorchOptimizerStep -Optimizer $optimizer -Parameter $parameter |
            Should -Be 1
        Clear-PowerTorchGradient -Parameter $parameter
        $parameter.Gradient | Should -BeNullOrEmpty
        $optimizer.Dispose()
        $loss.Dispose()
        $sigmoid.Dispose()
        $parameter.Dispose()
    }
}
