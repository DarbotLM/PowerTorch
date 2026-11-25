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
