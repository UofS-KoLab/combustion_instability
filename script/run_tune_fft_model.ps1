param (
    [string]$dataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile,
    [string]$approach,
    [int]$duration_sample_ms,
    [float]$ValidationSplit,
    [float]$TestSize,
    [int]$Seed,
    [string]$Features

)

# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\h2"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $stabilityFile) {
    $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\h2_label.csv"
}

if (-not $windowSize) {
    $windowSize = 300
}

if (-not $duration_sample_ms) {
    $duration_sample_ms = 12000
}

if (-not $approach) {
    $approach = "fft"
}

if (-not $ValidationSplit) {
    $ValidationSplit = 0.2
}

if (-not $TestSize) {
    $TestSize = 0.2
}

if (-not $Seed) {
    $Seed = 42
}

if (-not $Features) {
    $Features = "7,9"
}

$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src\tune_hyperparams_fft_model.py"

# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --duration_sample_ms $duration_sample_ms --validation_split $ValidationSplit --test_size $TestSize --seed $Seed --features $Features