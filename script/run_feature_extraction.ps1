param (
    [string]$dataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile,
    [string]$approach,
    [int]$duration_sample_ms,
    [string]$fuelType
)
# $fuel ="JP8_HRJ"
$fuel ="h2"
# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\${fuel}"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $stabilityFile) {
    # $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\h2_cluster_3.csv"
    $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\${fuel}_label.csv"
}

if (-not $fuelType) {
    $fuelType = $fuel
}

if (-not $windowSize) {
    $windowSize = 100#12000#500#300#30 #100 #12000
}

if (-not $duration_sample_ms) {
    $duration_sample_ms = 12000
}

if (-not $approach) {
    $approach = "fft"
}



$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src\feature_extraction.py"

# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --duration_sample_ms $duration_sample_ms --fuel_type $fuelType