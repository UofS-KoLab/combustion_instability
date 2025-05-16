param (
    [string]$dataRoot,
    [string]$oldDataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile_to_save,
    [string]$approach,
    [string]$fuelType,
    [string]$threshold
)

$fuel="h2"
$thr="0.7"

# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\cluster\h2_cluster_thr_${thr}.csv"
}

if (-not $oldDataRoot) {
    $oldDataRoot = "C:\Users\qpw475\Documents\combustion_instability_copy\data\raw\labels\rayleigh\${fuel}_one_peaks.csv"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $stabilityFile_to_save) {
    $stabilityFile_to_save = "C:\Users\qpw475\Documents\combustion_instability\data\labels"
}
if (-not $windowSize) {
    $windowSize = 12000
}

if (-not $approach) {
    $approach = "fft"
}

if (-not $fuelType) {
    $fuelType = $fuel
}

if (-not $threshold) {
    $threshold = $thr
}

$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src_test\dbscan_cluster.py" #  borrar
# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file_to_save $stabilityFile_to_save --window_size $windowSize --approach $approach --old_data_file $oldDataRoot --fuel_type $fuelType --threshold $threshold