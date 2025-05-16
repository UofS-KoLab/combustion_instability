param (
    [string]$dataRoot,
    [string]$oldDataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile,
    [string]$approach,
    [string]$fuelType
)

# $fuel="JP8_HRJ"
$fuel="h2"

# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\h2"
}

if (-not $oldDataRoot) {
    $oldDataRoot = "C:\Users\qpw475\Documents\combustion_instability_copy\data\raw\labels\rayleigh\${fuel}_one_peaks.csv"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $stabilityFile) {
    # $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\${fuel}_label.csv"
    $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\h2_label.csv"
}
if (-not $windowSize) {
    $windowSize = 12000 #100 #12000
}

if (-not $approach) {
    $approach = "time_series"
}

if (-not $fuelType) {
    $fuelType = $fuel
}

$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src_test\get_time_domain_features.py" #  borrar
# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --old_data_file $oldDataRoot --fuel_type $fuelType