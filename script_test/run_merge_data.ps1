param (
    [string]$dataRoot,
    [string]$oldDataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile,
    [string]$approach,
    [string]$fuelType
)

# $fuel="transient"
$fuel="h2"
# $fuel="A-2"
# $fuel="C-1"
# $fuel="C-5"
# $fuel="C-9"
# $fuel="Jet-A"
# $fuel="JP8_HRJ"

# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\${fuel}"
}

if (-not $oldDataRoot) {
    # $oldDataRoot = "C:\Users\qpw475\Documents\combustion_instability_copy\data\raw\labels\rayleigh\${fuel}_one_peaks.csv"
    $oldDataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\stats\${fuel}_stats.csv"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $stabilityFile) {
    # $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\${fuel}_label.csv"
     $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\h2_label.csv"
}
if (-not $windowSize) {
    $windowSize = 12000 #100
}

if (-not $approach) {
    $approach = "fft"
}

if (-not $fuelType) {
    $fuelType = $fuel
}

$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src_test\merge_data.py" #  borrar
# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --old_data_file $oldDataRoot --fuel_type $fuelType