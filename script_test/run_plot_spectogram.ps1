param (
    [string]$stabilityFile,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$approach,
    [string]$fuelType,

    [string]$dataRoot,
    [string]$oldDataRoot,
    [string]$fft_path,
    [string]$cluster_label_file,
    [string]$fft_stats_info_file
)

$fuel="h2"
# $fuel="A-2"
$threshold="0.7"

if (-not $stabilityFile) {
    $stabilityFile = "C:\Users\qpw475\Documents\combustion_instability\data\labels\${fuel}_label.csv"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $windowSize) {
    $windowSize = 12000
}

if (-not $approach) {
    $approach = "time_series"
}

if (-not $fuelType) {
    $fuelType = $fuel
}


# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\${fuel}"
}

if (-not $oldDataRoot) {
    $oldDataRoot = "C:\Users\qpw475\Documents\combustion_instability_copy\data\raw\labels\rayleigh\${fuel}_one_peaks.csv"
}

if (-not $fft_path) {
    $fft_path = "C:\Users\qpw475\Documents\combustion_instability\data\fft\${fuel}\window_12000ms\data.pkl"
}




if (-not $cluster_label_file) {
    $cluster_label_file = "C:\Users\qpw475\Documents\combustion_instability\data\labels\${fuel}kmeans_thr_${threshold}_label.csv"
}
if (-not $fft_stats_info_file) {
    $fft_stats_info_file = "C:\Users\qpw475\Documents\combustion_instability\data\cluster\${fuel}_cluster_thr_${threshold}.csv"
}





$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src_test\plot_spectogram.py" #  borrar
# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --old_data_file $oldDataRoot --fuel_type $fuelType --cluster_label_file $cluster_label_file --fft_stats_info_file $fft_stats_info_file --fft_path $fft_path