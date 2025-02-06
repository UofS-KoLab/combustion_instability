param (
    [string]$dataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$approach,
    [string]$modelName,
    [string]$typeFuel
)

# Assign default values to inputFolder and rmsFile if they are not provided
if (-not $dataRoot) {
    $dataRoot = "C:\Users\qpw475\Documents\combustion_instability\data\raw\h2"
}

if (-not $projectRoot) {
    $projectRoot = "C:\Users\qpw475\Documents\combustion_instability"
}

if (-not $windowSize) {
    $windowSize = 500
}

if (-not $approach) {
    $approach = "time_series"
}

if (-not $modelName) {
    $modelName = "model_3"
}

if (-not $typeFuel) {
    $typeFuel = "hidrogen"
}

$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src\evaluate_transient_time_series_model.py"
# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --window_size $windowSize --approach $approach --model_name $modelName --type_fuel $typeFuel