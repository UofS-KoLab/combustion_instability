param (
    [string]$dataRoot,
    [string]$projectRoot,
    [int]$windowSize,
    [string]$stabilityFile,
    [string]$approach,
    [int]$duration_sample_ms,
    [int]$LSTMUnits1,
    [int]$LSTMUnits2,
    [float]$DropoutRate1,
    [float]$DropoutRate2,
    [float]$L2Regularizer1,
    [float]$L2Regularizer2,
    [float]$LearningRate,
    [int]$Epochs,
    [float]$ValidationSplit,
    [float]$TestSize,
    [int]$Patience,
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
    $windowSize = 100
}

if (-not $duration_sample_ms) {
    $duration_sample_ms = 12000
}

if (-not $approach) {
    $approach = "fft"
}

if (-not $LSTMUnits1) {
    $LSTMUnits1 = 160
}
if (-not $LSTMUnits2) {
    $LSTMUnits2 = 192
}
if (-not $DropoutRate1) {
    $DropoutRate1 = 0.3
}
if (-not $DropoutRate2) {
    $DropoutRate2 = 0.0
}
if (-not $L2Regularizer1) {
    # $L2Regularizer1 = 0.0099
    $L2Regularizer1 = 0.00045
    
}
if (-not $L2Regularizer2) {
    # $L2Regularizer2 = 0.0004
    $L2Regularizer2 = 0.0012
}
if (-not $LearningRate) {
    # $LearningRate = 0.008
    $LearningRate = 0.002
}
if (-not $Epochs) {
    $Epochs = 150
}
if (-not $ValidationSplit) {
    $ValidationSplit = 0.2
}
if (-not $Patience) {
    $Patience = 3
}
if (-not $TestSize) {
    $TestSize = 0.2
}
if (-not $Seed) {
    $Seed = 42
}
if (-not $Features) {
    $Features = "2,7,9"
}


$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src\train_fft_model.py"

# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --duration_sample_ms $duration_sample_ms --lstm_units1 $LSTMUnits1 --lstm_units2 $LSTMUnits2 --dropout_rate1 $DropoutRate1 --dropout_rate2 $DropoutRate2 --l2_regularizer1 $L2Regularizer1 --l2_regularizer2 $L2Regularizer2 --learning_rate $LearningRate --epochs $Epochs --validation_split $ValidationSplit --patience $Patience --test_size $TestSize --seed $Seed --features $Features
