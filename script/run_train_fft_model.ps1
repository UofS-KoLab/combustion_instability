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
    $windowSize = 30 #300#100 #30
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
    $DropoutRate1 =0.2 # 0.4 #0.3
}
if (-not $DropoutRate2) {
    $DropoutRate2 = 0.1 #0.3 #0.0
}

if (-not $L2Regularizer1) {
    $L2Regularizer1= 0.00020285565908290863 # 0.006237 # 0.00042865485672462857 #0.0004934463 # 0.0003164584608354987 # 0.000493446318593248
    #0.00085388
    # $L2Regularizer1 = 0.004188

    #$L2Regularizer1 = 0.0099
    # $L2Regularizer1 = 0.006237

    # $L2Regularizer1 = 0.00045
    # $L2Regularizer1 = 0.008
    
    
}
if (-not $L2Regularizer2) {
    $L2Regularizer2= 0.0013660049365971218 #0.001 #0.002647525297275649 # 0.0002292899 #0.0015674105671406005 #0.0002292898535786854 #0.0002136367
    # $L2Regularizer2 = 0.0032857

    # $L2Regularizer2 = 0.0004
    # $L2Regularizer2 = 0.001
    # $L2Regularizer2 = 0.000953
    # $L2Regularizer2 = 0.003
}
if (-not $LearningRate) {
    # $LearningRate =  0.00365337  0.0003477
    $LearningRate = 0.000951175845418114
 #    0.002 #  0.0011910703642278476 #0.0005418032 #0.0003730278430367326 #0.0005418031892376496 #0.00641685 #
    # $LearningRate = 0.008
    # $LearningRate = 0.0003477
    # $LearningRate = 0.002
    
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
    $Features = "7,9" #7,9
}


$python_executable = "C:\Programs\Anaconda3\envs\hidrogen\python.exe"
$python_script = "C:\Users\qpw475\Documents\combustion_instability\src\train_fft_model.py"

# Construct the command to run the Python script
& $python_executable $python_script --data_root $dataRoot --project_root $projectRoot --stability_file $stabilityFile --window_size $windowSize --approach $approach --duration_sample_ms $duration_sample_ms --lstm_units1 $LSTMUnits1 --lstm_units2 $LSTMUnits2 --dropout_rate1 $DropoutRate1 --dropout_rate2 $DropoutRate2 --l2_regularizer1 $L2Regularizer1 --l2_regularizer2 $L2Regularizer2 --learning_rate $LearningRate --epochs $Epochs --validation_split $ValidationSplit --patience $Patience --test_size $TestSize --seed $Seed --features $Features
