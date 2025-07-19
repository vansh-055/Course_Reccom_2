
Write-Host "--- Starting SNPE Conversion ---" -ForegroundColor Green

# Step 1: Set up the SNPE environment
Write-Host "Setting up SNPE environment..."
# Execute envsetup.ps1 using dot-sourcing and explicit architecture
. "C:\QualcommSNPE\qairt\2.36.0.250627\bin\envsetup.ps1" -arch X86_64

# Check if the last command (envsetup.ps1) resulted in an error
# Note: envsetup.ps1 can sometimes exit without setting $LASTEXITCODE, but PowerShell might still capture errors.
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: envsetup.ps1 exited with code $LASTEXITCODE. This indicates an internal problem within the SNPE SDK setup script." -ForegroundColor Red;
    exit 1
}
if ($Error.Count -gt 0) {
    Write-Host "ERROR: PowerShell detected an error during envsetup.ps1 execution, even if \$LASTEXITCODE was 0." -ForegroundColor Red;
    Write-Host "Last PowerShell Error: $($Error[0].Exception.Message)" -ForegroundColor Red;
    # Clear the error to avoid it affecting subsequent commands unnecessarily
    $Error.Clear() # Clear errors from $Error collection for subsequent commands
    exit 1
}

Write-Host "[INFO] QAIRT_SDK_ROOT: C:\QualcommSNPE\qairt\2.36.0.250627"
Write-Host "[INFO] QAIRT SDK environment setup complete"
Write-Host "SNPE Environment check passed." -ForegroundColor Green


# Step 2: Run ONNX to DLC conversion
Write-Host "`nConverting ONNX to float DLC..."
# Note: User and course IDs are expected to be single values for this model, hence -d user_id 1 -d course_id 1
& "C:\QualcommSNPE\qairt\2.36.0.250627\bin\venv\Scripts\python.exe" "C:\QualcommSNPE\qairt\2.36.0.250627\bin\x86_64-windows-msvc\snpe-onnx-to-dlc" -d user_id 1 -d course_id 1 -i "C:\dev\recommendation-backend\snpe_conversion_output\model.onnx" -o "C:\dev\recommendation-backend\snpe_conversion_output\model_float.dlc"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: ONNX to DLC conversion failed." -ForegroundColor Red; exit 1 }

# Step 3: Run DLC Quantization
Write-Host "`nQuantizing DLC..."
& "C:\QualcommSNPE\qairt\2.36.0.250627\bin\x86_64-windows-msvc\snpe-dlc-quant" --input_dlc "C:\dev\recommendation-backend\snpe_conversion_output\model_float.dlc" --input_list "C:\dev\recommendation-backend\snpe_conversion_output\input_list.txt" --output_dlc "C:\dev\recommendation-backend\snpe_conversion_output\model_quantized.dlc"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: DLC quantization failed." -ForegroundColor Red; exit 1 }

Write-Host "`n--- CONVERSION COMPLETE ---" -ForegroundColor Green
Write-Host "Final model is ready at: C:\dev\recommendation-backend\snpe_conversion_output\model_quantized.dlc"
