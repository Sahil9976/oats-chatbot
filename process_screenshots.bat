@echo off
echo ============================================
echo  OATS Screenshot Processor
echo ============================================
echo.
echo This script will:
echo 1. Analyze all screenshots in flows/screenshots/
echo 2. Extract accurate workflow information
echo 3. Update both workflow text files
echo 4. Generate analysis reports
echo.

REM Check if screenshots directory exists
if not exist "flows\screenshots" (
    echo Creating screenshots directory...
    mkdir "flows\screenshots"
    echo.
    echo 📁 Created flows/screenshots/ directory
    echo 💡 Please add your OATS website screenshots to this folder
    echo    and run this script again.
    echo.
    pause
    exit /b
)

REM Count screenshots
set /a count=0
for %%f in (flows\screenshots\*.png flows\screenshots\*.jpg flows\screenshots\*.jpeg flows\screenshots\*.bmp flows\screenshots\*.gif flows\screenshots\*.webp) do (
    set /a count+=1
)

if %count%==0 (
    echo ⚠️  No screenshots found in flows/screenshots/
    echo.
    echo Please add your OATS website screenshots to:
    echo   flows/screenshots/
    echo.
    echo Supported formats: PNG, JPG, JPEG, BMP, GIF, WEBP
    echo.
    pause
    exit /b
)

echo 📸 Found %count% screenshot(s) to process
echo.

REM Run the screenshot processor
echo 🚀 Starting screenshot analysis...
python screenshot_processor.py

echo.
echo ✅ Processing complete!
echo.
echo Generated files:
echo   - Updated flows/oats_workflows.txt
echo   - Updated flows/additional_workflows_template.txt
echo   - screenshot_analysis_report.md
echo   - screenshot_analyses.json
echo.
echo 🎉 Your chatbot is now updated with real OATS workflow data!
echo.
pause
