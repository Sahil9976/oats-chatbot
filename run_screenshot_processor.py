#!/usr/bin/env python3
"""
Run Screenshot Processor to update workflows
This script processes screenshots and updates the workflow files
"""

import os
import sys
from pathlib import Path
from screenshot_processor import OATSScreenshotProcessor

def run_screenshot_processor():
    """Run the screenshot processor to analyze screenshots and update workflows."""
    print("🔄 Starting OATS Screenshot Processor...")
    
    # Initialize processor
    processor = OATSScreenshotProcessor()
    
    # Check for screenshots directory
    screenshots_dir = "flows/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        print("   Please create the directory and add screenshots to process.")
        return False
    
    # Find all image files in screenshots directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    screenshot_files = []
    
    for file_path in Path(screenshots_dir).glob('*'):
        if file_path.suffix.lower() in image_extensions:
            screenshot_files.append(str(file_path))
    
    if not screenshot_files:
        print(f"❌ No screenshot files found in {screenshots_dir}")
        print("   Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP")
        return False
    
    print(f"📸 Found {len(screenshot_files)} screenshots to process:")
    for file_path in screenshot_files:
        print(f"   - {os.path.basename(file_path)}")
    
    # Process each screenshot
    analyses = []
    for i, screenshot_path in enumerate(screenshot_files, 1):
        print(f"\n🔍 Analyzing screenshot {i}/{len(screenshot_files)}: {os.path.basename(screenshot_path)}")
        
        # Determine workflow context from filename
        filename = os.path.basename(screenshot_path)
        context = processor._infer_workflow_context_from_filename(filename)
        
        # Analyze screenshot
        analysis = processor.analyze_screenshot(screenshot_path, context)
        
        if analysis.get('analysis_status') == 'success':
            print(f"✅ Successfully analyzed: {filename}")
            workflow_title = analysis.get('workflow', {}).get('title', 'Unknown')
            print(f"   Workflow: {workflow_title}")
        elif analysis.get('analysis_status') == 'partial':
            print(f"⚠️ Partial analysis: {filename}")
        else:
            print(f"❌ Failed to analyze: {filename}")
            print(f"   Error: {analysis.get('error', 'Unknown error')}")
        
        analyses.append(analysis)
    
    # Update workflow files
    print(f"\n📝 Updating workflow files...")
    success = processor.update_workflow_files(analyses)
    
    if success:
        print("✅ Successfully updated workflow files!")
        print(f"   Main workflows: flows/oats_workflows.txt")
        print(f"   Additional workflows: flows/additional_workflows_template.txt")
        
        # Show summary
        successful_analyses = [a for a in analyses if a.get('analysis_status') == 'success']
        print(f"\n📊 Processing Summary:")
        print(f"   Screenshots processed: {len(screenshot_files)}")
        print(f"   Successful analyses: {len(successful_analyses)}")
        print(f"   Workflows updated: {len(successful_analyses)}")
        
        return True
    else:
        print("❌ Failed to update workflow files")
        return False

def main():
    """Main entry point."""
    try:
        success = run_screenshot_processor()
        if success:
            print("\n🎉 Screenshot processing completed successfully!")
            print("   You can now test the updated workflows in the chatbot.")
        else:
            print("\n💡 Tips:")
            print("   1. Make sure screenshots are in flows/screenshots/ directory")
            print("   2. Use descriptive filenames (e.g., 'create_job_posting_step1.png')")
            print("   3. Ensure screenshots are clear and show the full interface")
            
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
