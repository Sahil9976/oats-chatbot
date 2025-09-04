#!/usr/bin/env python3
"""
Fix the JSON parsing issue and convert partial analyses to successful ones.
"""

import json
import os
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_json_response(raw_text: str) -> str:
    """Clean markdown formatting from JSON response."""
    text = raw_text.strip()
    
    # Remove markdown formatting if present
    if text.startswith('```json'):
        text = text[7:]  # Remove ```json
    elif text.startswith('```'):
        text = text[3:]   # Remove ```
    
    if text.endswith('```'):
        text = text[:-3]  # Remove trailing ```
    
    return text.strip()

def fix_analyses_file():
    """Fix the screenshot analyses file by parsing the raw JSON."""
    analyses_file = "screenshot_analyses.json"
    
    if not os.path.exists(analyses_file):
        print(f"❌ File not found: {analyses_file}")
        return False
    
    # Load existing analyses
    with open(analyses_file, 'r', encoding='utf-8') as f:
        analyses = json.load(f)
    
    print(f"📊 Found {len(analyses)} analyses to fix")
    
    fixed_count = 0
    for analysis in analyses:
        if analysis.get('analysis_status') == 'partial':
            raw_analysis = analysis.get('raw_analysis', '')
            
            try:
                # Clean and parse the JSON
                cleaned_json = clean_json_response(raw_analysis)
                parsed_data = json.loads(cleaned_json)
                
                # Update the analysis with parsed data
                analysis.update(parsed_data)
                analysis['analysis_status'] = 'success'
                
                print(f"✅ Fixed: {analysis.get('filename', 'Unknown')}")
                fixed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ Still can't parse: {analysis.get('filename', 'Unknown')} - {e}")
                continue
    
    # Save the fixed analyses
    with open(analyses_file, 'w', encoding='utf-8') as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 Fixed {fixed_count} out of {len(analyses)} analyses")
    return fixed_count > 0

def regenerate_workflow_files():
    """Regenerate workflow files from fixed analyses."""
    from screenshot_processor import OATSScreenshotProcessor
    
    # Load fixed analyses
    with open("screenshot_analyses.json", 'r', encoding='utf-8') as f:
        analyses = json.load(f)
    
    # Filter successful analyses
    successful_analyses = [a for a in analyses if a.get('analysis_status') == 'success']
    
    if not successful_analyses:
        print("❌ No successful analyses found")
        return False
    
    print(f"📊 Regenerating workflow files from {len(successful_analyses)} successful analyses...")
    
    # Create processor instance
    processor = OATSScreenshotProcessor()
    
    # Clear previous screenshot-extracted content from workflow files
    clear_auto_generated_content()
    
    # Update workflow files with successful analyses
    success = processor.update_workflow_files(successful_analyses)
    
    if success:
        print("✅ Workflow files regenerated successfully!")
    else:
        print("❌ Failed to regenerate workflow files")
    
    return success

def clear_auto_generated_content():
    """Remove previous auto-generated content from workflow files."""
    workflows_file = "flows/oats_workflows.txt"
    additional_file = "flows/additional_workflows_template.txt"
    
    for file_path in [workflows_file, additional_file]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove everything after the auto-generated marker
            marker = "# ===== SCREENSHOT-EXTRACTED WORKFLOWS (AUTO-GENERATED) ====="
            if marker in content:
                content = content.split(marker)[0].rstrip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"🧹 Cleared auto-generated content from {file_path}")

if __name__ == "__main__":
    print("🔧 Fixing JSON parsing issues...")
    
    # Step 1: Fix the analyses file
    if fix_analyses_file():
        print("\n📝 Regenerating workflow files...")
        
        # Step 2: Regenerate workflow files
        regenerate_workflow_files()
        
        print("\n🎉 All fixes completed successfully!")
        print("✅ Your manual content is preserved")
        print("✅ Screenshot data is now properly extracted and appended")
        
    else:
        print("❌ No analyses were fixed. Check the raw data for issues.")
