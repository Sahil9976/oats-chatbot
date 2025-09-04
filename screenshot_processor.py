#!/usr/bin/env python3
"""
OATS Screenshot Processor
Analyzes website screenshots and extracts workflow information to update flow files
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class OATSScreenshotProcessor:
    """Processes screenshots to extract accurate workflow information for OATS system."""
    
    def __init__(self, gemini_api_key: str = "AIzaSyAsVkN0ygVBsl2tVAN_Dq5E0AY5aabyrqA"):
        """Initialize with Gemini API for image analysis."""
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini AI initialized for screenshot analysis")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            self.gemini_model = None
        
        self.screenshots_dir = "flows/screenshots"
        self.workflows_file = "flows/oats_workflows.txt"
        self.additional_workflows_file = "flows/additional_workflows_template.txt"
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    def analyze_screenshot(self, image_path: str, workflow_context: str = "") -> Dict:
        """Analyze a screenshot and extract workflow information."""
        if not self.gemini_model:
            return {"error": "Gemini AI not available"}
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Create the enhanced analysis prompt
            prompt = f"""
You are an expert OATS (Otomashen ATS) system analyst. Your job is to extract PRECISE, ACTIONABLE workflow information from screenshots.

**SCREENSHOT CONTEXT:**
- Filename: {os.path.basename(image_path)}
- Expected Workflow: {workflow_context if workflow_context else "General OATS workflow"}

**ANALYSIS FOCUS:**
Analyze this screenshot and provide precise workflow information in JSON format:

{{
  "workflow": {{
    "title": "Clear, descriptive title that users would search for",
    "description": "Brief description of what this workflow accomplishes",
    "steps": [
      "Navigate to [exact menu name] from the main navigation",
      "Click the '[exact button text]' button",
      "Fill in the '[exact field label]' field with [description]",
      "Select '[option]' from the '[dropdown label]' dropdown",
      "Click '[exact button text]' to complete the action"
    ]
  }},
  "keywords": [
    "Primary search terms users would type",
    "Alternative terms",
    "Button and field names",
    "Process-related terms"
  ],
  "ui_elements": {{
    "buttons": [
      {{"text": "Exact button text", "action": "What it does"}}
    ],
    "fields": [
      {{"label": "Exact field label", "type": "text/dropdown/checkbox", "required": true}}
    ]
  }},
  "page_info": {{
    "title": "Exact page title from screenshot",
    "section": "Main OATS section (Jobs/Candidates/Clients/etc.)",
    "breadcrumb": "Navigation path if visible"
  }}
}}

**REQUIREMENTS:**
- Use EXACT text from buttons, fields, and UI elements
- Focus on actionable steps
- Include navigation context
- Generate realistic search keywords
"""

            # Upload image and analyze
            uploaded_file = genai.upload_file(image_path)
            response = self.gemini_model.generate_content([prompt, uploaded_file])
            
            # Parse JSON response
            try:
                workflow_data = json.loads(response.text.strip())
                workflow_data["screenshot_path"] = image_path
                workflow_data["analysis_status"] = "success"
                return workflow_data
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return {
                    "analysis_status": "partial",
                    "raw_analysis": response.text,
                    "screenshot_path": image_path
                }
        
        except Exception as e:
            logger.error(f"Error analyzing screenshot {image_path}: {e}")
            return {
                "analysis_status": "error",
                "error": str(e),
                "screenshot_path": image_path
            }
            response = self.gemini_model.generate_content([prompt, uploaded_file])
            
            # Parse JSON response
            try:
                workflow_data = json.loads(response.text.strip())
                workflow_data["screenshot_path"] = image_path
                workflow_data["analysis_status"] = "success"
                return workflow_data
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return {
                    "analysis_status": "partial",
                    "raw_analysis": response.text,
                    "screenshot_path": image_path
                }
        
        except Exception as e:
            logger.error(f"Error analyzing screenshot {image_path}: {e}")
            return {
                "analysis_status": "error",
                "error": str(e),
                "screenshot_path": image_path
            }
    
    def process_all_screenshots(self) -> List[Dict]:
        """Process all screenshots in the screenshots directory."""
        results = []
        
        # Supported image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        
        screenshots_path = Path(self.screenshots_dir)
        if not screenshots_path.exists():
            logger.warning(f"Screenshots directory not found: {screenshots_path}")
            return results
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(screenshots_path.glob(f"*{ext}"))
            image_files.extend(screenshots_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} screenshots to process")
        
        for image_file in sorted(image_files):
            print(f"📸 Processing: {image_file.name}")
            
            # Try to determine context from filename
            context = self._get_context_from_filename(image_file.name)
            
            # Analyze screenshot
            analysis = self.analyze_screenshot(str(image_file), context)
            analysis["filename"] = image_file.name
            results.append(analysis)
            
            print(f"   ✅ Analyzed: {analysis.get('analysis_status', 'unknown')}")
        
        return results
    
    def _infer_workflow_context_from_filename(self, filename: str) -> str:
        """Infer workflow context from screenshot filename."""
        return self._get_context_from_filename(filename)
    
    def _get_context_from_filename(self, filename: str) -> str:
        """Extract context from screenshot filename with enhanced detection for descriptive naming."""
        filename_lower = filename.lower()
        
        # Parse numbered sections and descriptive elements
        # Remove file extension and split into parts
        base_name = filename_lower.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.gif', '')
        
        # Extract section number if present (e.g., "1_", "2_", "section1_")
        section_number = ""
        if base_name.startswith(tuple(f"{i}_" for i in range(1, 20))):
            section_number = base_name.split('_')[0]
            base_name = '_'.join(base_name.split('_')[1:])  # Remove number prefix
        
        # Extract descriptive context from the full filename
        descriptive_context = base_name.replace('_', ' ').strip()
        
        # Enhanced context keywords mapping based on common naming patterns
        context_keywords = {
            # Job-related workflows
            'job': 'Job creation and management workflow',
            'jd': 'Job description creation workflow',
            'job_creation': 'Job creation workflow',
            'job_posting': 'Job posting workflow',
            'job_form': 'Job form filling workflow',
            'create_job': 'Job creation workflow',
            'new_job': 'New job creation workflow',
            'add_job': 'Add job workflow',
            'post_job': 'Job posting workflow',
            
            # Candidate-related workflows
            'candidate': 'Candidate management workflow',
            'candidate_search': 'Candidate search and filtering workflow',
            'candidate_list': 'Candidate listing and browsing workflow',
            'candidate_profile': 'Candidate profile management workflow',
            'applicant': 'Applicant management workflow',
            'resume': 'Resume management workflow',
            'cv': 'CV management workflow',
            
            # Client-related workflows
            'client': 'Client management workflow',
            'client_management': 'Client management workflow',
            'client_form': 'Client form workflow',
            'company': 'Company/Client management workflow',
            'customer': 'Customer management workflow',
            
            # Dashboard and analytics
            'dashboard': 'Dashboard and analytics workflow',
            'analytics': 'Analytics and reporting workflow',
            'report': 'Reporting workflow',
            'overview': 'Overview and dashboard workflow',
            'stats': 'Statistics and metrics workflow',
            'metrics': 'Metrics and KPI workflow',
            
            # User and system management
            'login': 'Authentication and login workflow',
            'user': 'User management workflow',
            'admin': 'Administrative workflow',
            'settings': 'Settings configuration workflow',
            'profile': 'Profile management workflow',
            'account': 'Account management workflow',
            
            # Vendor and partner management
            'vendor': 'Vendor management workflow',
            'supplier': 'Supplier management workflow',
            'partner': 'Partner management workflow',
            
            # Search and filtering
            'search': 'Search and filtering workflow',
            'filter': 'Filtering and sorting workflow',
            'find': 'Search and find workflow',
            'lookup': 'Lookup and search workflow',
            
            # CRUD operations
            'create': 'Creation workflow',
            'edit': 'Editing workflow',
            'update': 'Update workflow',
            'modify': 'Modification workflow',
            'add': 'Adding new items workflow',
            'new': 'New item creation workflow',
            'delete': 'Deletion workflow',
            'remove': 'Removal workflow',
            
            # Listing and browsing
            'list': 'Listing and browsing workflow',
            'browse': 'Browsing workflow',
            'view': 'Viewing workflow',
            'display': 'Display workflow',
            
            # Forms and data entry
            'form': 'Form filling workflow',
            'input': 'Data input workflow',
            'entry': 'Data entry workflow',
            'submit': 'Form submission workflow',
            
            # Navigation and menu
            'menu': 'Navigation menu workflow',
            'nav': 'Navigation workflow',
            'sidebar': 'Sidebar navigation workflow',
            'header': 'Header navigation workflow',
            
            # Process-specific
            'interview': 'Interview management workflow',
            'onboard': 'Onboarding workflow',
            'hire': 'Hiring process workflow',
            'recruit': 'Recruitment workflow',
            'placement': 'Placement workflow'
        }
        
        # Find the most specific match (longer keywords first)
        sorted_keywords = sorted(context_keywords.items(), key=lambda x: len(x[0]), reverse=True)
        
        # First try to match specific keywords in the descriptive context
        matched_context = None
        for keyword, context in sorted_keywords:
            if keyword in filename_lower:
                matched_context = context
                break
        
        # If we have a section number, include it in the context
        if section_number and matched_context:
            return f"Section {section_number}: {matched_context} - {descriptive_context}"
        elif matched_context:
            return f"{matched_context} - {descriptive_context}"
        else:
            # If no specific keyword match, use the descriptive context
            return f"OATS interface workflow: {descriptive_context}"
    
    def update_workflow_files(self, analyses: List[Dict]) -> bool:
        """Update workflow files by OVERWRITING and integrating screenshot data with existing workflows."""
        try:
            # Read and parse existing workflow data
            existing_workflows = self._parse_existing_workflows()
            
            # Process screenshot analyses and merge with existing data
            updated_workflows = self._merge_screenshot_data_with_existing(existing_workflows, analyses)
            
            # Generate updated workflow files
            self._overwrite_oats_workflows_file(updated_workflows)
            print(f"✅ OVERWRITTEN and UPDATED {self.workflows_file} with integrated screenshot data")
            
            # Generate analysis report
            self._generate_analysis_report(analyses)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating workflow files: {e}")
            return False
    
    def _parse_existing_workflows(self) -> Dict:
        """Parse existing workflows from the main workflow file."""
        workflows = {}
        
        if not os.path.exists(self.workflows_file):
            return workflows
        
        try:
            with open(self.workflows_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by flow sections
            sections = content.split('=== FLOW_ID:')
            
            for section in sections[1:]:  # Skip the first part (header)
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                # Extract flow ID
                flow_id = lines[0].split('===')[0].strip()
                
                # Initialize workflow data
                workflow = {
                    'flow_id': flow_id,
                    'title': '',
                    'keywords': [],
                    'description': '',
                    'steps': [],
                    'related_flows': []
                }
                
                # Parse workflow content
                current_section = None
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith('TITLE:'):
                        workflow['title'] = line.replace('TITLE:', '').strip()
                    elif line.startswith('KEYWORDS:'):
                        keywords_str = line.replace('KEYWORDS:', '').strip()
                        workflow['keywords'] = [k.strip() for k in keywords_str.split(',')]
                    elif line.startswith('DESCRIPTION:'):
                        workflow['description'] = line.replace('DESCRIPTION:', '').strip()
                    elif line.startswith('STEPS:'):
                        current_section = 'steps'
                    elif line.startswith('RELATED_FLOWS:'):
                        related_str = line.replace('RELATED_FLOWS:', '').strip()
                        workflow['related_flows'] = [r.strip() for r in related_str.split(',')]
                    elif current_section == 'steps' and line and not line.startswith('==='):
                        # Clean step text
                        step_text = line
                        if step_text.startswith(tuple('0123456789')):
                            # Remove step numbering
                            step_text = '. '.join(step_text.split('. ')[1:]) if '. ' in step_text else step_text
                        if step_text:
                            workflow['steps'].append(step_text)
                
                workflows[flow_id] = workflow
            
            print(f"✅ Parsed {len(workflows)} existing workflows")
            return workflows
            
        except Exception as e:
            logger.error(f"Error parsing existing workflows: {e}")
            return {}
    
    def _merge_screenshot_data_with_existing(self, existing_workflows: Dict, analyses: List[Dict]) -> Dict:
        """Merge screenshot analysis data with existing workflows."""
        updated_workflows = existing_workflows.copy()
        
        for analysis in analyses:
            if analysis.get('analysis_status') != 'success':
                continue
            
            workflow_data = analysis.get('workflow', {})
            if not workflow_data:
                continue
            
            # Generate flow ID from title
            title = workflow_data.get('title', '')
            if not title:
                continue
            
            flow_id = self._generate_flow_id(title)
            
            # Check if this workflow already exists (by title similarity or flow_id)
            existing_flow_id = self._find_matching_workflow(updated_workflows, title, flow_id)
            
            if existing_flow_id:
                # Update existing workflow with screenshot data
                print(f"📝 Updating existing workflow: {existing_flow_id}")
                updated_workflows[existing_flow_id] = self._enhance_workflow_with_screenshot_data(
                    updated_workflows[existing_flow_id], analysis
                )
            else:
                # Create new workflow from screenshot
                print(f"🆕 Creating new workflow: {flow_id}")
                updated_workflows[flow_id] = self._create_workflow_from_screenshot(analysis)
        
        return updated_workflows
    
    def _generate_flow_id(self, title: str) -> str:
        """Generate a clean flow ID from title."""
        return title.upper().replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '')
    
    def _find_matching_workflow(self, workflows: Dict, title: str, flow_id: str) -> Optional[str]:
        """Find if a workflow already exists with similar title or flow_id."""
        title_lower = title.lower()
        
        # Direct flow_id match
        if flow_id in workflows:
            return flow_id
        
        # Title similarity match
        for existing_id, workflow in workflows.items():
            existing_title = workflow.get('title', '').lower()
            
            # Check for significant overlap in title words
            title_words = set(title_lower.split())
            existing_words = set(existing_title.split())
            
            # If 70% of words match, consider it the same workflow
            if len(title_words & existing_words) / max(len(title_words), len(existing_words)) > 0.7:
                return existing_id
        
        return None
    
    def _enhance_workflow_with_screenshot_data(self, existing_workflow: Dict, screenshot_analysis: Dict) -> Dict:
        """Enhance existing workflow with screenshot data."""
        enhanced = existing_workflow.copy()
        
        workflow_data = screenshot_analysis.get('workflow', {})
        ui_elements = screenshot_analysis.get('ui_elements', {})
        
        # Update description if screenshot provides more detail
        screenshot_desc = workflow_data.get('description', '')
        if screenshot_desc and len(screenshot_desc) > len(enhanced.get('description', '')):
            enhanced['description'] = screenshot_desc
        
        # Merge steps - prioritize screenshot steps as they're more accurate
        screenshot_steps = workflow_data.get('steps', [])
        if screenshot_steps:
            enhanced['steps'] = screenshot_steps
        
        # Add screenshot-specific keywords
        screenshot_keywords = screenshot_analysis.get('keywords', [])
        if ui_elements.get('buttons'):
            for button in ui_elements['buttons']:
                if isinstance(button, dict) and 'text' in button:
                    screenshot_keywords.append(button['text'])
                elif isinstance(button, str):
                    screenshot_keywords.append(button)
        
        # Merge keywords without duplicates
        all_keywords = enhanced.get('keywords', []) + screenshot_keywords
        enhanced['keywords'] = list(dict.fromkeys([k for k in all_keywords if k]))[:15]  # Remove duplicates, limit to 15
        
        # Add screenshot metadata
        enhanced['screenshot_source'] = screenshot_analysis.get('screenshot_path', '')
        enhanced['last_updated_from_screenshot'] = True
        
        return enhanced
    
    def _create_workflow_from_screenshot(self, screenshot_analysis: Dict) -> Dict:
        """Create a new workflow from screenshot analysis."""
        workflow_data = screenshot_analysis.get('workflow', {})
        ui_elements = screenshot_analysis.get('ui_elements', {})
        
        # Generate keywords from various sources
        keywords = screenshot_analysis.get('keywords', [])
        if ui_elements.get('buttons'):
            for button in ui_elements['buttons']:
                if isinstance(button, dict) and 'text' in button:
                    keywords.append(button['text'])
                elif isinstance(button, str):
                    keywords.append(button)
        
        title = workflow_data.get('title', 'Screenshot Extracted Workflow')
        flow_id = self._generate_flow_id(title)
        
        return {
            'flow_id': flow_id,
            'title': title,
            'keywords': list(dict.fromkeys(keywords))[:15],  # Remove duplicates, limit to 15
            'description': workflow_data.get('description', 'Workflow extracted from OATS system screenshots'),
            'steps': workflow_data.get('steps', []),
            'related_flows': screenshot_analysis.get('related_flows', []),
            'screenshot_source': screenshot_analysis.get('screenshot_path', ''),
            'created_from_screenshot': True
        }
    
    def _overwrite_oats_workflows_file(self, workflows: Dict) -> bool:
        """Overwrite the main workflows file with updated data."""
        try:
            content = """# OATS System Workflows and Processes
# This file contains all step-by-step workflows for the OATS recruitment system
# Each workflow is clearly separated and labeled for easy AI analysis
# Updated with screenshot-extracted data for maximum accuracy

"""
            
            for flow_id, workflow in workflows.items():
                content += f"""=== FLOW_ID: {flow_id} ===
TITLE: {workflow.get('title', '')}
KEYWORDS: {', '.join(workflow.get('keywords', []))}
DESCRIPTION: {workflow.get('description', '')}

STEPS:
"""
                
                steps = workflow.get('steps', [])
                for i, step in enumerate(steps, 1):
                    content += f"{i}. {step}\n"
                
                if workflow.get('related_flows'):
                    content += f"\nRELATED_FLOWS: {', '.join(workflow['related_flows'])}\n"
                
                content += "\n"
            
            with open(self.workflows_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing workflows file: {e}")
            return False
    
    def _generate_analysis_report(self, analyses: List[Dict]) -> bool:
        """Generate a report of the analysis results."""
        try:
            from datetime import datetime
            
            report_path = "screenshot_analysis_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Screenshot Analysis Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                successful = [a for a in analyses if a.get('analysis_status') == 'success']
                partial = [a for a in analyses if a.get('analysis_status') == 'partial']
                failed = [a for a in analyses if a.get('analysis_status') == 'error']
                
                f.write(f"## Summary\n")
                f.write(f"- Total screenshots: {len(analyses)}\n")
                f.write(f"- Successful analyses: {len(successful)}\n")
                f.write(f"- Partial analyses: {len(partial)}\n")
                f.write(f"- Failed analyses: {len(failed)}\n\n")
                
                for analysis in analyses:
                    filename = analysis.get('screenshot_path', 'Unknown')
                    status = analysis.get('analysis_status', 'unknown')
                    f.write(f"### {os.path.basename(filename)}\n")
                    f.write(f"Status: {status}\n\n")
                    
                    if status == 'success':
                        workflow = analysis.get('workflow', {})
                        f.write(f"Workflow: {workflow.get('title', 'N/A')}\n")
                        f.write(f"Steps: {len(workflow.get('steps', []))}\n\n")
                    elif status == 'partial':
                        f.write("Raw analysis available but JSON parsing failed\n\n")
                    elif status == 'error':
                        f.write(f"Error: {analysis.get('error', 'Unknown')}\n\n")
            
            print(f"📄 Analysis report saved to: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return False
    
    def _generate_oats_workflows(self, analyses: List[Dict]) -> str:
        """Generate content for oats_workflows.txt based on analyses."""
        workflow_content = """# OATS System Workflows and Processes
# This file contains all step-by-step workflows for the OATS recruitment system
# Each workflow is clearly separated and labeled for easy AI analysis
# Updated based on actual website screenshots

"""
        
        workflow_id_counter = 1
        
        for analysis in analyses:
            if analysis.get('analysis_status') != 'success':
                continue
            
            workflow = analysis.get('workflow', {})
            if not workflow:
                continue
            
            # Generate workflow ID
            workflow_title = workflow.get('title', 'Unknown Workflow')
            workflow_id = workflow_title.upper().replace(' ', '_').replace('/', '_')
            
            # Generate keywords from analysis
            keywords = analysis.get('keywords', [])
            ui_elements = analysis.get('ui_elements', {})
            if ui_elements.get('buttons'):
                keywords.extend(ui_elements['buttons'])
            
            # Build workflow section
            workflow_section = f"""
=== FLOW_ID: {workflow_id} ===
TITLE: {workflow_title}
KEYWORDS: {', '.join(keywords[:15])}  # Limit to 15 keywords
DESCRIPTION: {workflow.get('description', 'Workflow extracted from OATS system screenshots')}
SCREENSHOT: {analysis.get('filename', 'N/A')}

STEPS:
"""
            
            # Add steps
            steps = workflow.get('steps', [])
            for i, step in enumerate(steps, 1):
                workflow_section += f"{i}. {step}\n"
            
            # Add additional information
            if workflow.get('prerequisites'):
                workflow_section += f"\nPREREQUISITES:\n"
                for prereq in workflow['prerequisites']:
                    workflow_section += f"- {prereq}\n"
            
            if analysis.get('validation_rules'):
                workflow_section += f"\nVALIDATION RULES:\n"
                for rule in analysis['validation_rules']:
                    workflow_section += f"- {rule}\n"
            
            if analysis.get('tips'):
                workflow_section += f"\nTIPS:\n"
                for tip in analysis['tips']:
                    workflow_section += f"- {tip}\n"
            
            related_flows = analysis.get('related_flows', [])
            if related_flows:
                workflow_section += f"\nRELATED_FLOWS: {', '.join(related_flows)}\n"
            
            workflow_content += workflow_section + "\n"
            workflow_id_counter += 1
        
        return workflow_content
    
    def _generate_additional_workflows(self, analyses: List[Dict]) -> str:
        """Generate content for additional_workflows_template.txt based on analyses."""
        content = """# Additional OATS Workflows Template
# Extended workflows and advanced procedures
# Updated based on actual website screenshots

"""
        
        for analysis in analyses:
            if analysis.get('analysis_status') != 'success':
                continue
            
            page_info = analysis.get('page_info', {})
            ui_elements = analysis.get('ui_elements', {})
            
            # Add detailed UI information
            content += f"""
## {page_info.get('title', 'Unknown Page')} - Advanced Guide

**Section:** {page_info.get('section', 'N/A')}
**Purpose:** {page_info.get('purpose', 'N/A')}
**Screenshot:** {analysis.get('filename', 'N/A')}

### User Interface Elements:
"""
            
            if ui_elements.get('buttons'):
                content += "**Buttons:** " + ", ".join(ui_elements['buttons']) + "\n"
            
            if ui_elements.get('fields'):
                content += "**Input Fields:** " + ", ".join(ui_elements['fields']) + "\n"
            
            if ui_elements.get('dropdowns'):
                content += "**Dropdowns:**\n"
                for dropdown, options in ui_elements['dropdowns'].items():
                    content += f"  - {dropdown}: {', '.join(options)}\n"
            
            if ui_elements.get('navigation'):
                content += "**Navigation:** " + ", ".join(ui_elements['navigation']) + "\n"
            
            content += "\n"
        
        return content
    
    def _update_oats_workflows_file(self, content: str):
        """Update the oats_workflows.txt file."""
        with open(self.workflows_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _update_additional_workflows_file(self, content: str):
        """Update the additional_workflows_template.txt file."""
        with open(self.additional_workflows_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_analysis_report(self, analyses: List[Dict]) -> str:
        """Generate a comprehensive analysis report."""
        report = f"""
# OATS Screenshot Analysis Report
Generated on: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}

## Summary
- Total screenshots processed: {len(analyses)}
- Successful analyses: {sum(1 for a in analyses if a.get('analysis_status') == 'success')}
- Partial analyses: {sum(1 for a in analyses if a.get('analysis_status') == 'partial')}
- Failed analyses: {sum(1 for a in analyses if a.get('analysis_status') == 'error')}

## Screenshots Analysis Details:
"""
        
        for analysis in analyses:
            report += f"""
### {analysis.get('filename', 'Unknown')}
- **Status:** {analysis.get('analysis_status', 'unknown')}
- **Page:** {analysis.get('page_info', {}).get('title', 'N/A')}
- **Purpose:** {analysis.get('page_info', {}).get('purpose', 'N/A')}
- **Keywords:** {', '.join(analysis.get('keywords', [])[:10])}
"""
            
            if analysis.get('analysis_status') == 'error':
                report += f"- **Error:** {analysis.get('error', 'Unknown error')}\n"
        
        return report

def main():
    """Main function to process screenshots and update workflow files."""
    print("🖼️  OATS Screenshot Processor")
    print("=" * 50)
    
    processor = OATSScreenshotProcessor()
    
    # Check if Gemini is available
    if not processor.gemini_model:
        print("❌ Gemini AI not available. Cannot process screenshots.")
        return
    
    # Process all screenshots
    print("📸 Processing screenshots...")
    analyses = processor.process_all_screenshots()
    
    if not analyses:
        print("⚠️  No screenshots found in flows/screenshots/ directory")
        print("💡 Add your OATS website screenshots to flows/screenshots/ and run again")
        return
    
    # Generate analysis report
    report = processor.generate_analysis_report(analyses)
    with open("screenshot_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📊 Analysis report saved: screenshot_analysis_report.md")
    
    # Update workflow files
    print("\n📝 Updating workflow files...")
    success = processor.update_workflow_files(analyses)
    
    if success:
        print("\n🎉 Screenshot processing completed successfully!")
        print("✅ Workflow files updated with real OATS system data")
        print("✅ Chatbot will now provide accurate step-by-step instructions")
    else:
        print("\n❌ Error updating workflow files")
    
    # Save processed analyses
    with open("screenshot_analyses.json", "w", encoding="utf-8") as f:
        json.dump(analyses, f, indent=2)
    print(f"💾 Raw analyses saved: screenshot_analyses.json")

if __name__ == "__main__":
    main()
