# 📸 OATS Screenshot Processor Guide

## 🎯 Purpose
This system analyzes screenshots of your OATS website and automatically extracts accurate workflow information to update your chatbot's knowledge base.

## 🚀 How to Use

### Step 1: Take Screenshots
Capture screenshots of key OATS system pages:

**Essential Screenshots:**
- Job creation page
- Job description form
- Candidate search page
- Client management
- Dashboard overview
- User management
- Any workflow you want the chatbot to guide users through

**Screenshot Tips:**
- Use clear, high-resolution screenshots
- Capture full page or complete forms
- Include navigation menus when visible
- Save with descriptive filenames (e.g., "job_creation_form.png", "candidate_search.jpg")

### Step 2: Add Screenshots
1. Place all screenshots in: `flows/screenshots/`
2. Supported formats: PNG, JPG, JPEG, BMP, GIF, WEBP
3. Use descriptive filenames for better analysis

### Step 3: Run the Processor
**Option A - Easy Way (Windows):**
```bash
double-click process_screenshots.bat
```

**Option B - Command Line:**
```bash
python screenshot_processor.py
```

### Step 4: Results
The processor will generate:
- ✅ **Updated `flows/oats_workflows.txt`** - Main workflow file with precise steps
- ✅ **Updated `flows/additional_workflows_template.txt`** - Extended workflow details
- 📊 **`screenshot_analysis_report.md`** - Analysis summary
- 💾 **`screenshot_analyses.json`** - Raw analysis data

## 🔧 What the AI Analyzes

For each screenshot, the system extracts:

### 📋 **Page Information**
- Page title and purpose
- OATS section/module
- Main functionality

### 🎨 **UI Elements**
- All buttons (exact text, colors)
- Input fields (labels, placeholders)
- Dropdown options
- Navigation elements

### 📝 **Workflow Steps**
- Detailed step-by-step instructions
- Required vs optional fields
- Validation rules
- Error handling

### 🔗 **Navigation Flow**
- How to reach the page
- Where each action leads
- Complete user journey

## 🎉 Benefits

### Before (Generic):
```
User: "How do I create a JD?"
Chatbot: "I can help with OATS data..."
```

### After (Screenshot-Based):
```
User: "How do I create a JD?"
Chatbot: "Here's how to create a Job Description in OATS:

1. Login to OATS system
2. Navigate to Jobs section from main menu
3. Click 'Create New Job' button (blue button, top right)
4. Fill in Job Title field (required)
5. Select Client from dropdown
6. Add Job Description in the text area
7. Set Salary Range (Min: $X, Max: $Y)
8. Click 'Save Draft' or 'Publish Job'
..."
```

## 📂 File Structure
```
newoatschatbot-main/
├── flows/
│   ├── screenshots/          ← Put your screenshots here
│   ├── oats_workflows.txt    ← Updated automatically
│   └── additional_workflows_template.txt ← Updated automatically
├── screenshot_processor.py   ← Main processor
├── process_screenshots.bat   ← Easy run script
└── screenshot_analysis_report.md ← Generated report
```

## 🔄 Updating Workflows

Whenever your OATS system changes:
1. Take new screenshots
2. Replace old screenshots in `flows/screenshots/`
3. Run the processor again
4. Your chatbot automatically gets updated knowledge!

## 💡 Tips for Best Results

### 📸 **Screenshot Quality:**
- Full page screenshots when possible
- Clear text and buttons
- Include any dropdown menus opened
- Capture error messages or validation

### 📝 **Filename Conventions:**
- `job_creation_step1.png`
- `candidate_search_filters.jpg`
- `dashboard_overview.png`
- `client_management_form.png`

### 🎯 **Focus Areas:**
- Forms and input fields
- Button locations and text
- Navigation menus
- Multi-step processes
- Confirmation pages

## 🤖 Integration with Chatbot

The processor automatically:
1. ✅ Updates workflow keywords for better query matching
2. ✅ Generates precise step-by-step instructions
3. ✅ Adds UI element details (button names, field labels)
4. ✅ Creates contextual help for each workflow
5. ✅ Links related workflows together

Your chatbot will now provide **exact, accurate guidance** based on the real OATS interface!

## 🆘 Troubleshooting

**No screenshots found:**
- Check `flows/screenshots/` directory exists
- Verify image file formats (PNG, JPG, etc.)

**Analysis failed:**
- Check internet connection (Gemini AI required)
- Verify screenshot quality and size
- Ensure screenshots show OATS interface

**Workflow not updating:**
- Check file permissions
- Verify Python can write to files directory
- Look at `screenshot_analysis_report.md` for details

## 🎊 Success!
Once processed, your chatbot will provide accurate, step-by-step guidance for any OATS workflow you've captured in screenshots!
