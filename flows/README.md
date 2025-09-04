# OATS Flow Management System

## 🚀 Overview

The OATS Flow Management System provides intelligent, step-by-step guidance for users working with the OATS recruitment platform. Using advanced AI analysis, it automatically detects when users are asking for process guidance and provides detailed workflows.

## 🎯 Features

- **Intelligent Query Detection**: Automatically identifies when users need workflow guidance
- **AI-Powered Flow Matching**: Uses Gemini AI to match user queries to relevant workflows
- **Comprehensive Workflows**: Detailed step-by-step processes for all OATS functions
- **Multiple Flow Support**: Can suggest multiple related workflows for complex queries
- **Conversational Integration**: Seamlessly integrated with the main chatbot functionality

## 📁 File Structure

```
flows/
├── oats_workflows.txt              # Main workflows file
├── additional_workflows_template.txt # Template for adding new flows
└── screenshots/                    # Directory for workflow screenshots (future use)
```

## 📝 How It Works

### 1. Query Analysis
When a user asks a question, the system:
- Detects if it's a workflow-related query using keywords and patterns
- Analyzes the query using Gemini AI to understand intent
- Matches the query to relevant workflows with confidence scoring

### 2. Flow Response Generation
The system then:
- Retrieves matching workflow(s) from the text database
- Formats the response with step-by-step guidance
- Provides related workflows and follow-up suggestions
- Integrates the response into conversation memory

### 3. Supported Query Types
- **How-to questions**: "How to create a job posting?"
- **Process inquiries**: "What's the interview scheduling process?"
- **Workflow requests**: "Show me candidate management workflow"
- **Step-by-step guides**: "Steps to generate reports"

## 🛠 Adding New Workflows

### Format Structure
Each workflow follows this exact format:

```
=== FLOW_ID: UNIQUE_FLOW_NAME ===
TITLE: Human-readable title
KEYWORDS: keyword1, keyword2, keyword3, related terms
DESCRIPTION: Brief description of what this workflow accomplishes

STEPS:
1. First step description
2. Second step description
3. And so on...

RELATED_FLOWS: OTHER_FLOW_ID1, OTHER_FLOW_ID2
```

### Guidelines for Creating Flows

1. **Flow ID**: Use UPPERCASE with underscores (e.g., `CREATE_JOB_POSTING`)
2. **Keywords**: Include all possible terms users might use
3. **Steps**: Be specific and actionable
4. **Related Flows**: Link to other relevant workflows

### Example Workflow

```
=== FLOW_ID: EXAMPLE_WORKFLOW ===
TITLE: How to Do Something Important
KEYWORDS: example, demo, sample, test workflow
DESCRIPTION: This is an example workflow showing the proper format

STEPS:
1. Navigate to the main section
2. Click the relevant button
3. Fill in the required information
4. Save your changes
5. Verify the results

RELATED_FLOWS: RELATED_WORKFLOW1, RELATED_WORKFLOW2
```

## 🔧 System Integration

### In OATSChatbot Class
The flow manager is initialized in the chatbot constructor:

```python
# Initialize Flow Management System
if FLOW_MANAGEMENT_ENABLED:
    self.flow_manager = OATSFlowManager(gemini_model=self.gemini_model)
```

### In Query Processing
Flow detection happens early in the query processing pipeline:

```python
# Check for workflow/process guidance queries
if self.flow_manager and self.flow_manager.is_flow_query(user_query):
    flow_ids, confidence = self.flow_manager.analyze_user_query(user_query)
    if flow_ids and confidence > 0.3:
        return self.flow_manager.generate_flow_response(user_query, flow_ids, confidence)
```

## 🌐 API Endpoints

### List All Flows
```
GET /api/flows
```
Returns all available workflows with metadata.

### Get Specific Flow
```
GET /api/flows/{flow_id}
```
Returns detailed information about a specific workflow.

### Test Interface
```
GET /test-flows
```
Interactive testing interface for the flow system.

## 🧪 Testing

### Using the Test Interface
1. Navigate to `/test-flows` in your browser
2. Use sample queries or create your own
3. View system status and available workflows
4. Test specific flow IDs directly

### Sample Test Queries
- "How to create a job posting?"
- "How do I search for candidates?"
- "Schedule interview process"
- "Client management workflow"
- "How to generate reports?"

## 📊 Performance and Monitoring

### Confidence Scoring
- Queries must have confidence > 0.3 to trigger flow responses
- Higher confidence (>0.7) indicates strong workflow matches
- Lower confidence may suggest multiple related workflows

### Fallback System
If Gemini AI is unavailable, the system uses keyword-based matching:
- Searches flow keywords, titles, and descriptions
- Scores matches based on keyword frequency
- Provides reasonable fallback responses

## 🔄 Maintenance

### Updating Workflows
1. Edit `flows/oats_workflows.txt` directly
2. Follow the exact format structure
3. Test new workflows using the test interface
4. The system automatically reloads flows on restart

### Best Practices
- Keep workflow IDs descriptive and consistent
- Include comprehensive keywords for better matching
- Write clear, actionable steps
- Link related workflows appropriately
- Test new workflows thoroughly

## 🚨 Troubleshooting

### Common Issues

1. **Flow not detected**: Check keywords and query phrasing
2. **Low confidence scores**: Add more relevant keywords
3. **System not responding**: Verify Gemini AI connection
4. **File not loading**: Check file format and syntax

### Debug Information
- Check server logs for flow detection messages
- Use `/test-flows` interface to verify flow loading
- Monitor confidence scores in query processing

## 💡 Future Enhancements

### Planned Features
- Screenshot integration for visual workflows
- Interactive step completion tracking
- User feedback on workflow helpfulness
- Dynamic workflow updates based on usage
- Multi-language workflow support

### Integration Opportunities
- Link workflows to actual OATS functions
- Provide contextual help within OATS interface
- Create workflow templates for common tasks
- Generate custom workflows based on user behavior

## 📞 Support

For issues with the flow system:
1. Check the test interface at `/test-flows`
2. Review server logs for error messages
3. Verify workflow file syntax and format
4. Test with sample queries first

The flow system is designed to be self-contained and easy to maintain, providing comprehensive guidance for all OATS recruitment processes.
