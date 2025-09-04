# OATS Chatbot API Usage Guide

## Understanding the Data Flow

### 1. **Regular Chat Interface** (`/query`)
When you ask a question in the chat interface:
```
User Query → LLM selects endpoints → Fetch API data → Process data → LLM generates response → Display
```

### 2. **JSON API Endpoint** (`/api/query`)
For programmatic access with structured JSON output:
```
User Query → LLM selects endpoints → Fetch API data → Return structured JSON
```

## Key Information from Your Candidate API

Based on your API response, here's what we learned:

### Pagination Structure
```json
{
    "TotalPages": 97596,      // Total number of pages
    "TotalItems": 975952,     // Total candidates in database
    "CurrentPage": 1,         // Current page number
    "ItemsPerPage": 10,       // Default items per page
    "Data": [...]            // Array of candidate records
}
```

### API Limits
- **Default**: 10 records per page
- **With PerPage=1000000**: The API will return the maximum it allows
- **Actual Total**: 975,952 candidates in the system

## How the App Handles Large Data

### 1. **Data Processing**
- Extracts only key fields from each record
- Limits to 30 records for LLM processing
- Maintains total count information

### 2. **Smart Summarization**
The app processes candidate data to show:
- Name (FirstName + LastName)
- Current Job Title
- Location
- Experience (in years)
- Primary Skills (truncated if too long)
- Current Company
- Source

### 3. **Performance Optimization**
- Only sends summarized data to LLM (not full JSON)
- Caches responses in conversation history
- Uses direct routing for common queries

## Using the JSON API Endpoint

### Request:
```bash
POST http://localhost:5000/api/query
Content-Type: application/json

{
    "query": "list all candidates"
}
```

### Response:
```json
{
    "query": "list all candidates",
    "endpoints_used": ["candidates"],
    "data_summary": {
        "candidates": {
            "total_count": 975952,
            "current_page": 1,
            "items_per_page": 1000000,
            "total_pages": 1,
            "records_returned": 975952,
            "type": "candidates",
            "sample_records": [
                {
                    "id": "CID8175905",
                    "name": "raja doe",
                    "job_title": "Senior Developer",
                    "location": "Los Angeles, California, USA",
                    "experience": "5 years",
                    "skills": "Python, Django",
                    "company": "Tech Solutions",
                    "source": "LinkedIn"
                }
                // ... more records
            ]
        }
    }
}
```

## Example Queries

### 1. Count Queries
- "How many candidates do we have?"
- "What's the total number of jobs?"
- "Count all clients"

### 2. List Queries
- "List all candidates"
- "Show me all java developers"
- "Display all active jobs"

### 3. Search Queries
- "Find python developers"
- "Search for candidates in Mumbai"
- "Looking for senior developers"

## Tips for Better Performance

1. **Be Specific**: Instead of "show all data", ask "show all candidates" or "list all jobs"
2. **Use Keywords**: The app has direct routing for common terms like "candidates", "jobs", "clients"
3. **Pagination**: Even with PerPage=1000000, very large datasets might be paginated by the API

## API Response Fields

### Candidate Fields
- **Id**: Internal ID
- **CidId**: Candidate ID (e.g., "CID8175905")
- **FirstName**, **LastName**: Name fields
- **CurrentJobTitle**: Current position
- **Location**: Combined location string
- **ExperienceYears**: Years of experience
- **Skills**: All skills (can be very long)
- **PrimarySkills**: Key skills
- **CurrentCompany**: Current employer
- **Source**: How they were added (LinkedIn, system, etc.)
- **NoticePeriod**: Availability
- **Experiences**: Array of work history
- **EeoDetails**: Equal opportunity information
- **Languages**: Known languages

## Troubleshooting

### Issue: Only getting 10 records
**Solution**: The app now uses `PerPage=1000000` to request maximum data

### Issue: Response is too large
**Solution**: The app automatically:
- Limits display to 30 records
- Shows total count separately
- Provides summarized data

### Issue: Slow response
**Cause**: Processing 975,952 records takes time
**Solution**: Consider using search queries to filter data

## Advanced Usage

### Direct API Testing
You can test the OATS API directly:
```bash
GET https://dev.oats-backend.otomashen.com/candidate/get_candidate_list/filter_with_Paginator/?PerPage=100
Headers:
  Authorization: Bearer YOUR_ACCESS_TOKEN
  authtoken: YOUR_AUTH_TOKEN
```

### Custom Filtering
Add parameters to your queries:
- `?Search=java` - Search for Java developers
- `?Location=Mumbai` - Filter by location
- `?Experience=5` - Filter by experience

The chatbot will intelligently parse your natural language queries and apply appropriate filters!
