# Testing All Endpoints with Proper JSON Processing

## ✅ What's Been Updated

All endpoints now process JSON data in the same manner as candidates:

1. **Pagination**: All list endpoints use `PerPage=100`
2. **Total Count**: Correctly extracts `TotalItems` or `Count` fields
3. **Debug Logging**: Shows response structure for all types
4. **Data Processing**: Extracts key fields for each data type
5. **LLM Formatting**: Proper table display with relevant columns

## 🧪 Test Queries

### 1. **Candidates** (Already Working)
- Query: "list all candidates"
- Expected: "Total Candidates: 975,952" with table of 100 records

### 2. **Jobs**
- Query: "list all jobs"
- Expected: "Total Jobs: [X]" with table showing:
  - Job ID
  - Title
  - Client/Company
  - Location
  - Type
  - Status
  - Created Date

### 3. **Clients**
- Query: "list all clients"
- Expected: "Total Clients: [X]" with table showing:
  - Name
  - Company
  - Email
  - Phone
  - Location
  - Industry

### 4. **Vendors**
- Query: "list all vendors"
- Expected: "Total Vendors: [X]" with table showing:
  - Name
  - Company
  - Email
  - Phone
  - Location
  - Type

## 📊 What to Look for in Terminal

For each query, you should see:

```
🔍 DEBUG: [Type] API Response Structure:
   - TotalItems: [number]
   - Count: [number]
   - TotalPages: [number]
   - CurrentPage: 1
   - ItemsPerPage: 100
   - Data array length: 100
   - Using TotalItems for count: [number]
```

## 🔍 Count Queries

Test these to verify totals without tables:

1. "how many jobs do we have?"
2. "how many clients are there?"
3. "how many vendors do we have?"
4. "what's the total number of users?"

## 🔎 Search Queries

1. "search for active jobs"
2. "find clients in Mumbai"
3. "show me IT vendors"

## 📋 JSON API Testing

Use the `/api/query` endpoint for structured JSON:

```bash
POST http://localhost:5000/api/query
Content-Type: application/json

{
    "query": "list all jobs"
}
```

Response will include:
- `total_count`: Actual total from API
- `records_returned`: Number in current page
- `sample_records`: Processed records with key fields

## 🚀 Performance

- All endpoints now fetch 100 records (balanced between speed and data)
- Timeout protection prevents hanging
- Debug logs show exact data processing

## ⚠️ Common Issues

1. **If total shows only 10/100**: API might not have TotalItems field
2. **If timeout occurs**: API might be slow, try reducing PerPage
3. **If no data**: Check API endpoint URL and authentication
