# OATS Flask Chatbot

A Flask web application version of the OATS (Otomashen ATS) AI-powered chatbot with intelligent endpoint selection.

## Features

- 🌐 **Web Interface**: Modern, responsive web UI for easy interaction
- 🧠 **AI-Powered**: Uses Google Gemini AI for intelligent endpoint selection
- 🔐 **Secure**: Session-based authentication with the OATS backend
- 📱 **Responsive**: Works on desktop, tablet, and mobile devices
- 🚀 **Real-time**: Instant responses with loading indicators
- 💡 **Smart**: Contextual example queries and suggestions

## Quick Start

### Method 1: Using the Runner Script (Recommended)

```bash
cd flask/
python run.py
```

The runner script will:
- Ask if you want to install/update requirements
- Start the Flask development server
- Show you the access URL

### Method 2: Manual Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your browser and go to: `http://localhost:5000`

## Usage

1. **Login**: Click the "Login to OATS System" button to authenticate
2. **Ask Questions**: Use natural language to ask about:
   - Job listings and searches
   - Candidate information
   - Dashboard analytics
   - Client and vendor data
   - Team performance metrics

### Example Queries

- "How many open positions do we have?"
- "Show me dashboard overview"
- "Find Java developers in Mumbai"
- "Tell me about job JID000122"
- "What are our top clients?"
- "Show me team performance metrics"

## API Endpoints

The Flask app provides the following REST API endpoints:

- `GET /` - Main web interface
- `POST /login` - Authenticate with OATS system
- `POST /logout` - End session
- `POST /query` - Send chatbot queries
- `GET /status` - Check system and login status

## Configuration

The chatbot uses the following default configuration:

- **Base URL**: `https://dev.oats-backend.otomashen.com`
- **Email**: `gaurav.int@otomashen.com`
- **Port**: `5000`
- **Session Timeout**: 2 hours

> ⚠️ **Security Note**: Change the `SECRET_KEY` in production!

## File Structure

```
flask/
├── app.py              # Main Flask application
├── run.py              # Convenient runner script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Web interface template
```

## Features Converted from CLI

✅ **Completed Conversions**:
- Interactive chatbot functionality → Web chat interface
- Login/logout system → Session-based authentication
- Query processing → RESTful API endpoints
- AI endpoint selection → Maintained with async support
- Error handling → JSON error responses
- Status checking → Real-time status API

## Dependencies

- **Flask**: Web framework
- **Flask-Session**: Session management
- **google-generativeai**: AI/LLM integration
- **requests**: HTTP client for API calls

## Development

To run in development mode with auto-reload:

```bash
export FLASK_ENV=development
python app.py
```

## Troubleshooting

1. **Import Errors**: Make sure all requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Connection Issues**: Check that the OATS backend is accessible

3. **Login Problems**: Verify credentials in the `app.py` file

4. **Port Conflicts**: Change the port in `app.py` if 5000 is already in use

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## Security Considerations

- Session data is stored server-side
- HTTPS recommended for production
- API credentials should be environment variables in production
- CORS headers may need configuration for cross-domain requests
