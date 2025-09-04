"""
OATS (Otomashen ATS) Job Management Chatbot - Flask Web Application
- Enhanced with AI-Driven Endpoint Selection
- Uses LLM to intelligently select the most relevant endpoints for user queries
- Implements fallback responses when data fetching fails    
- Built with comprehensive endpoint coverage for jobs, dashboard, candidates, and business operations
- Flask web interface for easy access and interaction
"""

import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import time
import asyncio
from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_session import Session
import os
from datetime import datetime, timedelta
import pytz
from dateutil import parser as date_parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory tracking (optional, non-intrusive)
try:
    from memory_tracker import initialize_memory_tracking, log_interaction
    MEMORY_TRACKING_ENABLED = True
    print("✅ Memory tracking enabled")
except ImportError:
    MEMORY_TRACKING_ENABLED = False
    print("⚠️ Memory tracking disabled (memory_tracker.py not found)")

# Flow Management System
try:
    from flow_manager import OATSFlowManager
    FLOW_MANAGEMENT_ENABLED = True
    print("✅ Flow management system enabled")
except ImportError:
    FLOW_MANAGEMENT_ENABLED = False
    print("⚠️ Flow management disabled (flow_manager.py not found)")

class CandidateSearchUtils:
    """Utility class for intelligent candidate search and matching"""
    
    @staticmethod
    def get_skill_variations(skill: str) -> List[str]:
        """Get variations and related terms for a skill"""
        skill_mappings = {
            'python': ['python', 'pyhton', 'py', 'python3', 'python2', 'django', 'flask', 'fastapi'],
            'java': ['java', 'j2ee', 'java ee', 'spring', 'spring boot', 'hibernate'],
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'react', 'angular', 'vue'],
            'data engineer': ['data engineer', 'data engineering', 'etl', 'data pipeline', 'big data', 'spark', 'hadoop', 'kafka'],
            'data scientist': ['data scientist', 'data science', 'machine learning', 'ml', 'ai', 'analytics', 'statistics'],
            'devops': ['devops', 'dev ops', 'deployment', 'ci/cd', 'docker', 'kubernetes', 'aws', 'azure'],
            'full stack': ['full stack', 'fullstack', 'full-stack', 'frontend', 'backend'],
            'frontend': ['frontend', 'front-end', 'ui', 'ux', 'react', 'angular', 'vue', 'html', 'css'],
            'backend': ['backend', 'back-end', 'api', 'rest', 'database', 'server'],
            'database': ['database', 'db', 'sql', 'mysql', 'postgresql', 'mongodb', 'oracle'],
            'cloud': ['cloud', 'aws', 'azure', 'gcp', 'google cloud', 'cloud computing']
        }
        
        skill_lower = skill.lower()
        
        # Check for exact matches first
        for key, variations in skill_mappings.items():
            if skill_lower in variations or key in skill_lower:
                return variations
        
        # If no mapping found, return the original skill
        return [skill_lower]
    
    @staticmethod
    def extract_skills_from_query(query: str) -> List[str]:
        """Extract skills mentioned in the user query"""
        query_lower = query.lower()
        extracted_skills = []
        
        # Common skill patterns
        skill_patterns = [
            'python', 'java', 'javascript', 'react', 'angular', 'node', 'vue',
            'data engineer', 'data scientist', 'machine learning', 'ai',
            'devops', 'full stack', 'frontend', 'backend', 'database',
            'aws', 'azure', 'docker', 'kubernetes', 'sql', 'mongodb'
        ]
        
        for skill in skill_patterns:
            if skill in query_lower:
                extracted_skills.extend(CandidateSearchUtils.get_skill_variations(skill))
        
        return list(set(extracted_skills))  # Remove duplicates
    
    @staticmethod
    def extract_role_from_query(query: str) -> Optional[str]:
        """Extract role/position mentioned in the user query"""
        query_lower = query.lower()
        
        role_patterns = {
            'data engineer': ['data engineer', 'data engineering'],
            'data scientist': ['data scientist', 'data science'],
            'python developer': ['python developer', 'python dev'],
            'java developer': ['java developer', 'java dev'],
            'full stack developer': ['full stack', 'fullstack', 'full-stack'],
            'frontend developer': ['frontend', 'front-end', 'ui developer'],
            'backend developer': ['backend', 'back-end', 'api developer'],
            'devops engineer': ['devops', 'dev ops'],
            'software engineer': ['software engineer', 'swe'],
            'developer': ['developer', 'programmer', 'engineer']
        }
        
        for role, patterns in role_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return role
        
        return None
    
    @staticmethod
    def score_candidate_for_skills(candidate: Dict, target_skills: List[str]) -> float:
        """Score a candidate based on skill match (0-100)"""
        if not target_skills:
            return 0.0
        
        candidate_skills = candidate.get('Skills', '').lower()
        primary_skills = candidate.get('PrimarySkills', '').lower()
        job_title = candidate.get('CurrentJobTitle', '').lower()
        
        # Combine all skill sources
        all_candidate_text = f"{candidate_skills} {primary_skills} {job_title}"
        
        matches = 0
        for skill in target_skills:
            if skill.lower() in all_candidate_text:
                matches += 1
        
        # Calculate percentage match
        return (matches / len(target_skills)) * 100
    
    @staticmethod
    def score_candidate_for_role(candidate: Dict, target_role: str) -> float:
        """Score a candidate based on role match (0-100)"""
        if not target_role:
            return 0.0
        
        job_title = candidate.get('CurrentJobTitle', '').lower()
        skills = candidate.get('Skills', '').lower()
        
        # Role-specific scoring
        role_keywords = CandidateSearchUtils.get_skill_variations(target_role)
        
        title_score = 0
        skills_score = 0
        
        # Higher weight for job title match
        for keyword in role_keywords:
            if keyword in job_title:
                title_score += 60  # Job title match is worth more
            if keyword in skills:
                skills_score += 30  # Skills match
        
        return min(100, title_score + skills_score)
    
    @staticmethod
    def rank_candidates(candidates: List[Dict], skills: List[str] = None, role: str = None) -> List[Dict]:
        """Rank candidates based on skills and role match"""
        scored_candidates = []
        
        for candidate in candidates:
            skill_score = CandidateSearchUtils.score_candidate_for_skills(candidate, skills) if skills else 0
            role_score = CandidateSearchUtils.score_candidate_for_role(candidate, role) if role else 0
            
            # Combined score with weights
            total_score = (skill_score * 0.7) + (role_score * 0.3) if skills and role else max(skill_score, role_score)
            
            candidate_copy = candidate.copy()
            candidate_copy['_match_score'] = total_score
            candidate_copy['_skill_score'] = skill_score
            candidate_copy['_role_score'] = role_score
            
            scored_candidates.append(candidate_copy)
        
        # Sort by score (highest first)
        return sorted(scored_candidates, key=lambda x: x['_match_score'], reverse=True)

class TimeUtils:
    """Utility class for handling time-based queries in IST timezone"""
    
    @staticmethod
    def get_ist_now():
        """Get current time in IST timezone"""
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist)
    
    @staticmethod
    def parse_api_date(date_string: str) -> datetime:
        """Parse API date string to datetime object in IST"""
        ist = pytz.timezone('Asia/Kolkata')
        
        # Handle various date formats from the API
        common_formats = [
            '%m/%d/%Y',       # 08/18/2025
            '%Y-%m-%d',       # 2025-08-18
            '%d/%b/%Y',       # 18/Aug/2025
            '%d/%m/%Y',       # 18/08/2025
            '%Y-%m-%dT%H:%M:%S',  # ISO format
            '%Y-%m-%d %H:%M:%S'   # SQL format
        ]
        
        for fmt in common_formats:
            try:
                parsed_date = datetime.strptime(date_string, fmt)
                return ist.localize(parsed_date) if parsed_date.tzinfo is None else parsed_date.astimezone(ist)
            except ValueError:
                continue
        
        # If no format matches, try dateutil parser as fallback
        try:
            parsed_date = date_parser.parse(date_string)
            return parsed_date.astimezone(ist) if parsed_date.tzinfo else ist.localize(parsed_date)
        except Exception as e:
            logger.warning(f"Could not parse date string '{date_string}': {e}")
            return None
    
    @staticmethod
    def get_time_period_range(period: str) -> Tuple[datetime, datetime]:
        """Get start and end datetime for a given time period in IST"""
        ist = pytz.timezone('Asia/Kolkata')
        now = TimeUtils.get_ist_now()
        
        if period == 'today':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif period == 'yesterday':
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif period == 'this_week':
            # Current week (Monday to Sunday)
            days_since_monday = now.weekday()
            monday = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
            start, end = monday, sunday
        
        elif period == 'last_week':
            # Previous week (Monday to Sunday)
            days_since_monday = now.weekday()
            this_monday = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            last_monday = this_monday - timedelta(days=7)
            last_sunday = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
            start, end = last_monday, last_sunday
        
        elif period == 'this_month':
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Last day of current month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_month = now.replace(month=now.month + 1, day=1)
            end = (next_month - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif period == 'last_month':
            # First day of current month
            first_day_current = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Last day of previous month
            end = (first_day_current - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            # First day of previous month
            start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        elif period == 'last_7_days':
            end = now
            start = now - timedelta(days=7)
        
        elif period == 'last_30_days':
            end = now
            start = now - timedelta(days=30)
        
        else:
            # Default to today
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return start, end
    
    @staticmethod
    def detect_time_period(query: str) -> Optional[str]:
        """Detect time period from user query"""
        query_lower = query.lower()
        
        time_patterns = {
            'today': ['today', 'this day'],
            'yesterday': ['yesterday'],
            'this_week': ['this week', 'current week'],
            'last_week': ['last week', 'previous week'],
            'this_month': ['this month', 'current month'],
            'last_month': ['last month', 'previous month'],
            'last_7_days': ['last 7 days', 'past 7 days', 'last seven days'],
            'last_30_days': ['last 30 days', 'past 30 days', 'last thirty days']
        }
        
        for period, patterns in time_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return period
        
        return None
    
    @staticmethod
    def filter_data_by_time_period(data: List[Dict], date_field: str, time_period: str) -> List[Dict]:
        """Filter list of data by time period based on date field"""
        if not data or not time_period:
            return data
        
        start_date, end_date = TimeUtils.get_time_period_range(time_period)
        filtered_data = []
        
        for item in data:
            date_value = item.get(date_field)
            if not date_value:
                continue
            
            parsed_date = TimeUtils.parse_api_date(str(date_value))
            if parsed_date and start_date <= parsed_date <= end_date:
                filtered_data.append(item)
        
        return filtered_data

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'oats-chatbot-secret-key-change-in-production'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
Session(app)


@dataclass
class APIEndpoint:
    """Data class to represent an API endpoint"""
    url: str
    description: str
    category: str = "general"

@dataclass
class APIResponse:
    """Data class to represent API response"""
    success: bool
    status_code: int
    data: Optional[Dict] = None
    error_message: Optional[str] = None
    endpoint_url: Optional[str] = None

class OATSChatbot:
    """Fixed OATS Chatbot with a new, intelligent query processing engine."""
    
    def __init__(self):
        # Configuration
        self.base_url = "https://dev.oats-backend.otomashen.com"
        self.email = "anamika.k@otomashen.com"
        self.password = "@anamika.123"
        self.login_url = f"{self.base_url}/rbca/token/"
        self.logout_url = f"{self.base_url}/login-api/logout/"
        self.gemini_api_key = "AIzaSyAsVkN0ygVBsl2tVAN_Dq5E0AY5aabyrqA"
        
        # Session setup
        self.session = self._create_session()
        self.access_token = None
        self.available_tokens = {}
        
        # Time-based query handling
        self.current_time_period = None
        
        # Candidate search handling
        self.current_search_skills = []
        self.current_search_role = None
        self.current_search_limit = 10
        self.fetch_resumes = False
        
        # Conversation Memory & Context (Enhanced for better context awareness)
        self.conversation_history = []
        self.conversation_data_cache = {}  # Store actual data from recent queries
        self.max_conversation_history = 30  # Store last 30 interactions
        self.user_context = {
            'last_query': None,
            'recent_data_types': [],
            'session_start': time.time()
        }
        
        # Setup Gemini AI
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
            logger.info("Gemini AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            self.gemini_model = None
        
        # Define available endpoints (now with templates)
        self.endpoints = self._get_available_endpoints()
        self.endpoint_descriptions = self._get_endpoint_descriptions()
        
        # Test Gemini connection
        self._test_gemini_connection()
        
        # Auto-login during initialization
        self._perform_auto_login()
        
        # Initialize memory tracking (non-intrusive)
        if MEMORY_TRACKING_ENABLED:
            try:
                initialize_memory_tracking(self)
                logger.info("Memory tracking initialized successfully")
            except Exception as e:
                logger.warning(f"Memory tracking initialization failed: {e}")
        
        # Initialize Flow Management System
        if FLOW_MANAGEMENT_ENABLED:
            try:
                self.flow_manager = OATSFlowManager(gemini_model=self.gemini_model)
                logger.info("Flow management system initialized successfully")
            except Exception as e:
                logger.warning(f"Flow management initialization failed: {e}")
                self.flow_manager = None
        else:
            self.flow_manager = None
    
    def add_to_conversation_memory(self, user_query: str, response_data: Dict = None, endpoints_used: List[str] = None, ai_response: str = None):
        """Add interaction to conversation memory with enhanced data storage."""
        interaction_id = len(self.conversation_history) + 1
        
        interaction = {
            "id": interaction_id,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query,
            "endpoints_used": endpoints_used or [],
            "data_types_returned": list(response_data.keys()) if response_data else [],
            "record_counts": {k: len(v.get('data', [])) if isinstance(v, dict) and 'data' in v else 0 
                             for k, v in (response_data or {}).items()},
            "ai_response_preview": ai_response[:200] + "..." if ai_response and len(ai_response) > 200 else ai_response,
            "ai_response_full": ai_response  # Store full response for memory monitoring
        }
        
        self.conversation_history.append(interaction)
        
        # Log interaction with memory tracker (non-intrusive)
        if MEMORY_TRACKING_ENABLED:
            try:
                log_interaction(user_query, response_data, endpoints_used, ai_response)
            except Exception as e:
                pass  # Silent fail to avoid disrupting chatbot operation
        
        # Store actual data in cache for context-aware responses
        if response_data:
            self.conversation_data_cache[interaction_id] = {
                'query': user_query,
                'timestamp': time.time(),
                'data': response_data,
                'endpoints': endpoints_used or []
            }
        
        # Keep only last 30 interactions for performance (as requested)
        if len(self.conversation_history) > self.max_conversation_history:
            # Remove the oldest interaction
            removed_interaction = self.conversation_history.pop(0)
            # Also remove its data from cache
            if removed_interaction['id'] in self.conversation_data_cache:
                del self.conversation_data_cache[removed_interaction['id']]
        
        # Keep data cache size manageable (store only last 10 interactions with data)
        if len(self.conversation_data_cache) > 10:
            oldest_cache_id = min(self.conversation_data_cache.keys())
            del self.conversation_data_cache[oldest_cache_id]
        
        # Update context
        self.user_context['last_query'] = user_query
        if response_data:
            self.user_context['recent_data_types'].extend(response_data.keys())
            # Keep only last 10 data types
            self.user_context['recent_data_types'] = self.user_context['recent_data_types'][-10:]
    
    def get_conversation_context(self) -> str:
        """Generate conversation context for LLM to understand user's session."""
        if not self.conversation_history:
            return "This is the start of the conversation."
        
        context_parts = []
        
        # Recent queries (last 3)
        recent_queries = [h['user_query'] for h in self.conversation_history[-3:]]
        if recent_queries:
            context_parts.append(f"Recent queries: {', '.join(recent_queries)}")
        
        # Data types user has accessed
        if self.user_context['recent_data_types']:
            unique_types = list(set(self.user_context['recent_data_types']))
            context_parts.append(f"User has accessed: {', '.join(unique_types)}")
        
        # Session duration
        session_duration = int((time.time() - self.user_context['session_start']) / 60)
        context_parts.append(f"Session duration: {session_duration} minutes")
        
        return " | ".join(context_parts)
    
    def search_conversation_data(self, search_term: str) -> Dict:
        """Search through cached conversation data for relevant information."""
        search_results = {
            'found_data': [],
            'matching_queries': [],
            'relevant_context': None,
            'specific_item': None
        }
        
        search_term_lower = search_term.lower()
        
        # Extract specific IDs from search term (like JID000123, CID000123, etc.)
        specific_id = None
        id_patterns = [r'jid\d+', r'cid\d+', r'uid\d+', r'vid\d+']
        for pattern in id_patterns:
            match = re.search(pattern, search_term_lower)
            if match:
                specific_id = match.group().upper()
                break
        
        # Search through cached data
        for interaction_id, cache_data in self.conversation_data_cache.items():
            query = cache_data['query'].lower()
            data = cache_data['data']
            
            # Check if search term appears in the previous query
            if search_term_lower in query:
                search_results['matching_queries'].append({
                    'id': interaction_id,
                    'query': cache_data['query'],
                    'timestamp': cache_data['timestamp']
                })
            
            # Search through the actual data for matches
            for endpoint_key, api_response in data.items():
                if hasattr(api_response, 'data') and api_response.data:
                    # Search through different data structures
                    data_list = None
                    if isinstance(api_response.data, dict):
                        if 'data' in api_response.data:
                            data_list = api_response.data['data']
                        elif 'Data' in api_response.data:
                            data_list = api_response.data['Data']
                        elif 'Clients' in api_response.data:
                            data_list = api_response.data['Clients']
                        elif 'Users' in api_response.data:
                            data_list = api_response.data['Users']
                    
                    if data_list and isinstance(data_list, list):
                        for item in data_list:
                            # First check for specific ID matches
                            if specific_id and self._item_has_specific_id(item, specific_id):
                                search_results['specific_item'] = {
                                    'interaction_id': interaction_id,
                                    'endpoint': endpoint_key,
                                    'item': item,
                                    'source_query': cache_data['query'],
                                    'matched_id': specific_id
                                }
                                search_results['found_data'].append(search_results['specific_item'])
                            elif self._item_matches_search_term(item, search_term_lower):
                                search_results['found_data'].append({
                                    'interaction_id': interaction_id,
                                    'endpoint': endpoint_key,
                                    'item': item,
                                    'source_query': cache_data['query']
                                })
        
        # Set relevant context if we found matches
        if search_results['specific_item']:
            search_results['relevant_context'] = f"Found specific item '{specific_id}' in recent conversation history from query: '{search_results['specific_item']['source_query']}'"
        elif search_results['found_data'] or search_results['matching_queries']:
            search_results['relevant_context'] = f"Found {len(search_results['found_data'])} data matches and {len(search_results['matching_queries'])} query matches for '{search_term}' in recent conversation history."
        
        return search_results
    
    def _item_matches_search_term(self, item: Dict, search_term: str) -> bool:
        """Check if a data item matches the search term."""
        if not isinstance(item, dict):
            return False
        
        # Convert all values to strings and search
        for key, value in item.items():
            if value and str(value).lower().find(search_term) != -1:
                return True
        
        return False
    
    def _item_has_specific_id(self, item: Dict, specific_id: str) -> bool:
        """Check if a data item contains the specific ID."""
        if not isinstance(item, dict):
            return False
        
        specific_id_upper = specific_id.upper()
        
        # Check common ID field names (both lowercase and uppercase versions)
        id_fields = ['id', 'job_code', 'JobCode', 'cid_id', 'client_id', 'user_id', 'vendor_id', 'uid', 'vid', 'Id', 'ID']
        
        for field in id_fields:
            if field in item and item[field]:
                if str(item[field]).upper() == specific_id_upper:
                    return True
        
        # Also check all fields for the ID pattern
        for key, value in item.items():
            if value and str(value).upper() == specific_id_upper:
                return True
        
        return False
    
    def store_chat_history(self, chat_entry: Dict):
        """Store chat in rolling buffer of 30 entries, maintaining exact UI format."""
        try:
            # Initialize chat history file if it doesn't exist
            chat_history_file = "chat_history.json"
            
            # Load existing chat history
            try:
                with open(chat_history_file, 'r', encoding='utf-8') as f:
                    chat_history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                chat_history = []
            
            # Add new chat entry
            chat_history.append(chat_entry)
            
            # Maintain rolling buffer of 30 entries
            if len(chat_history) > 30:
                chat_history = chat_history[-30:]  # Keep last 30 entries
            
            # Save back to file
            with open(chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Chat history stored successfully. Total entries: {len(chat_history)}")
            
        except Exception as e:
            logger.error(f"Error storing chat history: {e}")

    def get_chat_history(self) -> List[Dict]:
        """Retrieve chat history for memory monitor."""
        try:
            chat_history_file = "chat_history.json"
            with open(chat_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []
    
    def _generate_conversation_aware_response(self, user_query: str, conversation_search: Dict, search_id: str = None) -> str:
        """Generate response using conversation context for follow-up questions."""
        print(f"🧠 Generating conversation-aware response for: {user_query}")
        
        if not conversation_search['found_data']:
            return "I couldn't find any information about that in our recent conversation. Could you please be more specific or ask me to search for it again?"
        
        # Check if we found a specific item with exact ID match
        if conversation_search.get('specific_item'):
            return self._format_specific_item_response(conversation_search['specific_item'])
        
        # Find the most relevant data
        relevant_data = None
        source_query = None
        
        if search_id:
            # Look for specific ID matches
            for found_item in conversation_search['found_data']:
                item_str = str(found_item['item']).upper()
                if search_id in item_str:
                    relevant_data = found_item['item']
                    source_query = found_item['source_query']
                    break
        
        if not relevant_data and conversation_search['found_data']:
            # Use the first relevant item if no specific ID match
            relevant_data = conversation_search['found_data'][0]['item']
            source_query = conversation_search['found_data'][0]['source_query']
        
        if relevant_data:
            return self._format_item_details(relevant_data, user_query, source_query)
        
        return "I found some related information in our conversation, but couldn't extract specific details. Could you please rephrase your question?"
    
    def _format_specific_item_response(self, specific_item: Dict) -> str:
        """Format a detailed response for a specific item found in conversation history."""
        item = specific_item['item']
        matched_id = specific_item['matched_id']
        source_query = specific_item['source_query']
        
        # Determine item type based on the ID pattern
        if matched_id.startswith('JID'):
            return self._format_job_details(item, matched_id, source_query)
        elif matched_id.startswith('CID'):
            return self._format_candidate_details(item, matched_id, source_query)
        elif matched_id.startswith('UID'):
            return self._format_user_details(item, matched_id, source_query)
        elif matched_id.startswith('VID'):
            return self._format_vendor_details(item, matched_id, source_query)
        else:
            return self._format_generic_details(item, matched_id, source_query)
    
    def _format_job_details(self, job_item: Dict, job_id: str, source_query: str) -> str:
        """Format detailed job information from conversation history."""
        html_response = f"<div class='ai-response'>"
        html_response += f"<strong>📋 Job Details for {job_id}</strong><br>"
        html_response += f"<em>Retrieved from previous query: \"{source_query}\"</em><br><br>"
        
        html_response += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        
        # Map job fields to display names - comprehensive mapping
        field_mappings = {
            'job_code': 'Job ID',
            'JobCode': 'Job ID',
            'job_title': 'Title',
            'JobTitle': 'Title',
            'client': 'Company',
            'Client': 'Company',
            'location': 'Location',
            'Location': 'Location',
            'job_status': 'Status',
            'JobStatus': 'Status',
            'primary_skills': 'Primary Skills',
            'PrimarySkills': 'Primary Skills',
            'experience_range': 'Experience Range',
            'Experience': 'Experience Range',
            'created_at': 'Created Date',
            'CreatedAt': 'Created Date',
            'modified_at': 'Modified Date',
            'ModifiedAt': 'Modified Date',
            'client_bill_rate__value': 'Bill Rate',
            'ClientBillRate': 'Bill Rate',
            'pay_rates__min_salary': 'Min Salary',
            'pay_rates__max_salary': 'Max Salary',
            'PayRate': 'Pay Rate',
            'job_created_by': 'Created By',
            'JobCreatedBy': 'Created By',
            'skills': 'Skills',
            'requirements': 'Requirements',
            'description': 'Description',
            'Department': 'Department',
            'department': 'Department',
            'employment_type': 'Employment Type',
            'JobType': 'Job Type',
            'salary': 'Salary',
            'currency': 'Currency',
            'remote': 'Remote Work',
            'visa_sponsorship': 'Visa Sponsorship',
            'BusinessUnit': 'Business Unit',
            'ClientID': 'Client ID',
            'Industry': 'Industry',
            'Country': 'Country',
            'State': 'State',
            'City': 'City',
            'JobPreferences': 'Job Preferences',
            'WorkAuthorization': 'Work Authorization',
            'NumberOfPositions': 'Number of Positions'
        }
        
        for field, value in job_item.items():
            if value is not None and value != '':
                # Use mapped name if available, otherwise format the field name
                if field in field_mappings:
                    display_name = field_mappings[field]
                else:
                    # Format field name nicely (convert snake_case to Title Case)
                    display_name = field.replace('_', ' ').replace('__', ' - ').title()
                
                html_response += f"<tr><td style='padding:8px; font-weight:bold; border-bottom:1px solid #eee;'>{display_name}:</td>"
                html_response += f"<td style='padding:8px; border-bottom:1px solid #eee;'>{value}</td></tr>"
        
        html_response += "</table><br>"
        html_response += f"💡 <em>This information was retrieved from our recent conversation where you asked: \"{source_query}\"</em>"
        html_response += "</div>"
        
        return html_response
    
    def _format_candidate_details(self, candidate_item: Dict, candidate_id: str, source_query: str) -> str:
        """Format detailed candidate information from conversation history."""
        html_response = f"<div class='ai-response'>"
        html_response += f"<strong>👤 Candidate Details for {candidate_id}</strong><br>"
        html_response += f"<em>Retrieved from previous query: \"{source_query}\"</em><br><br>"
        
        # Format candidate details similarly to job details
        field_mappings = {
            'cid_id': 'Candidate ID',
            'first_name': 'Name',
            'last_name': 'Last Name',
            'current_job_title': 'Current Title',
            'city': 'Location',
            'source': 'Source',
            'notice_period': 'Notice Period',
            'relocation': 'Relocation',
            'created_by': 'Created By'
        }
        
        html_response += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        for field, value in candidate_item.items():
            if value is not None and value != '':
                # Use mapped name if available, otherwise format the field name
                if field in field_mappings:
                    display_name = field_mappings[field]
                else:
                    display_name = field.replace('_', ' ').replace('__', ' - ').title()
                
                html_response += f"<tr><td style='padding:8px; font-weight:bold; border-bottom:1px solid #eee;'>{display_name}:</td>"
                html_response += f"<td style='padding:8px; border-bottom:1px solid #eee;'>{value}</td></tr>"
        
        html_response += "</table><br>"
        html_response += f"💡 <em>This information was retrieved from our recent conversation.</em>"
        html_response += "</div>"
        
        return html_response
    
    def _format_user_details(self, user_item: Dict, user_id: str, source_query: str) -> str:
        """Format detailed user/team member information from conversation history."""
        html_response = f"<div class='ai-response'>"
        html_response += f"<strong>👨‍💼 Team Member Details for {user_id}</strong><br>"
        html_response += f"<em>Retrieved from previous query: \"{source_query}\"</em><br><br>"
        
        # Format user details
        field_mappings = {
            'id': 'User ID',
            'name': 'Name',
            'email': 'Email',
            'role': 'Role',
            'department': 'Department',
            'created_at': 'Created Date'
        }
        
        html_response += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        for field, value in user_item.items():
            if value is not None and value != '':
                # Use mapped name if available, otherwise format the field name
                if field in field_mappings:
                    display_name = field_mappings[field]
                else:
                    display_name = field.replace('_', ' ').replace('__', ' - ').title()
                
                html_response += f"<tr><td style='padding:8px; font-weight:bold; border-bottom:1px solid #eee;'>{display_name}:</td>"
                html_response += f"<td style='padding:8px; border-bottom:1px solid #eee;'>{value}</td></tr>"
        
        html_response += "</table><br>"
        html_response += f"💡 <em>This information was retrieved from our recent conversation.</em>"
        html_response += "</div>"
        
        return html_response
    
    def _format_vendor_details(self, vendor_item: Dict, vendor_id: str, source_query: str) -> str:
        """Format detailed vendor information from conversation history."""
        html_response = f"<div class='ai-response'>"
        html_response += f"<strong>🏢 Vendor Details for {vendor_id}</strong><br>"
        html_response += f"<em>Retrieved from previous query: \"{source_query}\"</em><br><br>"
        
        # Format vendor details
        html_response += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        for field, value in vendor_item.items():
            if value is not None and value != '':
                display_name = field.replace('_', ' ').title()
                html_response += f"<tr><td style='padding:8px; font-weight:bold; border-bottom:1px solid #eee;'>{display_name}:</td>"
                html_response += f"<td style='padding:8px; border-bottom:1px solid #eee;'>{value}</td></tr>"
        
        html_response += "</table><br>"
        html_response += f"💡 <em>This information was retrieved from our recent conversation.</em>"
        html_response += "</div>"
        
        return html_response
    
    def _format_generic_details(self, item: Dict, item_id: str, source_query: str) -> str:
        """Format generic item details from conversation history."""
        html_response = f"<div class='ai-response'>"
        html_response += f"<strong>📄 Details for {item_id}</strong><br>"
        html_response += f"<em>Retrieved from previous query: \"{source_query}\"</em><br><br>"
        
        html_response += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
        for field, value in item.items():
            if value is not None and value != '':
                display_name = field.replace('_', ' ').title()
                html_response += f"<tr><td style='padding:8px; font-weight:bold; border-bottom:1px solid #eee;'>{display_name}:</td>"
                html_response += f"<td style='padding:8px; border-bottom:1px solid #eee;'>{value}</td></tr>"
        
        html_response += "</table><br>"
        html_response += f"💡 <em>This information was retrieved from our recent conversation.</em>"
        html_response += "</div>"
        
        return html_response
    
    def _format_item_details(self, relevant_data: Dict, user_query: str, source_query: str) -> str:
        """Format item details for general cases."""
        if not relevant_data:
            return "I found some related information in our conversation, but I couldn't match it to your specific request. Could you please clarify what you're looking for?"
        
        # Generate detailed response using the found data
        if self.gemini_model:
            try:
                prompt = f"""
You are a friendly and helpful OATS recruitment system assistant. The user is asking a follow-up question about data from a previous query.

CURRENT USER QUERY: "{user_query}"
PREVIOUS QUERY CONTEXT: "{source_query}"

RELEVANT DATA FOUND:
{json.dumps(relevant_data, indent=2)}

CONVERSATIONAL INSTRUCTIONS:
1. Start with appreciation: "Great question!" or "I'd be happy to help with that!"
2. Provide a natural, conversational response that directly answers the user's question
3. Reference that this information was from their previous query about "{source_query}" in a friendly way
4. Present the information in a clear, detailed format with enthusiasm
5. If it's job data, show job details like title, company, location, status, requirements, etc.
6. If it's candidate data, show candidate details like name, skills, experience, contact info, etc.
7. If it's client data, show client details like name, contact info, business details, etc.
8. End with a helpful follow-up question like:
   - "Would you like me to show you more details about this?"
   - "Are you interested in exploring similar [jobs/candidates/clients]?"
   - "Should I help you search for more information related to this?"

Format the response as conversational text with key details highlighted and end with a relevant question.
"""
                
                response = self.gemini_model.generate_content(prompt)
                return f"<div class='ai-response'>{response.text.strip()}</div>"
            except Exception as e:
                logger.error(f"Error generating conversation-aware response: {e}")
        
        # Fallback response
        return f"""<div class='ai-response'>Based on your previous query "{source_query}", here are the details I found:

<pre>{json.dumps(relevant_data, indent=2)}</pre>

Let me know if you need any additional information or have other questions!</div>"""

    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        })
        return session

    def _test_gemini_connection(self):
        """Test if Gemini AI is working properly"""
        if not self.gemini_model:
            logger.warning("Gemini AI model not initialized - will use fallback responses")
            return False
        
        try:
            # Simple test query
            test_response = self.gemini_model.generate_content("Hello, respond with just 'OK'")
            if test_response and test_response.text:
                logger.info("Gemini AI connection test successful")
                return True
            else:
                logger.warning("Gemini AI test failed - no response received")
                self.gemini_model = None
                return False
        except Exception as e:
            logger.error(f"Gemini AI connection test failed: {e}")
            self.gemini_model = None
            return False

    def _perform_auto_login(self):
        """Automatically login during initialization for seamless functionality."""
        print("🔐 Performing automatic login...")
        try:
            if self.login():
                print("   ✅ Auto-login successful!")
                return True
            else:
                print("   ⚠️ Auto-login failed - manual login required")
                return False
        except Exception as e:
            print(f"   ❌ Auto-login error: {e}")
            return False

    def login(self):
        print("🔐 Logging in...")
        payload = {"email": self.email, "password": self.password}
        try:
            response = self.session.post(self.login_url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Login successful (HTTP 200)")
                self.access_token = self._extract_token(data)
                if self.access_token:
                    print("   🔑 Token extracted successfully.")
                    return True
            print(f"   ❌ Login failed (HTTP {response.status_code})")
            return False
        except Exception as e:
            print(f"   ❌ Exception during login: {e}")
            return False

    def _extract_token(self, response_data):
        self.available_tokens = {}
        access_token = response_data.get("Tokens", {}).get("access")
        authtoken = response_data.get("Authtoken")
        if access_token: self.available_tokens['access'] = access_token
        if authtoken: self.available_tokens['authtoken'] = authtoken
        return access_token

    def logout(self):
        if not self.access_token: return
        print("🔓 Logging out...")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            self.session.post(self.logout_url, headers=headers, timeout=5)
        finally:
            self.access_token = None
            self.available_tokens = {}
            print("   ✅ Session cleaned up.")

    def _get_available_endpoints(self) -> Dict[str, APIEndpoint]:
        """Define available API endpoints - proven working versions from chatbotv5v1."""
        return {
            # --- JOB ENDPOINTS (VERIFIED WORKING) ---
            "jobs": APIEndpoint(
                url=f"{self.base_url}/jobs/get-job-list-filter-with-Paginator/?page=1&per_page=100&IncludeFields[]=job_code,job_title,client,job_status,client_bill_rate__value,pay_rates__min_salary,pay_rates__max_salary,location,job_created_by,created_at,modified_at&Search=",
                description="Get job listings with filters",
                category="job"
            ),
            "job_search": APIEndpoint(
                url=f"{self.base_url}/jobs/get-job-list-filter-with-Paginator/?page=1&per_page=10&IncludeFields[]=job_code,job_title,client,job_status,client_bill_rate__value,pay_rates__min_salary,pay_rates__max_salary,location,job_created_by,created_at,modified_at&Search={{search_term}}",
                description="Search for jobs using a filter term.",
                category="job_search"
            ),
            
            # --- DASHBOARD ENDPOINTS ---
            "dashboard_overview": APIEndpoint(
                url=f"{self.base_url}/dashboard/get-dashboard-data/",
                description="Get comprehensive dashboard overview with total counts, upcoming interviews, team performance, and hiring progress.",
                category="dashboard"
            ),
            
            # --- CANDIDATE ENDPOINTS (WORKING) ---
            "candidates": APIEndpoint(
                url=f"{self.base_url}/candidate/get_candidate_list/filter_with_Paginator/?PerPage=10",
                description="Get candidate listings (10 per page, but shows total count)",
                category="candidate"
            ),
            "candidate_details": APIEndpoint(
                url=f"{self.base_url}/candidate/get/personal-detail-list/?CidId={{CidId}}",
                description="Get all details for a specific candidate by their ID.",
                category="candidate_specific"
            ),
            "candidate_search": APIEndpoint(
                url=f"{self.base_url}/candidate/get_candidate_list/filter_with_Paginator/?Page=1&PerPage={{per_page}}&IncludeFields[]=cid_id,first_name,current_job_title,city,source,notice_period,relocation,created_by&Search={{search_term}}",
                description="Search for candidates using filters with dynamic page size and search terms.",
                category="candidate_search"
            ),
            "candidate_resume": APIEndpoint(
                url=f"{self.base_url}/candidate/get/personal-detail-list/?CidId={{CidId}}",
                description="Get detailed candidate information including resume data by CID.",
                category="candidate_resume"
            ),
            
            # --- CLIENT ENDPOINTS ---
            "clients": APIEndpoint(
                url=f"{self.base_url}/client/get-client-list-filter-with-Paginator/?Page=1&PerPage=10&IncludeFields[]=id,name,contact_number,email,website,status,primary_owner,created_by&Search=",
                description="Get list of all clients with contact information",
                category="client"
            ),
            "client_search": APIEndpoint(
                url=f"{self.base_url}/client/get-client-list-filter-with-Paginator/?Page=1&PerPage={{{{per_page}}}}&IncludeFields[]=id,name,contact_number,email,website,status,primary_owner,created_by&Search={{{{search_term}}}}",
                description="Search for specific clients by name or company",
                category="client_search"
            ),
            
            # --- VENDOR ENDPOINTS ---
            "vendors": APIEndpoint(
                url=f"{self.base_url}/vendor/get-vendor-list-filter/?IncludeFields[]=vendor_id,vendor_name,federal_id,contact_number,website,address,ownership,created_by,modify_by&Search=&Page=1&PerPage=10",
                description="Get list of all vendors with contact and business information",
                category="vendor"
            ),
            "vendor_search": APIEndpoint(
                url=f"{self.base_url}/vendor/get-vendor-list-filter/?IncludeFields[]=vendor_id,vendor_name,federal_id,contact_number,website,address,ownership,created_by,modify_by&Search={{{{search_term}}}}&Page=1&PerPage={{{{per_page}}}}",
                description="Search for specific vendors by name or company",
                category="vendor_search"
            ),

            "client_inputs": APIEndpoint(
                url=f"{self.base_url}/client/get-client-inputs/?field={{field}}",
                description="Get client input data for specific fields.",
                category="client_specific"
            ),
            
            # --- VENDOR ENDPOINTS (WORKING) ---
            "vendors": APIEndpoint(
                url=f"{self.base_url}/vendor/get-vendor-list-filter/",
                description="Get vendor listings",
                category="vendor"
            ),
            
            # --- RBAC/USER ENDPOINTS ---
            "users": APIEndpoint(f"{self.base_url}/rbca/get-users/", "Get user listings", "user"),
            "current_business_unit": APIEndpoint(
                url=f"{self.base_url}/rbca/get-current-business-unit/",
                description="Get current business unit information.",
                category="rbca"
            ),
            "role_list": APIEndpoint(
                url=f"{self.base_url}/rbca/role-list/",
                description="Get list of all roles.",
                category="rbca"
            ),
            "role_permissions": APIEndpoint(
                url=f"{self.base_url}/rbca/role-permissions/?RoleId={{role_id}}",
                description="Get permissions for a specific role.",
                category="rbca_specific"
            ),
            "user_notifications": APIEndpoint(
                url=f"{self.base_url}/rbca/user-notify/",
                description="Get user notifications.",
                category="rbca"
            ),
            "hierarchy_users": APIEndpoint(
                url=f"{self.base_url}/rbca/hierarchy-users/",
                description="Get hierarchy of users.",
                category="rbca"
            ),
            
            # --- ADDITIONAL ENDPOINTS ---
            "recruiter_profile": APIEndpoint(
                url=f"{self.base_url}/recuriter/profile",
                description="Get recruiter profile information.",
                category="profile"
            ),
            "location_info": APIEndpoint(
                url=f"{self.base_url}/accounts/location-info/",
                description="Get location information.",
                category="location"
            )
        }
    
    def _get_endpoint_descriptions(self) -> Dict[str, str]:
        """Get conversational descriptions of all available endpoints for LLM endpoint selection."""
        return {
            # Job Endpoints - More Conversational
            "jobs": "Show me all available jobs and positions - perfect when someone asks 'what jobs do we have?', 'show me open positions', or wants to browse through job listings",
            "job_search": "Find specific jobs based on skills, titles, or requirements - use when someone asks 'find Java developer jobs', 'search for remote positions', or 'jobs in Mumbai'",
            
            # Dashboard Endpoints - More Conversational
            "dashboard_overview": "Show me the big picture - use for 'how are we doing?', 'dashboard summary', 'overall performance', 'show me dashboard', 'give me an overview', 'what's our status?', 'how many jobs do we have?', 'how many candidates?', 'show me hiring progress', 'team performance', 'upcoming interviews', 'offer trends', 'applicant pipeline'",
            "dashboard_candidate_source": "Where are our best candidates coming from? - use for 'which job boards work best?', 'candidate source analysis', or 'recruitment channel effectiveness'",
            "dashboard_jobs_by_skills": "What skills are in demand? - use for 'trending technologies', 'what skills do clients want?', or 'most requested skills'",
            "dashboard_team_report": "How is our team performing? - use for 'team productivity', 'who's doing well?', or 'team performance metrics'",
            "dashboard_client_scorecard": "How happy are our clients? - use for 'client satisfaction', 'client performance', or 'client feedback'",
            "dashboard_confirmations": "How many offers are being accepted? - use for 'acceptance rates', 'offer confirmations', or 'hiring success'",
            "dashboard_applicant_status": "What's happening with applicants for a specific client? - use for client-specific tracking or 'how are client X's candidates doing?'",
            "dashboard_inputs": "Get configuration data for dashboards - use for technical dashboard setup or field information",
            
            # Candidate Endpoints - More Conversational
            "candidates": "Show me all our talent pool - use for 'list all candidates', 'show me candidates', or 'who's in our database?'",
            "candidate_details": "Tell me everything about a specific person - use when someone mentions 'CID12345' or 'details about this candidate'",
            "candidate_search": "Find the perfect candidate for my needs - use for 'find Python developers', 'candidates in Bangalore', or 'experienced marketers'",
            
            # Client Endpoints - More Conversational
            "clients": "Show me our client companies - use for 'who are our clients?', 'list all companies', or 'show me client information'",
            "client_inputs": "Get client configuration data - use for technical client setup or field information",
            
            # Vendor Endpoints - More Conversational
            "vendors": "Show me our vendor partners - use for 'who do we work with?', 'list suppliers', or 'vendor information'",
            
            # User Management Endpoints - More Conversational
            "users": "Show me our team members - use for 'who works here?', 'list all users', or 'team directory'",
            "current_business_unit": "What business unit am I in? - use for 'my organization', 'current unit', or organizational context",
            "role_list": "What roles exist in our system? - use for 'available roles', 'permission levels', or 'access types'",
            "role_permissions": "What can this role do? - use for 'permissions for role X', 'access rights', or 'what can managers do?'",
            "user_notifications": "Any messages for me? - use for 'notifications', 'alerts', or 'what's new?'",
            "hierarchy_users": "Who reports to whom? - use for 'organizational chart', 'team structure', or 'reporting hierarchy'",
            
            # Additional Endpoints - More Conversational
            "recruiter_profile": "Tell me about this recruiter - use for recruiter information, profiles, or 'who is this person?'",
            "location_info": "Where are we located? - use for office locations, geographical data, or 'where do we operate?'"
        }
    
    async def get_relevant_endpoints(self, user_query: str) -> List[str]:
        """Use LLM to determine which endpoints are most relevant for the user query with conversation context."""
        if not self.gemini_model:
            logger.warning("Gemini AI not available, falling back to keyword matching")
            return self._fallback_endpoint_selection(user_query)
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        prompt = f"""
You are an AI assistant for OATS recruitment system. Select endpoints to answer the user's query.

USER QUERY: "{user_query}"

AVAILABLE ENDPOINTS:
{json.dumps(self.endpoint_descriptions, indent=2)}

RULES:
- Select 1-2 most relevant endpoints only
- For "how many" questions about candidates/jobs/clients, use the list endpoint
- For searches (find, search for), use the search endpoint
- For specific IDs, this is handled separately
- For general queries, dashboard queries, or testing, use "dashboard_overview"
- IMPORTANT: 
  * For job searches (jobs, positions, openings), use "job_search"
  * For candidate searches (people, talent, applicants), use "candidate_search"
  * For test queries, general questions, or anything unclear, use "dashboard_overview"
  * For person name queries (like "John Doe", "who is John Doe", "what is John Doe"), use "candidate_search"

EXAMPLES:
- "how many candidates do we have" → ["candidates"]
- "find java developers" → ["candidate_search"]
- "find java developer jobs" → ["job_search"]
- "search for python developer positions" → ["job_search"]
- "show me all java developer" → ["candidate_search"]
- "java developer jobs" → ["job_search"]
- "data engineer" → ["candidate_search"]
- "data engineer jobs" → ["job_search"]
- "show me all jobs" → ["jobs"]
- "list our clients" → ["clients"]
- "search for data engineer" → ["candidate_search"]
- "test unified JSON handling" → ["dashboard_overview"]
- "test something" → ["dashboard_overview"]
- "how are we doing" → ["dashboard_overview"]
- "yash sharma" → ["candidate_search"]
- "who is yash sharma" → ["candidate_search"]
- "what is john doe" → ["candidate_search"]
- "tell me about jane smith" → ["candidate_search"]
- "show me sarah wilson" → ["candidate_search"]
- "find mike johnson" → ["candidate_search"]

RESPONSE FORMAT: Return ONLY a valid JSON array with endpoint names in quotes.
Example: ["dashboard_overview"]
Example: ["job_search", "candidates"]

Do NOT include any explanation, just the JSON array.
"""

        try:
            result = self.gemini_model.generate_content(prompt)
            response_text = result.text.strip()
            
            # Log the raw response for debugging
            logger.info(f"LLM raw response: {response_text}")
            
            # Try multiple parsing strategies
            endpoint_keys = None
            
            # Strategy 1: Direct JSON parsing if response starts with [
            if response_text.startswith('[') and response_text.endswith(']'):
                try:
                    endpoint_keys = json.loads(response_text)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Extract JSON from markdown or mixed text
            if endpoint_keys is None:
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        endpoint_keys = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Extract quoted strings if JSON parsing fails
            if endpoint_keys is None:
                quoted_matches = re.findall(r'"([^"]+)"', response_text)
                if quoted_matches:
                    endpoint_keys = quoted_matches
            
            # Validate and filter endpoints
            if endpoint_keys and isinstance(endpoint_keys, list):
                # Validate that all keys exist in our endpoints
                valid_keys = [key for key in endpoint_keys if key in self.endpoints]
                if valid_keys:
                    logger.info(f"LLM selected endpoints: {valid_keys}")
                    return valid_keys
                else:
                    logger.warning(f"LLM returned invalid endpoint keys: {endpoint_keys}")
            else:
                logger.warning(f"LLM response could not be parsed as endpoint list: {response_text}")
            
            logger.warning("LLM returned invalid endpoint selection, falling back to keyword matching")
            return self._fallback_endpoint_selection(user_query)
            
        except Exception as e:
            logger.error(f"Error in LLM endpoint selection: {e}")
            return self._fallback_endpoint_selection(user_query)
    
    def _detect_person_name_query(self, user_query: str) -> bool:
        """Detect if the query is asking about a person by name."""
        query_lower = user_query.lower().strip()
        
        # Common patterns for asking about people - Updated patterns
        name_query_patterns = [
            r'^(who\s+is\s+)([a-zA-Z\s]+)$',  # "who is John Doe" or "who is john doe smith"
            r'^(what\s+is\s+)([a-zA-Z\s]+)$',  # "what is John Doe" 
            r'^(tell\s+me\s+about\s+)([a-zA-Z\s]+)$',  # "tell me about John Doe"
            r'^(show\s+me\s+)([a-zA-Z\s]+)$',  # "show me John Doe"
            r'^(find\s+)([a-zA-Z\s]+)$',  # "find John Doe"
            r'^([a-zA-Z]+\s+[a-zA-Z]+)$',  # Just "John Doe"
        ]
        
        for pattern in name_query_patterns:
            match = re.match(pattern, query_lower)
            if match:
                # Extract the name part
                name_part = match.group(2) if len(match.groups()) > 1 else match.group(1)
                # Check if it looks like a person's name (avoid technical terms)
                non_name_terms = ['job', 'position', 'role', 'company', 'client', 'vendor', 'dashboard', 'report', 'analytics', 'overview', 'status', 'system', 'api', 'endpoint', 'data', 'database', 'developer', 'engineer', 'manager']
                if not any(term in name_part.lower() for term in non_name_terms):
                    words = name_part.strip().split()
                    # Check if it's 1-4 words and all alphabetic (person name characteristics)
                    if 1 <= len(words) <= 4 and all(word.replace('-', '').replace("'", '').isalpha() for word in words):
                        return True
        
        return False

    def _extract_name_from_query(self, user_query: str) -> str:
        """Extract the person's name from a name-based query."""
        query_lower = user_query.lower().strip()
        
        # Remove common question prefixes more robustly
        name = query_lower
        prefixes_to_remove = [
            'who is ',
            'what is ',
            'tell me about ',
            'show me ',
            'find ',
            'search for ',
            'look for '
        ]
        
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        
        # Return the cleaned name, ensuring it's properly formatted
        return name.strip()

    def _fallback_endpoint_selection(self, user_query: str) -> List[str]:
        """Fallback endpoint selection using keyword matching when LLM is not available."""
        query_lower = user_query.lower().strip()
        selected_endpoints = []
        
        # Check for specific IDs first
        if re.search(r'cid\d+', query_lower):
            selected_endpoints.append('candidate_details')
            
        # Check for person name queries - NEW FUNCTIONALITY
        if self._detect_person_name_query(user_query):
            selected_endpoints.append('candidate_search')
            logger.info(f"🎯 Detected person name query, using candidate_search for: {user_query}")
            return selected_endpoints

        
        # Comprehensive keyword-based selection
        
        # Job-related queries
        if any(word in query_lower for word in ['job', 'jobs', 'position', 'opening', 'vacancy', 'opportunities']):
            if any(word in query_lower for word in ['search', 'find', 'looking', 'filter']):
                selected_endpoints.append('job_search')
            else:
                selected_endpoints.append('jobs')
        
        # Candidate-related queries (but NOT job searches)
        if any(word in query_lower for word in ['candidate', 'candidates', 'talent', 'people', 'applicant', 'resume']) and not any(word in query_lower for word in ['job', 'jobs', 'position', 'opening']):
            if any(word in query_lower for word in ['search', 'find', 'looking', 'filter']) and not any(word in query_lower for word in ['all', 'list', 'show me']):
                selected_endpoints.append('candidate_search')
            else:
                selected_endpoints.append('candidates')
            

        
        # Dashboard and analytics
        if any(word in query_lower for word in ['dashboard', 'overview', 'analytics', 'performance', 'metrics', 'report', 'status', 'progress', 'trends', 'pipeline', 'interviews', 'hiring', 'offers', 'test', 'testing', 'check', 'unified', 'json', 'handling']):
            selected_endpoints.append('dashboard_overview')
            
            # Specific dashboard types
            if 'candidate' in query_lower and 'source' in query_lower:
                selected_endpoints.append('dashboard_candidate_source')
            if 'skill' in query_lower:
                selected_endpoints.append('dashboard_jobs_by_skills')
            if 'team' in query_lower:
                selected_endpoints.append('dashboard_team_report')
            if 'client' in query_lower and 'score' in query_lower:
                selected_endpoints.append('dashboard_client_scorecard')
            if 'candidate' in query_lower and 'source' in query_lower:
                selected_endpoints.append('dashboard_candidate_source')
        

        
        # Client-related queries
        if any(word in query_lower for word in ['client', 'clients', 'company', 'companies', 'customer']):
            selected_endpoints.append('clients')
            
        # Vendor-related queries
        if any(word in query_lower for word in ['vendor', 'vendors', 'supplier', 'partner']):
            selected_endpoints.append('vendors')
            
        # User/Team related queries
        if any(word in query_lower for word in ['user', 'users', 'team', 'member', 'employee', 'staff']):
            selected_endpoints.append('users')
            
        # Role and permissions
        if any(word in query_lower for word in ['role', 'permission', 'access']):
            selected_endpoints.append('role_list')
        
        # Default fallback - if no specific match, try to get general overview
        if not selected_endpoints:
            # For general queries, provide multiple data sources
            if any(word in query_lower for word in ['show', 'list', 'all', 'everything']):
                selected_endpoints.extend(['jobs', 'candidates', 'clients'])
            else:
                selected_endpoints.append('dashboard_overview')
        
        # Ensure unique endpoints
        return list(set(selected_endpoints))
    
    def fetch_data_from_endpoint(self, endpoint: APIEndpoint) -> APIResponse:
        """Fetch data from a single endpoint using working authentication."""
        if not self.access_token:
            return APIResponse(False, 401, error_message="No access token available", endpoint_url=endpoint.url)
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "authtoken": self.available_tokens.get("authtoken", ""),
            "Content-Type": "application/json"
        }
        
        try:
            # Determine HTTP method based on endpoint
            # All endpoints use GET method
            method = "GET"
            payload = {}
            
            print(f"   📞 Making {method} request to: {endpoint.url}")
            
            # Make the request with appropriate method
            if method == "POST":
                response = self.session.request(method, endpoint.url, headers=headers, json=payload, timeout=30)
            else:
                response = self.session.request(method, endpoint.url, headers=headers, timeout=30)
                
            print(f"   📬 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   📊 Data received: {len(str(data))} characters")
                return APIResponse(True, 200, data=data, endpoint_url=endpoint.url)
            elif response.status_code == 401:
                # Authentication error - try to re-login
                error_text = response.text[:200]
                print(f"   ❌ Authentication error: {error_text}")
                
                if "other_location" in error_text.lower():
                    print("   🔄 Token expired due to login from another location. Attempting re-login...")
                    if self.login():
                        print("   🔄 Re-login successful, retrying request...")
                        # Update headers with new token
                        headers["Authorization"] = f"Bearer {self.access_token}"
                        headers["authtoken"] = self.available_tokens.get("authtoken", "")
                        
                        # Retry the request
                        if method == "POST":
                            retry_response = self.session.request(method, endpoint.url, headers=headers, json=payload, timeout=30)
                        else:
                            retry_response = self.session.request(method, endpoint.url, headers=headers, timeout=30)
                        
                        if retry_response.status_code == 200:
                            data = retry_response.json()
                            print(f"   ✅ Data fetched successfully after re-login")
                            return APIResponse(True, 200, data=data, endpoint_url=endpoint.url)
                        else:
                            print(f"   ❌ Retry failed: HTTP {retry_response.status_code}")
                            return APIResponse(False, retry_response.status_code, 
                                             error_message=f"HTTP Error {retry_response.status_code} after re-login", 
                                             endpoint_url=endpoint.url)
                
                return APIResponse(False, response.status_code, 
                                 error_message=f"Authentication failed: {error_text}", 
                                 endpoint_url=endpoint.url)
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text[:200]}")
                return APIResponse(False, response.status_code, error_message=f"HTTP Error {response.status_code}: {response.text[:200]}", endpoint_url=endpoint.url)
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout for API call to {endpoint.url}: {e}")
            return APIResponse(False, 0, error_message="Request timed out", endpoint_url=endpoint.url)
        except requests.exceptions.RequestException as e:
            logger.error(f"API call to {endpoint.url} failed: {e}")
            return APIResponse(False, 0, error_message=str(e), endpoint_url=endpoint.url)
    
    def _prepare_endpoint_with_parameters(self, endpoint_key: str, user_query: str) -> APIEndpoint:
        """Prepare endpoint URL with dynamic parameters extracted from user query."""
        endpoint = self.endpoints[endpoint_key]
        
        # Check if endpoint URL has template parameters
        if '{{' not in endpoint.url and '{' not in endpoint.url:
            return endpoint
        
        # Extract parameters from user query
        query_lower = user_query.lower().strip()
        prepared_url = endpoint.url

        
        # Handle CID parameters
        if '{{CidId}}' in prepared_url:
            cid_match = re.search(r'(CID\d+)', query_lower, re.IGNORECASE)
            if cid_match:
                prepared_url = prepared_url.replace('{{CidId}}', cid_match.group(1).upper())
            else:
                # If no CID found in query, this endpoint might not be appropriate
                logger.warning(f"No CID found in query for endpoint {endpoint_key}")
                return None
        

        
        # Handle candidate search parameters
        if endpoint_key == 'candidate_search':
            # Handle per_page parameter - check for both single and double braces
            if '{per_page}' in prepared_url or '{{per_page}}' in prepared_url:
                per_page = max(getattr(self, 'current_search_limit', 10), 50)  # Get more data for ranking
                prepared_url = prepared_url.replace('{per_page}', str(per_page))
                prepared_url = prepared_url.replace('{{per_page}}', str(per_page))
                logger.info(f"Set per_page to: {per_page}")
            
            # Handle search_term for candidate search - check for both single and double braces
            if '{search_term}' in prepared_url or '{{search_term}}' in prepared_url:
                search_term = ""
                
                # Check for name-based search terms first (NEW FUNCTIONALITY)
                if hasattr(self, 'current_search_term') and self.current_search_term:
                    search_term = self.current_search_term
                    logger.info(f"Using name as search term: {search_term}")
                # Use detected skills as search terms
                elif hasattr(self, 'current_search_skills') and self.current_search_skills:
                    # Use the first/most relevant skill for the API search
                    search_term = self.current_search_skills[0]
                    logger.info(f"Using skill as search term: {search_term}")
                elif hasattr(self, 'current_search_role') and self.current_search_role:
                    # Use role as search term
                    search_term = self.current_search_role.replace(' developer', '').replace(' engineer', '')
                    logger.info(f"Using role as search term: {search_term}")
                else:
                    # Fallback to standard extraction
                    search_term = self.extract_search_term(user_query, ['candidates', 'developers', 'engineers'])
                
                if search_term:
                    encoded_term = requests.utils.quote(search_term)
                    prepared_url = prepared_url.replace('{search_term}', encoded_term)
                    prepared_url = prepared_url.replace('{{search_term}}', encoded_term)
                    logger.info(f"Candidate search URL prepared with term: {encoded_term}")
                else:
                    prepared_url = prepared_url.replace('{search_term}', '')
                    prepared_url = prepared_url.replace('{{search_term}}', '')
        
        # Handle client search parameters
        elif endpoint_key == 'client_search':
            # Handle per_page parameter
            if '{per_page}' in prepared_url or '{{per_page}}' in prepared_url:
                per_page = getattr(self, 'current_search_limit', 50)
                prepared_url = prepared_url.replace('{per_page}', str(per_page))
                prepared_url = prepared_url.replace('{{per_page}}', str(per_page))
                logger.info(f"Client search per_page set to: {per_page}")
            
            # Handle search_term for client search
            if '{search_term}' in prepared_url or '{{search_term}}' in prepared_url:
                search_term = getattr(self, 'current_search_term', '')
                if search_term:
                    encoded_term = requests.utils.quote(search_term)
                    prepared_url = prepared_url.replace('{search_term}', encoded_term)
                    prepared_url = prepared_url.replace('{{search_term}}', encoded_term)
                    logger.info(f"Client search URL prepared with term: {encoded_term}")
                else:
                    prepared_url = prepared_url.replace('{search_term}', '')
                    prepared_url = prepared_url.replace('{{search_term}}', '')
        
        # Handle vendor search parameters
        elif endpoint_key == 'vendor_search':
            # Handle per_page parameter
            if '{per_page}' in prepared_url or '{{per_page}}' in prepared_url:
                per_page = getattr(self, 'current_search_limit', 50)
                prepared_url = prepared_url.replace('{per_page}', str(per_page))
                prepared_url = prepared_url.replace('{{per_page}}', str(per_page))
                logger.info(f"Vendor search per_page set to: {per_page}")
            
            # Handle search_term for vendor search
            if '{search_term}' in prepared_url or '{{search_term}}' in prepared_url:
                search_term = getattr(self, 'current_search_term', '')
                if search_term:
                    encoded_term = requests.utils.quote(search_term)
                    prepared_url = prepared_url.replace('{search_term}', encoded_term)
                    prepared_url = prepared_url.replace('{{search_term}}', encoded_term)
                    logger.info(f"Vendor search URL prepared with term: {encoded_term}")
                else:
                    prepared_url = prepared_url.replace('{search_term}', '')
                    prepared_url = prepared_url.replace('{{search_term}}', '')
        
        # Handle other search term parameters
        elif '{{search_term}}' in prepared_url or '{search_term}' in prepared_url:
            search_term = self.extract_search_term(user_query, ['search', 'find', 'looking for', 'show me'])
            logger.info(f"Extracted search term: '{search_term}' from query: '{user_query}'")
            
            if search_term and len(search_term.strip()) > 1:
                encoded_term = requests.utils.quote(search_term)
                # Handle both single and double braces
                if '{{search_term}}' in prepared_url:
                    prepared_url = prepared_url.replace('{{search_term}}', encoded_term)
                elif '{search_term}' in prepared_url:
                    prepared_url = prepared_url.replace('{search_term}', encoded_term)
                logger.info(f"Prepared search URL with encoded term: {encoded_term}")
            else:
                # Try to extract search term more aggressively
                logger.warning(f"No valid search term found with standard extraction, trying aggressive extraction")
                aggressive_term = self._extract_search_term_aggressive(user_query)
                if aggressive_term and len(aggressive_term.strip()) > 1:
                    encoded_term = requests.utils.quote(aggressive_term)
                    # Handle both single and double braces
                    if '{{search_term}}' in prepared_url:
                        prepared_url = prepared_url.replace('{{search_term}}', encoded_term)
                    elif '{search_term}' in prepared_url:
                        prepared_url = prepared_url.replace('{search_term}', encoded_term)
                    logger.info(f"Prepared search URL with aggressive term: {encoded_term}")
                else:
                    # Default to empty search if no specific term found
                    if '{{search_term}}' in prepared_url:
                        prepared_url = prepared_url.replace('{{search_term}}', '')
                    elif '{search_term}' in prepared_url:
                        prepared_url = prepared_url.replace('{search_term}', '')
                    logger.warning(f"No valid search term found, using empty search")
        
        # Handle client_id parameters
        if '{{client_id}}' in prepared_url:
            client_id_match = re.search(r'client[_\s]*id[_\s]*(\d+)', query_lower)
            if client_id_match:
                prepared_url = prepared_url.replace('{{client_id}}', client_id_match.group(1))
            else:
                # Default to client_id=26 as seen in HAR file
                prepared_url = prepared_url.replace('{{client_id}}', '26')
        
        # Handle field parameters for dashboard/client inputs
        if '{{field}}' in prepared_url:
            field_mapping = {
                'client': 'Clients',
                'user': 'Users',
                'role': 'Roles'
            }
            field = 'Clients'  # Default
            for key, value in field_mapping.items():
                if key in query_lower:
                    field = value
                    break
            prepared_url = prepared_url.replace('{{field}}', field)
        
        # Handle role_id parameters
        if '{{role_id}}' in prepared_url:
            role_id_match = re.search(r'role[_\s]*id[_\s]*(\d+)', query_lower)
            if role_id_match:
                prepared_url = prepared_url.replace('{{role_id}}', role_id_match.group(1))
            else:
                logger.warning(f"No role_id found in query for endpoint {endpoint_key}")
                return None
        
        return APIEndpoint(url=prepared_url, description=endpoint.description, category=endpoint.category)
    
    async def fetch_multiple_endpoints(self, endpoint_keys: List[str], user_query: str) -> Dict[str, APIResponse]:
        """Fetch data from multiple endpoints efficiently."""
        results = {}
        
        for endpoint_key in endpoint_keys:
            if endpoint_key not in self.endpoints:
                logger.warning(f"Unknown endpoint key: {endpoint_key}")
                continue
            
            # Prepare endpoint with parameters
            prepared_endpoint = self._prepare_endpoint_with_parameters(endpoint_key, user_query)
            if prepared_endpoint is None:
                logger.warning(f"Could not prepare endpoint {endpoint_key} - missing required parameters")
                continue
            
            print(f"   📡 Fetching from {endpoint_key}: {prepared_endpoint.url}")
            
            # Add small delay between requests to avoid overwhelming the server
            if len(results) > 0:
                time.sleep(0.1)
            
            response = self.fetch_data_from_endpoint(prepared_endpoint)
            results[endpoint_key] = response
        
        return results
    
    def generate_ai_response(self, user_query: str, api_responses_dict: Dict[str, APIResponse]) -> str:
        """Generate conversational AI response based on fetched data."""
        # Separate successful and failed responses
        successful_data = {}
        failed_endpoints = []
        
        for endpoint_key, response in api_responses_dict.items():
            if response.success and response.data:
                successful_data[endpoint_key] = response.data
            else:
                failed_endpoints.append(f"{endpoint_key} ({response.error_message or 'Unknown error'})")
        
        # If no successful data, provide fallback response
        if not successful_data:
            return self._generate_fallback_response(user_query, failed_endpoints)
        
        # Generate conversational response (no tables)
        return self._generate_conversational_ai_response(user_query, successful_data)
    
    def _generate_table_response(self, user_query: str, successful_data: Dict[str, any]) -> str:
        """Generate AI-powered conversational response instead of just tables."""
        # Let AI generate the complete response with data
        if self.gemini_model:
            try:
                return self._generate_conversational_ai_response(user_query, successful_data)
            except Exception as e:
                logger.error(f"Error generating AI response: {e}")
        
        # Fallback to simple response if AI fails
        return self._generate_simple_data_response(user_query, successful_data)
    
    def _generate_conversational_ai_response(self, user_query: str, successful_data: Dict[str, any]) -> str:
        """Generate natural conversational response analyzing the raw data."""
        # Extract actual data from API responses
        extracted_data = {}
        total_records = {}
        
        for endpoint_key, api_response in successful_data.items():
            if isinstance(api_response, dict):
                # Log the response structure for debugging
                logger.info(f"Processing {endpoint_key} response with keys: {list(api_response.keys())[:10]}")
                
                # Debug print for all paginated data
                if 'TotalItems' in api_response or 'Count' in api_response:
                    endpoint_type = 'Unknown'
                    if 'candidate' in endpoint_key.lower():
                        endpoint_type = 'Candidate'
                    elif 'job' in endpoint_key.lower():
                        endpoint_type = 'Job'
                    elif 'client' in endpoint_key.lower():
                        endpoint_type = 'Client'
                    elif 'vendor' in endpoint_key.lower():
                        endpoint_type = 'Vendor'
                    
                    print(f"\n🔍 DEBUG: {endpoint_type} API Response Structure:")
                    print(f"   - TotalItems: {api_response.get('TotalItems', 'N/A')}")
                    print(f"   - Count: {api_response.get('Count', 'N/A')}")
                    print(f"   - TotalPages: {api_response.get('TotalPages', 'N/A')}")
                    print(f"   - CurrentPage: {api_response.get('CurrentPage', 'N/A')}")
                    print(f"   - ItemsPerPage: {api_response.get('ItemsPerPage', 'N/A')}")
                    print(f"   - Data array length: {len(api_response.get('Data', []))}")
                    
                    total_field = 'TotalItems' if 'TotalItems' in api_response else 'Count'
                    total_value = api_response.get('TotalItems') or api_response.get('Count', 0)
                    print(f"   - Using {total_field} for count: {total_value}\n")
                
                # Extract the actual data list
                raw_data = None
                if 'data' in api_response:
                    raw_data = api_response['data']
                elif 'Data' in api_response:
                    raw_data = api_response['Data']
                elif 'Clients' in api_response:
                    raw_data = api_response['Clients']
                elif 'Users' in api_response:
                    raw_data = api_response['Users']
                elif 'results' in api_response:
                    raw_data = api_response['results']
                
                # Apply time-based filtering if applicable
                if raw_data and hasattr(self, 'current_time_period') and self.current_time_period:
                    # Determine the date field based on endpoint type
                    date_field = None
                    if 'job' in endpoint_key.lower():
                        date_field = 'CreatedAt'  # Jobs use CreatedAt
                    elif 'candidate' in endpoint_key.lower():
                        date_field = 'CreatedAt'  # Candidates use CreatedAt
                    
                    if date_field:
                        filtered_data = TimeUtils.filter_data_by_time_period(raw_data, date_field, self.current_time_period)
                        logger.info(f"🕐 Time filtering: {len(raw_data)} → {len(filtered_data)} records for {self.current_time_period}")
                        raw_data = filtered_data
                
                # Apply candidate ranking/filtering if applicable
                if (raw_data and endpoint_key == 'candidate_search' and 
                    ((hasattr(self, 'current_search_skills') and self.current_search_skills) or 
                     (hasattr(self, 'current_search_role') and self.current_search_role))):
                    # Rank candidates based on skills and role match
                    ranked_candidates = CandidateSearchUtils.rank_candidates(
                        raw_data, 
                        skills=self.current_search_skills, 
                        role=self.current_search_role
                    )
                    
                    # Limit to requested number
                    limited_candidates = ranked_candidates[:self.current_search_limit]
                    
                    logger.info(f"🎯 Candidate ranking: {len(raw_data)} → {len(limited_candidates)} top candidates")
                    if limited_candidates:
                        top_scores = []
                        for c in limited_candidates[:3]:
                            name = f"{c.get('FirstName', '')} {c.get('LastName', '')}"
                            score = c.get('_match_score', 0)
                            top_scores.append(f"{name.strip()} ({score:.1f}%)")
                        logger.info(f"Top candidate scores: {top_scores}")
                    
                    raw_data = limited_candidates
                
                # Store the (possibly filtered) data
                if 'data' in api_response:
                    extracted_data[endpoint_key] = raw_data
                    total_records[endpoint_key] = len(raw_data) if raw_data else 0
                elif 'Data' in api_response:
                    extracted_data[endpoint_key] = raw_data
                    # Store original total counts (before filtering)
                    original_total = api_response.get('TotalItems') or api_response.get('Count', 0)
                    filtered_total = len(raw_data) if raw_data else 0
                    
                    # If we filtered by time, use the filtered count
                    if self.current_time_period and raw_data is not None:
                        total_records[endpoint_key] = filtered_total
                        extracted_data[endpoint_key + '_total_count'] = filtered_total
                        extracted_data[endpoint_key + '_original_total'] = original_total
                        extracted_data[endpoint_key + '_time_period'] = self.current_time_period
                    else:
                        total_records[endpoint_key] = original_total
                        extracted_data[endpoint_key + '_total_count'] = original_total
                    
                    # Also store pagination info
                    extracted_data[endpoint_key + '_pagination'] = {
                        'total_items': original_total,
                        'filtered_items': filtered_total if self.current_time_period else original_total,
                        'total_pages': api_response.get('TotalPages', 0),
                        'current_page': api_response.get('CurrentPage', 1),
                        'items_per_page': api_response.get('ItemsPerPage', 10),
                        'time_filtered': bool(self.current_time_period)
                    }
                elif 'Clients' in api_response:
                    extracted_data[endpoint_key] = raw_data
                    # Store original total counts (before filtering)
                    original_total = api_response.get('TotalItems') or api_response.get('Count', 0)
                    filtered_total = len(raw_data) if raw_data else 0
                    
                    # If we filtered by time, use the filtered count
                    if self.current_time_period and raw_data is not None:
                        total_records[endpoint_key] = filtered_total
                        extracted_data[endpoint_key + '_total_count'] = filtered_total
                        extracted_data[endpoint_key + '_original_total'] = original_total
                        extracted_data[endpoint_key + '_time_period'] = self.current_time_period
                    else:
                        total_records[endpoint_key] = original_total
                        extracted_data[endpoint_key + '_total_count'] = original_total
                    
                    # Also store pagination info
                    extracted_data[endpoint_key + '_pagination'] = {
                        'total_items': original_total,
                        'filtered_items': filtered_total if self.current_time_period else original_total,
                        'total_pages': api_response.get('TotalPages', 0),
                        'current_page': api_response.get('CurrentPage', 1),
                        'items_per_page': api_response.get('ItemsPerPage', 10),
                        'time_filtered': bool(self.current_time_period)
                    }
                elif 'Users' in api_response:
                    extracted_data[endpoint_key] = raw_data
                    # Store original total counts (before filtering)
                    original_total = api_response.get('TotalUsers') or api_response.get('Count', 0)
                    filtered_total = len(raw_data) if raw_data else 0
                    
                    # If we filtered by time, use the filtered count
                    if self.current_time_period and raw_data is not None:
                        total_records[endpoint_key] = filtered_total
                        extracted_data[endpoint_key + '_total_count'] = filtered_total
                        extracted_data[endpoint_key + '_original_total'] = original_total
                        extracted_data[endpoint_key + '_time_period'] = self.current_time_period
                    else:
                        total_records[endpoint_key] = original_total
                        extracted_data[endpoint_key + '_total_count'] = original_total
                    
                    # Also store pagination info
                    extracted_data[endpoint_key + '_pagination'] = {
                        'total_items': original_total,
                        'filtered_items': filtered_total if self.current_time_period else original_total,
                        'total_pages': api_response.get('TotalPages', 0),
                        'current_page': api_response.get('CurrentPage', 1),
                        'items_per_page': api_response.get('PerPage', 10),
                        'time_filtered': bool(self.current_time_period)
                    }
                elif 'results' in api_response:
                    extracted_data[endpoint_key] = raw_data
                    total_records[endpoint_key] = api_response.get('count', len(raw_data)) if raw_data else 0
                else:
                    # Check if the response itself is empty
                    if not api_response:
                        logger.warning(f"{endpoint_key} returned empty response")
                        extracted_data[endpoint_key] = []
                        total_records[endpoint_key] = 0
                    else:
                        # For non-list data like counts
                        extracted_data[endpoint_key] = api_response
                        total_records[endpoint_key] = 1
        
        # Process data to reduce size for LLM
        print("   🧠 Processing data for AI response...")
        processed_data = self._extract_key_data_for_llm(extracted_data, user_query)
        
        # Log the processed data size
        print(f"   📊 Processed data summary: {len(str(processed_data))} characters")
        
        # Check if time filtering was applied
        time_context = ""
        if hasattr(self, 'current_time_period') and self.current_time_period:
            time_context = f"""
TIME FILTERING APPLIED:
- Time Period: {self.current_time_period.replace('_', ' ').title()}
- Current IST Time: {TimeUtils.get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}
- Data has been filtered to show only records from the specified time period
"""
        
        # Check if candidate search/ranking was applied
        candidate_context = ""
        if ((hasattr(self, 'current_search_skills') and self.current_search_skills) or 
            (hasattr(self, 'current_search_role') and self.current_search_role)):
            candidate_context = f"""
CANDIDATE SEARCH APPLIED:
- Skills Searched: {', '.join(self.current_search_skills) if self.current_search_skills else 'None'}
- Role Searched: {self.current_search_role or 'None'}
- Results Limit: Top {self.current_search_limit}
- Candidates have been ranked by skill/role match score (0-100%)
- Look for '_match_score', '_skill_score', '_role_score' fields in data
"""

        prompt = f"""
You are a friendly and knowledgeable OATS recruitment system assistant. Provide a natural, conversational response that feels engaging and helpful to the user's query.

USER QUERY: "{user_query}"
{time_context}{candidate_context}
PROCESSED DATA SUMMARY:
{json.dumps(processed_data, indent=2)}

TOTAL RECORDS: {json.dumps(total_records, indent=2)}

IMPORTANT NOTES:
- Look for "_total_count" or "_pagination.total_items" for the TOTAL in database
- DO NOT use the length of the records array for total count
- The API returns paginated data - we might have 975,952 total but only show 10 per page
- Always use TotalItems or Count fields, never count the Data array
- If time filtering was applied, explain the time period in your response
- For time-based queries, focus on the filtered count, not the original total

CONVERSATIONAL INSTRUCTIONS:
1. START with a warm, engaging introduction that acknowledges what the user is looking for
2. Use appreciative phrases like "Great question!", "I'd be happy to help!", "That's interesting data!", "Nice insight!", "Excellent choice!", etc.
3. For "how many" questions, start with enthusiasm: "Great question! We currently have X jobs/candidates..." or "I'd be happy to help! We have X..."
4. For searches, show appreciation: "Excellent choice! I found X Python developers..." or "That's a popular skill! Here are X candidates..."
5. For time-based queries, be conversational: "Looking at this week's data (Aug 19-25, 2025), I can see we created X jobs - that's quite active!"
6. CRITICAL: For total counts, use the total_count field or pagination.total_items
7. If time filtering was applied, mention the time period conversationally
8. If candidate search/ranking was applied, explain the search criteria enthusiastically
9. After the conversational intro, display data in a clean, professional HTML table
10. Show relevant fields based on data type:
    - Candidates: Name, Job Title, Location, Experience, Skills, Company, Match Score (if ranked)
    - Jobs: ID, Title, Client/Company, Location, Type, Status, Pipeline Count, Submission Count, Skills Required, Experience Range, Created Date
    - Clients: Client ID, Name, Contact Number, Email, Website, Status, Primary Owner, Created By
    - Users/Team Members: Name, Role, Email, Status, Business Unit, Reporting To, Created By
    - Vendors: Vendor ID, Name, Federal ID, Contact Number, Website, Address, Technologies, Created By
    - Dashboard: Use the ACTUAL metric names and values from the data_summaries.dashboard.metrics - DO NOT create placeholder names like "Overview 1", use the real field names like "Total Overview", "Upcoming Interviews", etc.
11. CRITICAL FOR DASHBOARD DATA: When showing dashboard metrics, always use the actual field names and values from the API response. Never create placeholder or generic names.
12. Always indicate: "Showing X of Y total records" after your conversational response (except for dashboard data which shows all metrics)
13. For empty results, provide helpful context (e.g., "No jobs were created this week, but we have X total jobs in the system")
14. END with a relevant follow-up question or suggestion related to the data shown:
    - "Would you like me to dive deeper into any of these results?"
    - "Are you interested in seeing more details about any specific [job/candidate/client]?"
    - "Would you like to explore [related topic] as well?"
    - "Should I help you search for candidates with different skills?"
    - "Would you like to see how this compares to previous periods?"
    - "Any particular [job/candidate/client] catch your attention?"

HTML FORMAT:
<div class="ai-response">
<strong>Total [Type]: [Count]</strong><br>
<em>Showing [X] of [Total] records</em><br><br>
<table style="width:100%; border-collapse:collapse; margin-top:10px;">
<thead>
<tr style="background:#f5f5f5; font-weight:bold;">
<th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">Column1</th>
<th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">Column2</th>
...
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px;">Data1</td>
<td style="padding:8px;">Data2</td>
...
</tr>
</tbody>
</table>
</div>

FOR DASHBOARD DATA, USE THIS FORMAT INSTEAD:
<div class="ai-response">
<strong>Dashboard Overview</strong><br><br>
<table style="width:100%; border-collapse:collapse; margin-top:10px;">
<thead>
<tr style="background:#f5f5f5; font-weight:bold;">
<th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">Metric</th>
<th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">Value</th>
</tr>
</thead>
<tbody>
[Use actual metric names from data_summaries.dashboard.metrics - NOT placeholders]
</tbody>
</table>
</div>
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return f"<div class='ai-response'>{response.text.strip()}</div>"
        except Exception as e:
            logger.error(f"Error in conversational AI response: {e}")
            raise
    
    def _extract_key_data_for_llm(self, extracted_data: Dict, user_query: str) -> Dict:
        """Extract and summarize key data for LLM processing to avoid token limits."""
        processed = {
            'query_type': self._determine_query_type(user_query),
            'data_summaries': {},
            'total_count': 0
        }
        
        for endpoint_key, data in extracted_data.items():
            if '_total_count' in endpoint_key:
                processed['total_count'] = data
                continue
                
            if isinstance(data, list) and len(data) > 0:
                # Determine data type and extract relevant fields
                sample_size = min(30, len(data))  # Limit to 30 records
                
                if 'candidate' in endpoint_key.lower():
                    # Get the actual total from pagination info or total_count
                    pagination_info = extracted_data.get(endpoint_key + '_pagination', {})
                    actual_total = (
                        extracted_data.get(endpoint_key + '_total_count') or 
                        pagination_info.get('total_items') or 
                        len(data)
                    )
                    
                    processed['data_summaries']['candidates'] = {
                        'total_count': actual_total,
                        'records_shown': sample_size,
                        'actual_page_size': len(data),
                        'pagination': pagination_info,
                        'records': []
                    }
                    processed['total_count'] = actual_total  # Update the main total
                    
                    for item in data[:sample_size]:
                        processed['data_summaries']['candidates']['records'].append({
                            'name': f"{item.get('FirstName', '')} {item.get('LastName', '')}".strip() or 'N/A',
                            'job_title': item.get('CurrentJobTitle', 'N/A'),
                            'location': item.get('Location', 'N/A'),
                            'experience': f"{item.get('ExperienceYears', 0)} years",
                            'skills': self._truncate_skills(item.get('PrimarySkills') or item.get('Skills', '')),
                            'company': item.get('CurrentCompany', 'N/A'),
                            'source': item.get('Source', 'N/A')
                        })
                
                elif 'job' in endpoint_key.lower():
                    # Get the actual total from pagination info or total_count
                    pagination_info = extracted_data.get(endpoint_key + '_pagination', {})
                    actual_total = (
                        extracted_data.get(endpoint_key + '_total_count') or 
                        pagination_info.get('total_items') or 
                        len(data)
                    )
                    
                    processed['data_summaries']['jobs'] = {
                        'total_count': actual_total,
                        'records_shown': sample_size,
                        'actual_page_size': len(data),
                        'pagination': pagination_info,
                        'records': []
                    }
                    if actual_total > processed.get('total_count', 0):
                        processed['total_count'] = actual_total
                    
                    for item in data[:sample_size]:
                        processed['data_summaries']['jobs']['records'].append({
                            'id': item.get('JobCode') or item.get('job_id') or item.get('JobId', 'N/A'),
                            'title': item.get('JobTitle') or item.get('job_title', 'N/A'),
                            'company': item.get('Client') or item.get('company_name') or item.get('CompanyName', 'N/A'),
                            'client': item.get('Client') or item.get('client_name') or item.get('ClientName', 'N/A'),
                            'location': item.get('Location') or item.get('location', 'N/A'),
                            'type': item.get('JobType') or item.get('job_type', 'N/A'),
                            'status': item.get('JobStatus') or item.get('status') or item.get('Status', 'N/A'),
                            'created_date': item.get('CreatedAt') or item.get('created_date') or item.get('CreatedDate', 'N/A'),
                            'pipeline_count': item.get('PipelineCount', 0),
                            'submission_count': item.get('SubmissionCount', 0),
                            'primary_skills': item.get('PrimarySkills', 'N/A'),
                            'experience_min': item.get('Experience', {}).get('Min', 'N/A') if isinstance(item.get('Experience'), dict) else 'N/A',
                            'experience_max': item.get('Experience', {}).get('Max', 'N/A') if isinstance(item.get('Experience'), dict) else 'N/A'
                        })
                
                elif 'client' in endpoint_key.lower():
                    # Get the actual total from pagination info or total_count
                    pagination_info = extracted_data.get(endpoint_key + '_pagination', {})
                    actual_total = (
                        extracted_data.get(endpoint_key + '_total_count') or 
                        pagination_info.get('total_items') or 
                        len(data)
                    )
                    
                    processed['data_summaries']['clients'] = {
                        'total_count': actual_total,
                        'records_shown': sample_size,
                        'actual_page_size': len(data),
                        'pagination': pagination_info,
                        'records': []
                    }
                    if actual_total > processed.get('total_count', 0):
                        processed['total_count'] = actual_total
                    
                    for item in data[:sample_size]:
                        processed['data_summaries']['clients']['records'].append({
                            'id': item.get('ClientId') or item.get('CliId') or item.get('client_id', 'N/A'),
                            'name': item.get('Name') or item.get('client_name') or item.get('ClientName', 'N/A'),
                            'company': item.get('Name') or item.get('company_name') or item.get('CompanyName', 'N/A'),
                            'email': item.get('Email') or item.get('email', 'N/A'),
                            'phone': item.get('ContactNumber') or item.get('phone') or item.get('Phone', 'N/A'),
                            'location': item.get('Location') or item.get('location', 'N/A'),
                            'status': item.get('Status') or item.get('status', 'Active'),
                            'primary_owner': item.get('PrimaryOwnerName') or item.get('PrimaryOwner', 'N/A'),
                            'created_by': item.get('CreatedBy') or item.get('created_by', 'N/A'),
                            'website': item.get('Website') or item.get('website', 'N/A')
                        })
                
                elif 'vendor' in endpoint_key.lower():
                    # Get the actual total from pagination info or total_count
                    pagination_info = extracted_data.get(endpoint_key + '_pagination', {})
                    actual_total = (
                        extracted_data.get(endpoint_key + '_total_count') or 
                        pagination_info.get('total_items') or 
                        len(data)
                    )
                    
                    processed['data_summaries']['vendors'] = {
                        'total_count': actual_total,
                        'records_shown': sample_size,
                        'actual_page_size': len(data),
                        'pagination': pagination_info,
                        'records': []
                    }
                    if actual_total > processed.get('total_count', 0):
                        processed['total_count'] = actual_total
                    
                    for item in data[:sample_size]:
                        processed['data_summaries']['vendors']['records'].append({
                            'id': item.get('vendor_id') or item.get('VendorId', 'N/A'),
                            'name': item.get('vendor_name') or item.get('VendorName', 'N/A'),
                            'company': item.get('company_name') or item.get('CompanyName', 'N/A'),
                            'email': item.get('email') or item.get('Email', 'N/A'),
                            'phone': item.get('phone') or item.get('Phone', 'N/A'),
                            'location': item.get('location') or item.get('Location', 'N/A'),
                            'status': item.get('status') or item.get('Status', 'Active'),
                            'type': item.get('vendor_type') or item.get('VendorType', 'N/A')
                        })
                
                elif 'user' in endpoint_key.lower():
                    # Get the actual total from pagination info or total_count
                    pagination_info = extracted_data.get(endpoint_key + '_pagination', {})
                    actual_total = (
                        extracted_data.get(endpoint_key + '_total_count') or 
                        pagination_info.get('total_items') or 
                        len(data)
                    )
                    
                    processed['data_summaries']['users'] = {
                        'total_count': actual_total,
                        'records_shown': sample_size,
                        'actual_page_size': len(data),
                        'pagination': pagination_info,
                        'records': []
                    }
                    if actual_total > processed.get('total_count', 0):
                        processed['total_count'] = actual_total
                    
                    for item in data[:sample_size]:
                        processed['data_summaries']['users']['records'].append({
                            'id': item.get('Id') or item.get('user_id') or item.get('UserId', 'N/A'),
                            'name': item.get('Name') or item.get('name') or item.get('UserName', 'N/A'),
                            'email': item.get('Email') or item.get('email', 'N/A'),
                            'role': item.get('Role') or item.get('role', 'N/A'),
                            'status': item.get('Status') or item.get('status', 'Active'),
                            'business_unit': item.get('BusinessUnit') or item.get('business_unit', 'N/A'),
                            'reporting_to': item.get('Reporting') or item.get('reporting_to', 'N/A'),
                            'created_by': item.get('CreatedBy') or item.get('created_by', 'N/A'),
                            'created_at': item.get('CreatedAt') or item.get('created_at', 'N/A'),
                            'admin': item.get('Admin', False)
                        })
            
            # Handle dashboard data specifically - not in a list format
            elif 'dashboard' in endpoint_key.lower() and isinstance(data, dict):
                # Dashboard data comes as key-value pairs, not as a list
                dashboard_metrics = {}
                for key, value in data.items():
                    # Convert snake_case to readable names and preserve actual values
                    readable_key = key.replace('_', ' ').title()
                    dashboard_metrics[readable_key] = value
                
                processed['data_summaries']['dashboard'] = {
                    'type': 'dashboard_metrics',
                    'total_metrics': len(dashboard_metrics),
                    'metrics': dashboard_metrics
                }
                
                # Log dashboard processing for debugging
                logger.info(f"🎯 Dashboard data processed: {len(dashboard_metrics)} metrics")
                for key, value in list(dashboard_metrics.items())[:5]:
                    logger.info(f"   - {key}: {value}")
        
        return processed
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for better response formatting."""
        query_lower = query.lower()
        if any(word in query_lower for word in ['how many', 'count', 'total', 'number']):
            return 'count'
        elif any(word in query_lower for word in ['list', 'show', 'display', 'all']):
            return 'list'
        elif any(word in query_lower for word in ['search', 'find', 'looking for']):
            return 'search'
        return 'general'
    
    def _truncate_skills(self, skills: str) -> str:
        """Truncate skills string if too long."""
        if not skills:
            return 'N/A'
        # Remove weird comma-separated characters
        if len(skills) > 100 and skills.count(',') > 20:
            # This is likely the weird format, extract actual skills
            skills_list = [s.strip() for s in skills.split(',') if s.strip() and len(s.strip()) > 1]
            return ', '.join(skills_list[:5]) + '...' if len(skills_list) > 5 else ', '.join(skills_list)
        return skills[:100] + '...' if len(skills) > 100 else skills
    
    def _generate_simple_data_response(self, user_query: str, successful_data: Dict[str, any]) -> str:
        """Simple fallback response when AI is not available."""
        response_parts = [f"<h3>Results for: {user_query}</h3>"]
        
        for endpoint_key, data in successful_data.items():
            if isinstance(data, dict):
                if 'Data' in data:
                    count = data.get('Count', len(data['Data']))
                    response_parts.append(f"<p>Found {count} {endpoint_key.replace('_', ' ')}</p>")
                elif 'data' in data:
                    count = len(data['data'])
                    response_parts.append(f"<p>Found {count} {endpoint_key.replace('_', ' ')}</p>")
        
        return "\n".join(response_parts)
    
    def _generate_ai_summary(self, user_query: str, successful_data: Dict[str, any]) -> str:
        """Generate AI summary to accompany the tables."""
        prompt = f"""
You are an AI assistant for the OATS recruitment system. Analyze the following data and provide a brief, insightful summary in 2-3 sentences.

User Query: "{user_query}"

Data Summary:
{json.dumps({k: {"type": type(v).__name__, "count": len(v) if hasattr(v, '__len__') else 1} for k, v in successful_data.items()}, indent=2)}

Provide key insights, trends, or important findings. Be concise and focus on what matters most to the user.
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return ""
    
    def _generate_fallback_response(self, user_query: str, failed_endpoints: List[str] = None) -> str:
        """Generate a helpful fallback response when data fetching fails."""
        if not self.gemini_model:
            return """I apologize, but I'm currently unable to process your request as the AI service is unavailable. 
However, I can tell you that this system supports queries about:
- Job listings and searches
- Candidate information and searches  
- Dashboard analytics and reports
- Client and vendor management
- User and role management

Please try again later or contact your system administrator."""

        # Log failed endpoints for debugging
        if failed_endpoints:
            logger.error(f"Failed to fetch data from endpoints: {failed_endpoints} for query: '{user_query}'")

        # Check if this is an authentication issue
        auth_keywords = ['token is required', 'authentication', '401', 'unauthorized']
        is_auth_issue = any(keyword in str(failed_endpoints).lower() for keyword in auth_keywords)
        
        if is_auth_issue or not self.access_token:
            return f"""🔐 **Authentication Issue Detected**

I encountered an authentication issue while retrieving data from the OATS system for your query: "{user_query}"

This usually means:
1. ✅ The system is working correctly
2. 🔑 There was a temporary authentication token issue
3. 🔄 I'll automatically retry authentication on the next request

**What you can do:**
- Try your query again (I'll attempt to re-authenticate)
- If the issue persists, please log out and log back in
- Contact your system administrator if problems continue

**Your query was understood as:** {user_query}
**I was attempting to fetch from:** {', '.join(failed_endpoints) if failed_endpoints else 'dashboard data'}

The OATS system normally provides comprehensive information about jobs, candidates, analytics, and more. Please try again!"""

        # For candidate search failures, provide specific guidance
        if 'candidate_search' in (failed_endpoints or []) and any(term in user_query.lower() for term in ['java', 'python', 'developer', 'engineer']):
            return f"""I'm having trouble searching for candidates at the moment. 

Based on your query "{user_query}", I was trying to search our candidate database of 975,952 professionals, but the search didn't return any results.

This could mean:
1. The search term needs to be more specific (try "Java" instead of "Java developer")
2. There might be a temporary connection issue
3. The candidates might be listed under different titles

Please try:
- Using just the technology name: "Java", "Python", "React"
- Using broader terms: "developer", "engineer"
- Checking if you're logged in properly

Would you like me to try a different search?"""

        # Use LLM to generate a helpful response even without data
        prompt = f"""
The user asked: "{user_query}"

I attempted to retrieve data from the OATS recruitment system but was unable to fetch the information successfully.
{f"Failed endpoints: {', '.join(failed_endpoints)}" if failed_endpoints else ""}

As a friendly OATS recruitment system assistant, provide a helpful response that:
1. Acknowledges the issue with empathy and understanding
2. Explains what the system normally provides for this type of query in an encouraging way
3. Suggests alternative ways the user might get the information they need
4. Offers to help with related queries enthusiastically
5. End with a helpful follow-up question like:
   - "Would you like me to try a different approach?"
   - "Can I help you with a related query instead?"
   - "Should I show you what data is currently available?"

Be conversational, supportive, and helpful.
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return f"""I apologize, but I'm currently unable to retrieve the requested information for "{user_query}". 

The OATS system normally provides comprehensive data about jobs, candidates, dashboard analytics, and business operations. Please try again in a moment, or contact your system administrator if the issue persists."""
    
    def _generate_structured_response(self, user_query: str, successful_data: Dict[str, any]) -> str:
        """Generate a structured response when AI is not available but data was fetched successfully."""
        response_parts = []
        response_parts.append(f"📊 **Query Results for:** {user_query}")
        response_parts.append("=" * 50)
        
        for endpoint_key, data in successful_data.items():
            endpoint_desc = self.endpoint_descriptions.get(endpoint_key, endpoint_key)
            response_parts.append(f"\n🔹 **{endpoint_key.replace('_', ' ').title()}:**")
            
            if isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    items = data['data']
                    response_parts.append(f"   Found {len(items)} items")
                    
                    # Show first few items
                    for i, item in enumerate(items[:3]):
                        if isinstance(item, dict):
                            # Extract key fields based on endpoint type
                            if 'job' in endpoint_key:
                                title = item.get('job_title', 'N/A')
                                client = item.get('client', 'N/A')
                                location = item.get('location', 'N/A')
                                response_parts.append(f"   {i+1}. {title} at {client} ({location})")
                            elif 'candidate' in endpoint_key:
                                name = item.get('first_name', 'N/A')
                                title = item.get('current_job_title', 'N/A')
                                city = item.get('city', 'N/A')
                                response_parts.append(f"   {i+1}. {name} - {title} ({city})")
                            elif 'client' in endpoint_key:
                                name = item.get('name', 'N/A')
                                email = item.get('email', 'N/A')
                                response_parts.append(f"   {i+1}. {name} ({email})")
                            else:
                                # Generic handling
                                key_field = next((k for k in ['name', 'title', 'id'] if k in item), list(item.keys())[0] if item else 'N/A')
                                response_parts.append(f"   {i+1}. {item.get(key_field, 'N/A')}")
                    
                    if len(items) > 3:
                        response_parts.append(f"   ... and {len(items) - 3} more items")
                
                elif 'count' in data or any(k in data for k in ['total', 'summary']):
                    # Dashboard/summary data
                    for key, value in data.items():
                        if isinstance(value, (int, float, str)):
                            response_parts.append(f"   {key.replace('_', ' ').title()}: {value}")
                else:
                    # Generic dict handling
                    for key, value in list(data.items())[:5]:
                        if isinstance(value, (int, float, str)):
                            response_parts.append(f"   {key.replace('_', ' ').title()}: {value}")
            
            elif isinstance(data, list):
                response_parts.append(f"   Found {len(data)} items")
                for i, item in enumerate(data[:3]):
                    response_parts.append(f"   {i+1}. {str(item)[:100]}...")
                if len(data) > 3:
                    response_parts.append(f"   ... and {len(data) - 3} more items")
            else:
                response_parts.append(f"   {str(data)[:200]}...")
        
        response_parts.append("\n" + "=" * 50)
        response_parts.append("💡 **Note:** AI processing is currently unavailable, showing raw data summary.")
        response_parts.append("🔄 For detailed analysis, please try again later when AI services are restored.")
        
        return "\n".join(response_parts)
    
    def format_data_as_table(self, data: any, endpoint_key: str, max_rows: int = 50) -> str:
        """Convert API response data into HTML table format."""
        try:
            if not data:
                return "<p><em>No data available</em></p>"
            
            # Handle different data structures
            if isinstance(data, dict):
                # First, try to find actual list data in various nested structures
                list_data = None
                total_count = 0
                
                # Check for common API response patterns
                if 'data' in data and isinstance(data['data'], list):
                    list_data = data['data']
                    total_count = len(data['data'])
                elif 'results' in data and isinstance(data['results'], list):
                    list_data = data['results']
                    total_count = len(data['results'])
                elif 'items' in data and isinstance(data['items'], list):
                    list_data = data['items']
                    total_count = len(data['items'])
                elif 'clients' in data and isinstance(data['clients'], list):
                    list_data = data['clients']
                    total_count = len(data['clients'])
                else:
                    # Look for any list in the data structure
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            list_data = value
                            total_count = len(value)
                            logger.info(f"Found list data in key '{key}' with {total_count} items")
                            break
                
                # If we found list data, create a table
                if list_data:
                    items = list_data[:max_rows]  # Limit rows for performance
                    if not items:
                        return "<p><em>No records found</em></p>"
                    
                    return self._create_html_table(items, endpoint_key, total_count)
                
                # If no list data found, show summary table but also log the structure
                logger.info(f"No list data found in {endpoint_key}. Data keys: {list(data.keys())}")
                return self._create_summary_table(data, endpoint_key)
            
            elif isinstance(data, list):
                if not data:
                    return "<p><em>No records found</em></p>"
                
                items = data[:max_rows]
                return self._create_html_table(items, endpoint_key, len(data))
            
            else:
                return f"<p><strong>Data:</strong> {str(data)}</p>"
                
        except Exception as e:
            logger.error(f"Error formatting table for {endpoint_key}: {e}")
            return f"<p><em>Error formatting data: {str(e)}</em></p>"
    
    def _create_html_table(self, items: List[Dict], endpoint_key: str, total_count: int) -> str:
        """Create HTML table from list of dictionaries."""
        if not items or not isinstance(items[0], dict):
            return "<p><em>Invalid data format for table</em></p>"
        
        # Get column headers from first item, prioritizing important fields
        sample_item = items[0]
        
        # Define priority columns for different endpoint types
        priority_columns = {
            'job': ['job_code', 'job_title', 'client', 'location', 'job_status', 'created_at'],
            'candidate': ['cid_id', 'first_name', 'last_name', 'email', 'mobile_phone', 'city', 'current_job_title'],
            'client': ['id', 'name', 'email', 'contact_number', 'website', 'status'],
            'vendor': ['vendor_id', 'vendor_name', 'contact_number', 'email', 'address'],
            'user': ['id', 'username', 'email', 'first_name', 'last_name', 'role'],
            'dashboard': list(sample_item.keys())[:10]  # Show first 10 fields for dashboard
        }
        
        # Determine endpoint type
        endpoint_type = 'dashboard'
        for key_type in priority_columns:
            if key_type in endpoint_key:
                endpoint_type = key_type
                break
        
        # Get relevant columns
        preferred_cols = priority_columns.get(endpoint_type, list(sample_item.keys())[:10])
        columns = [col for col in preferred_cols if col in sample_item] or list(sample_item.keys())[:10]
        
        # Start building HTML table
        table_html = ['<div class="table-container">']
        
        # Add summary info
        showing = min(len(items), 50)
        if total_count > showing:
            table_html.append(f'<p class="table-summary">Showing {showing} of {total_count} records</p>')
        else:
            table_html.append(f'<p class="table-summary">Total: {total_count} records</p>')
        
        table_html.append('<table class="data-table">')
        
        # Table header
        table_html.append('<thead><tr>')
        for col in columns:
            # Format column names
            display_name = col.replace('_', ' ').title()
            table_html.append(f'<th>{display_name}</th>')
        table_html.append('</tr></thead>')
        
        # Table body
        table_html.append('<tbody>')
        for item in items:
            table_html.append('<tr>')
            for col in columns:
                value = item.get(col, '')
                
                # Format different value types
                if value is None:
                    formatted_value = '<em>N/A</em>'
                elif isinstance(value, dict):
                    # Handle nested objects
                    if 'name' in value:
                        formatted_value = str(value['name'])
                    elif 'value' in value:
                        formatted_value = str(value['value'])
                    else:
                        formatted_value = str(value)[:50] + ('...' if len(str(value)) > 50 else '')
                elif isinstance(value, list):
                    formatted_value = ', '.join(str(v) for v in value[:3])
                    if len(value) > 3:
                        formatted_value += '...'
                elif isinstance(value, str) and len(value) > 100:
                    formatted_value = value[:100] + '...'
                else:
                    formatted_value = str(value)
                
                table_html.append(f'<td>{formatted_value}</td>')
            table_html.append('</tr>')
        table_html.append('</tbody>')
        
        table_html.append('</table>')
        table_html.append('</div>')
        
        return '\n'.join(table_html)
    
    def _create_summary_table(self, data: Dict, endpoint_key: str) -> str:
        """Create summary table for non-list data."""
        table_html = ['<div class="summary-container">']
        
        # Add a note that this is summary info and actual data might be available
        table_html.append('<p class="table-summary">📋 API Response Summary (Raw data structure)</p>')
        table_html.append('<table class="summary-table">')
        
        # Handle nested data structures
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict) and len(v) < 10:  # Only flatten small dictionaries
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = flatten_dict(data)
        
        for key, value in list(flattened.items())[:20]:  # Limit to 20 items
            display_key = key.replace('_', ' ').title()
            
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    # This is a list of objects - show sample keys
                    sample_keys = list(value[0].keys())[:5]
                    display_value = f"List ({len(value)} items) - Sample fields: {', '.join(sample_keys)}"
                else:
                    display_value = f"List ({len(value)} items)"
            elif isinstance(value, dict):
                display_value = f"Object ({len(value)} fields)"
            else:
                display_value = str(value)[:200] + ('...' if len(str(value)) > 200 else '')
            
            table_html.append(f'<tr><td class="summary-key">{display_key}</td><td class="summary-value">{display_value}</td></tr>')
        
        table_html.append('</table>')
        
        # Add debugging info for developers
        if any(isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) for v in data.values()):
            table_html.append('<p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">')
            table_html.append('⚠️ <strong>Note:</strong> This response contains list data that should be displayed as a table. ')
            table_html.append('The table formatter may need adjustment to extract the list properly.')
            table_html.append('</p>')
        
        table_html.append('</div>')
        
        return '\n'.join(table_html)
    
    def extract_search_term(self, query: str, search_keywords: List[str]) -> str:
        """Extract the actual search term from user query by removing search keywords."""
        query_lower = query.lower().strip()
        
        # Remove search keywords from the beginning or middle of the query
        for keyword in search_keywords:
            if query_lower.startswith(keyword):
                query_lower = query_lower[len(keyword):].strip()
                break
            elif keyword in query_lower:
                # Remove the keyword and everything before it
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    query_lower = parts[1].strip()
                break
        
        # For candidate name searches, preserve the full name
        if 'raja' in query_lower and 'doe' in query_lower:
            return 'raja doe'
        elif 'raja' in query_lower:
            return 'raja'
        elif 'doe' in query_lower:
            return 'doe'
        
        # For technology + role combinations
        if 'python developer' in query_lower:
            return 'python developer'
        elif 'java developer' in query_lower:
            return 'java developer'
        elif 'react developer' in query_lower:
            return 'react developer'
        elif 'angular developer' in query_lower:
            return 'angular developer'
        elif 'node developer' in query_lower:
            return 'node developer'
        elif 'full stack developer' in query_lower:
            return 'full stack developer'
        elif 'frontend developer' in query_lower:
            return 'frontend developer'
        elif 'backend developer' in query_lower:
            return 'backend developer'
        elif 'data engineer' in query_lower:
            return 'data engineer'
        elif 'software engineer' in query_lower:
            return 'software engineer'
        elif 'devops engineer' in query_lower:
            return 'devops engineer'
        
        # For specific role/technology searches, preserve the full term
        if 'java developer' in query_lower:
            return 'java developer'
        elif 'python developer' in query_lower:
            return 'python developer'
        elif 'react developer' in query_lower:
            return 'react developer'
        elif 'angular developer' in query_lower:
            return 'angular developer'
        elif 'data engineer' in query_lower:
            return 'data engineer'
        elif 'software engineer' in query_lower:
            return 'software engineer'
        elif 'java' in query_lower and 'developer' in query_lower:
            return 'java developer'
        elif 'developer' in query_lower:
            return 'developer'
        elif 'analyst' in query_lower:
            return 'analyst'
        
        # If we have location terms, simplify to just the city name
        if 'jaipur' in query_lower:
            return 'jaipur'
        elif 'delhi' in query_lower:
            return 'delhi'
        elif 'mumbai' in query_lower:
            return 'mumbai'
        elif 'bangalore' in query_lower or 'bengaluru' in query_lower:
            return 'bangalore'
        elif 'hyderabad' in query_lower:
            return 'hyderabad'
        
        # Remove common words that might interfere with search, but be more conservative
        stop_words = ['me', 'for', 'about', 'that', 'is', 'are', 'the', 'a', 'an', 'please', 'can', 'you', 'person', 'who', 'live', 'in', 'all']
        words = query_lower.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words) if filtered_words else query_lower
    
    def _extract_search_term_aggressive(self, query: str) -> str:
        """More aggressive search term extraction that tries to find meaningful terms."""
        query_lower = query.lower().strip()
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = ['search for', 'find', 'looking for', 'show me', 'get', 'find me', 'search', 'find all']
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break
        
        # Remove common suffixes
        suffixes_to_remove = ['candidate', 'candidates', 'job', 'jobs', 'position', 'positions', 'developer', 'developers']
        for suffix in suffixes_to_remove:
            if query_lower.endswith(suffix):
                query_lower = query_lower[:-len(suffix)].strip()
                break
        
        # Remove stop words more aggressively
        stop_words = ['me', 'for', 'about', 'that', 'is', 'are', 'the', 'a', 'an', 'please', 'can', 'you', 'person', 'who', 'live', 'in', 'all', 'some', 'any', 'with', 'and', 'or', 'but', 'of', 'to', 'from', 'by', 'at', 'on', 'in', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once']
        words = query_lower.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # If we still have meaningful words, return them
        if filtered_words:
            return ' '.join(filtered_words)
        
        # If no filtered words, try to extract any meaningful term
        # Look for technology names, programming languages, etc.
        tech_terms = ['java', 'python', 'javascript', 'react', 'angular', 'node', 'sql', 'html', 'css', 'php', 'ruby', 'go', 'rust', 'c++', 'c#', '.net', 'aws', 'azure', 'docker', 'kubernetes', 'devops', 'data', 'machine', 'learning', 'ai', 'ml']
        
        for tech in tech_terms:
            if tech in query_lower:
                return tech
        
        # Look for role terms
        role_terms = ['developer', 'engineer', 'analyst', 'manager', 'designer', 'architect', 'consultant', 'specialist', 'lead', 'senior', 'junior']
        
        for role in role_terms:
            if role in query_lower:
                return role
        
        # If nothing else works, return the original query cleaned up
        return query_lower.strip()
    
    def _direct_endpoint_routing(self, query_lower: str) -> List[str]:
        """Direct endpoint routing based on keywords - proven logic from working chatbot files."""
        selected_endpoints = []
        
        # Check for person name queries first - NEW FUNCTIONALITY
        if self._detect_person_name_query(query_lower):
            selected_endpoints.append('candidate_search')
            logger.info(f"🎯 Detected person name query, using candidate_search for: {query_lower}")
            # Extract the name for search term
            self.current_search_term = self._extract_name_from_query(query_lower)
            self.current_search_limit = 30  # Get more results for name searches
            return selected_endpoints
        
        # Check for time-based queries first
        time_period = TimeUtils.detect_time_period(query_lower)
        if time_period:
            # Store time period for later use in filtering
            self.current_time_period = time_period
            logger.info(f"🕐 Detected time period: {time_period}")
        else:
            self.current_time_period = None
        
        # Check for candidate skill/role-based queries
        skills = CandidateSearchUtils.extract_skills_from_query(query_lower)
        role = CandidateSearchUtils.extract_role_from_query(query_lower)
        
        # Extract number limit (e.g., "top 10", "first 5")
        import re
        limit_match = re.search(r'(?:top|first|best)\s+(\d+)', query_lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        
        if skills or role:
            self.current_search_skills = skills
            self.current_search_role = role
            self.current_search_limit = limit
            logger.info(f"🔍 Detected candidate search - Skills: {skills}, Role: {role}, Limit: {limit}")
        else:
            self.current_search_skills = []
            self.current_search_role = None
            self.current_search_limit = 10
        
        # Define keyword mappings (from chatbotv5v1 copy 3)
        endpoint_keywords = {
                         'job': {
                 'keywords': ['job', 'jobs', 'position', 'positions', 'opening', 'openings', 'vacancy', 'vacancies'],
                 'endpoints': ['jobs']  # Only show job listings
             },
            'client': {
                'keywords': ['client', 'clients', 'company', 'companies', 'customer', 'customers'],
                'endpoints': ['clients']
            },
            'vendor': {
                'keywords': ['vendor', 'vendors', 'supplier', 'suppliers', 'partner', 'partners'],
                'endpoints': ['vendors']
            },
            'user': {
                'keywords': ['user', 'users', 'employee', 'employees', 'staff', 'team member', 'team members'],
                'endpoints': ['users']
            },
            'candidate': {
                'keywords': ['candidate', 'candidates', 'applicant', 'applicants', 'talent', 'people'],
                'endpoints': ['candidates']  # Use list endpoint for general queries
            },
                         'dashboard': {
                 'keywords': ['dashboard', 'overview', 'analytics', 'metrics', 'report', 'performance'],
                 'endpoints': ['dashboard_overview']
             }
        }
        
        # Check for specific common queries first
        if "how many total candidates" in query_lower or ("how many" in query_lower and "candidates" in query_lower and "total" in query_lower):
            selected_endpoints.append('candidates')
            logger.info(f"🎯 Direct match: Total candidates query")
            return selected_endpoints
        
        if "how many jobs" in query_lower and any(period in query_lower for period in ['week', 'month', 'today', 'yesterday']):
            selected_endpoints.append('jobs')
            logger.info(f"🎯 Direct match: Jobs with time period")
            return selected_endpoints
        
        # Check for client total queries
        if ("how many" in query_lower and any(word in query_lower for word in ['client', 'clients'])) or \
           ("total" in query_lower and any(word in query_lower for word in ['client', 'clients'])):
            selected_endpoints.append('clients')
            logger.info(f"🎯 Direct match: Total clients query")
            return selected_endpoints
        
        # Check for vendor total queries
        if ("how many" in query_lower and any(word in query_lower for word in ['vendor', 'vendors'])) or \
           ("total" in query_lower and any(word in query_lower for word in ['vendor', 'vendors'])):
            selected_endpoints.append('vendors')
            logger.info(f"🎯 Direct match: Total vendors query")
            return selected_endpoints
        
        # Check for specific client searches
        if any(word in query_lower for word in ['client', 'clients']) and \
           any(search_word in query_lower for search_word in ['search', 'find', 'get me', 'show me']):
            # Extract search term for clients
            self.current_search_term = self._extract_client_vendor_search_term(query_lower, ['client', 'clients'])
            self.current_search_limit = limit if limit else 50
            selected_endpoints.append('client_search')
            logger.info(f"🎯 Direct match: Client search for '{self.current_search_term}'")
            return selected_endpoints
        
        # Check for specific vendor searches  
        if any(word in query_lower for word in ['vendor', 'vendors']) and \
           any(search_word in query_lower for search_word in ['search', 'find', 'get me', 'show me']):
            # Extract search term for vendors
            self.current_search_term = self._extract_client_vendor_search_term(query_lower, ['vendor', 'vendors'])
            self.current_search_limit = limit if limit else 50
            selected_endpoints.append('vendor_search')
            logger.info(f"🎯 Direct match: Vendor search for '{self.current_search_term}'")
            return selected_endpoints
        
        # Check for developer/role searches - these should use candidate_search
        developer_roles = ['developer', 'engineer', 'analyst', 'manager', 'designer', 'architect', 
                          'python', 'java', 'javascript', 'react', 'angular', 'node', 'sql', 
                          'data scientist', 'devops', 'frontend', 'backend', 'fullstack']
        
        if any(role in query_lower for role in developer_roles) or skills or role:
            # This is looking for candidates with specific skills/roles
            # Use candidate_search endpoint for skill-based searches
            selected_endpoints.append('candidate_search')
            logger.info(f"🎯 Using candidate_search for skill/role query")
            return selected_endpoints  # Return early to avoid duplicate endpoints
        
        # Check for resume requests
        if any(word in query_lower for word in ['resume', 'resumes', 'cv', 'curriculum vitae']):
            # If specific candidates are found, we'll need to fetch their resumes
            # This will be handled in post-processing after getting candidate list
            self.fetch_resumes = True
            logger.info(f"📄 Resume request detected")
        else:
            self.fetch_resumes = False
        
        # If we already have endpoints selected (like candidate_search), skip general matching
        if selected_endpoints:
            return selected_endpoints
        
        # Match keywords and add corresponding endpoints
        for category, config in endpoint_keywords.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                selected_endpoints.extend(config['endpoints'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_endpoints = []
        for endpoint in selected_endpoints:
            if endpoint not in seen and endpoint in self.endpoints:
                seen.add(endpoint)
                unique_endpoints.append(endpoint)
        
        return unique_endpoints
    
    def _extract_client_vendor_search_term(self, query_lower: str, keywords: List[str]) -> str:
        """Extract search term for client/vendor searches."""
        import re
        
        # Remove the keywords and common search words to get the actual search term
        remove_words = keywords + ['search', 'find', 'get', 'me', 'show', 'for', 'the', 'all', 'please', 'can', 'you']
        
        # Split query into words and remove common words
        words = query_lower.split()
        search_words = [word for word in words if word not in remove_words and len(word) > 2]
        
        # Join remaining words as search term
        search_term = ' '.join(search_words)
        
        # If no meaningful search term found, return empty string for general listing
        return search_term.strip()
    
    def _is_greeting(self, user_query: str) -> bool:
        """Check if the query is a simple greeting."""
        query_lower = user_query.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'hola', 'howdy']
        return query_lower in greetings or any(greeting in query_lower for greeting in greetings if len(query_lower) <= 20)

    def _generate_greeting_response(self, user_query: str) -> str:
        """Generate a friendly greeting response."""
        import random
        
        greetings = [
            "Hello! Great to see you! I'm your OATS recruitment assistant. How can I help you today?",
            "Hi there! I'm excited to help you with anything related to jobs, candidates, clients, or recruitment analytics. What would you like to explore?",
            "Hey! Welcome! I'm here to help you navigate through our OATS recruitment data. What can I show you?",
            "Hello! Nice to meet you! I can help you find information about jobs, candidates, dashboard metrics, and much more. What interests you?"
        ]
        
        follow_ups = [
            "Would you like to see our latest job postings?",
            "Should I show you our recruitment dashboard?",
            "Want to explore our candidate database?",
            "How about checking out some recruitment analytics?"
        ]
        
        greeting = random.choice(greetings)
        follow_up = random.choice(follow_ups)
        
        return f"<div class='ai-response'>{greeting}<br><br>🤔 {follow_up}</div>"

    def _is_history_inquiry(self, user_query: str) -> bool:
        """Check if the user is asking about chat history or previous questions."""
        query_lower = user_query.lower().strip()
        
        history_phrases = [
            'what did i ask', 'what have i asked', 'what was my last question',
            'previous question', 'previous questions', 'chat history', 'conversation history',
            'what did we discuss', 'what we talked about', 'our conversation',
            'my last query', 'earlier question', 'before i asked', 'previously asked',
            'show my questions', 'show chat', 'show conversation', 'show history',
            'remind me what', 'what questions did i', 'my previous', 'our previous'
        ]
        
        return any(phrase in query_lower for phrase in history_phrases)

    def _generate_history_response(self, user_query: str) -> str:
        """Generate a smart response about chat history using stored conversation data."""
        
        # Get chat history from file
        chat_history = self.get_chat_history()
        
        if not chat_history:
            return """<div class='ai-response'>
                <h3>📚 Chat History</h3>
                <p>This is the beginning of our conversation! You haven't asked me any questions yet.</p>
                <p>Feel free to ask me about:</p>
                <ul>
                    <li>📊 Job postings and openings</li>
                    <li>👥 Candidates and applicants</li>
                    <li>🏢 Client information</li>
                    <li>📈 Recruitment analytics</li>
                    <li>📋 Dashboard metrics</li>
                </ul>
            </div>"""
        
        # Get recent conversation history (last 5-10 interactions)
        recent_history = chat_history[-10:]
        
        # Create a formatted history summary using Gemini AI
        history_text = []
        for i, chat in enumerate(recent_history, 1):
            timestamp = chat.get('timestamp', 'Unknown time')[:19]  # Format: YYYY-MM-DD HH:MM:SS
            user_q = chat['user_query']
            bot_response = chat['bot_response'][:200] + "..." if len(chat['bot_response']) > 200 else chat['bot_response']
            
            history_text.append(f"{i}. [{timestamp}] You asked: '{user_q}'\n   I responded about: {bot_response}\n")
        
        # Use Gemini to create a smart summary
        try:
            gemini_prompt = f"""Based on this chat history, create a friendly, helpful summary for the user who asked "{user_query}". 

Recent conversation history:
{chr(10).join(history_text)}

Please provide:
1. A brief overview of what topics were discussed
2. Highlight 2-3 most recent or important questions
3. Format it as HTML with proper headings and styling
4. Be conversational and helpful
5. Use emojis and nice formatting
6. Keep it concise but informative
7. DO NOT include any "Quick Actions" buttons or sections
8. DO NOT include any action buttons or clickable elements

Format as a complete HTML response with <div class='ai-response'> wrapper. End with a closing </div> tag."""

            response = self.gemini_model.generate_content(gemini_prompt)
            
            # Clean up any Quick Actions that might still appear
            cleaned_response = response.text
            
            # Remove Quick Actions sections (comprehensive patterns)
            import re
            # Remove entire Quick Actions sections with various delimiters
            cleaned_response = re.sub(r'<[^>]*>\s*🖊\s*Quick Actions?:.*?</[^>]*>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'🖊\s*Quick Actions?:.*?(?=<(?:h[1-6]|div|p|br|$)|$)', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'Quick Actions?:.*?(?=<(?:h[1-6]|div|p|br|$)|$)', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove all button elements and action buttons
            cleaned_response = re.sub(r'<button[^>]*>.*?</button>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'<a[^>]*class[^>]*btn[^>]*>.*?</a>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove green box/panel sections that contain actions
            cleaned_response = re.sub(r'<[^>]*style[^>]*background[^>]*green[^>]*>.*?</[^>]*>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'<[^>]*class[^>]*action[^>]*>.*?</[^>]*>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any remaining action-related content
            cleaned_response = re.sub(r'Show Job Creation Steps', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'📝.*?Show.*?Steps', '', cleaned_response, flags=re.IGNORECASE)
            
            # Clean up extra whitespace and empty lines
            cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
            cleaned_response = re.sub(r'<p>\s*</p>', '', cleaned_response)
            
            return cleaned_response
        except Exception as e:
            logger.error(f"Error generating AI history summary: {e}")
            
            # Fallback: Create a simple formatted history
            history_html = """<div class='ai-response'>
                <h3>📚 Your Recent Chat History</h3>
                <div style='max-height: 400px; overflow-y: auto; padding: 10px; background: #f8f9fa; border-radius: 8px;'>"""
            
            for i, chat in enumerate(recent_history[-5:], 1):
                timestamp = chat.get('timestamp', 'Unknown time')[:16]  # Format: YYYY-MM-DD HH:MM
                user_q = chat['user_query']
                
                history_html += f"""
                    <div style='margin-bottom: 15px; padding: 10px; background: white; border-radius: 6px; border-left: 3px solid #007bff;'>
                        <strong>🕒 {timestamp}</strong><br>
                        <strong>You:</strong> {user_q}<br>
                        <small style='color: #666;'>✅ I provided information about this topic</small>
                    </div>"""
            
            history_html += """</div>
                <p><em>💡 You can ask me for more details about any of these topics, or ask something new!</em></p>
            </div>"""
            
            return history_html
            ai_resp = interaction.get('ai_response_full', interaction.get('ai_response_preview', ''))
            endpoints = interaction.get('endpoints_used', [])
            
            conversation_summary += f"\n{i}. At {timestamp}, you asked: \"{user_q}\""
            if endpoints:
                conversation_summary += f" (I used: {', '.join(endpoints)})"
        
        # Generate intelligent response using AI
        prompt = f"""
You are the OATS recruitment chatbot. A user is asking about their chat history. Here's our recent conversation:

{conversation_summary}

Generate a helpful response that:
1. Summarizes what they've asked about recently
2. Highlights key topics or patterns
3. Offers to continue or expand on any previous topic
4. Keep it conversational and helpful

User's current question: "{user_query}"
"""

        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            ai_response = model.generate_content(prompt).text
            
            return f"""<div class='ai-response'>
                <h3>📚 Our Conversation History</h3>
                {ai_response}
                <br><br>
                <strong>Recent Questions Summary:</strong>
                <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                {conversation_summary}
                </div>
                <em>💡 I can expand on any of these topics or help you with something new!</em>
            </div>"""
        except Exception as e:
            # Fallback response if AI fails
            return f"""<div class='ai-response'>
                <h3>📚 Our Conversation History</h3>
                <p>Here's what you've asked me recently:</p>
                <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                {conversation_summary}
                </div>
                <p>You've had <strong>{len(self.conversation_history)}</strong> total interactions with me.</p>
                <p>💡 I can expand on any of these topics or help you with something new!</p>
            </div>"""

    def _is_oats_related_query(self, user_query: str) -> bool:
        """Check if the query is related to OATS recruitment system data."""
        query_lower = user_query.lower().strip()
        
        # First check if it's a greeting
        if self._is_greeting(user_query):
            return True  # Treat greetings as valid (they get special handling)
        
        # OATS-related keywords
        oats_keywords = [
            'job', 'jobs', 'position', 'positions', 'opening', 'openings',
            'candidate', 'candidates', 'applicant', 'applicants', 'resume', 'resumes',
            'client', 'clients', 'vendor', 'vendors', 'company', 'companies',
            'dashboard', 'analytics', 'report', 'reports', 'data',
            'hire', 'hiring', 'recruit', 'recruiting', 'recruitment',
            'skill', 'skills', 'experience', 'qualification', 'qualifications',
            'interview', 'interviews', 'application', 'applications',
            'pipeline', 'submission', 'submissions', 'placement', 'placements',
            'user', 'users', 'role', 'roles', 'permission', 'permissions',
            'oats', 'ats', 'otomashen', 'how many', 'show me', 'list', 'find', 'search',
            'python', 'java', 'javascript', 'react', 'angular', 'data engineer', 'data scientist',
            'full stack', 'frontend', 'backend', 'devops', 'database', 'sql', 'aws', 'azure',
            'cid', 'jid', 'cli', 'created', 'updated', 'status', 'active', 'inactive',
            # Workflow and process keywords (for flow management)
            'jd', 'job description', 'create', 'post', 'how to', 'workflow', 'process', 'steps',
            'guide', 'tutorial', 'setup', 'configure', 'add', 'edit', 'delete', 'update',
            'manage', 'administration', 'system', 'procedure', 'instruction'
        ]
        
        # Check if query contains any OATS-related keywords
        for keyword in oats_keywords:
            if keyword in query_lower:
                return True
        
        # Check for ID patterns (CID123, JID456, etc.)
        id_patterns = [r'cid\d+', r'jid\d+', r'cli\d+', r'\w+\d+']
        for pattern in id_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _generate_off_topic_response(self, user_query: str) -> str:
        """Generate a polite response for non-OATS related questions."""
        responses = [
            f"I appreciate your question about '{user_query}', but I'm specifically designed to help with OATS recruitment system data. I can assist you with information about jobs, candidates, clients, dashboard analytics, and other recruitment-related queries.",
            f"That's an interesting question! However, I'm dedicated to helping with OATS recruitment system data only. I can help you find information about job postings, candidate profiles, client details, recruitment analytics, and much more within our system.",
            f"Thanks for asking! While I'd love to help with that, I'm specialized in OATS recruitment data. I can help you explore job opportunities, candidate information, client data, recruitment metrics, and other HR-related insights from our database.",
        ]
        
        import random
        base_response = random.choice(responses)
        
        follow_up_suggestions = [
            "Would you like me to show you our latest job postings?",
            "Can I help you search for candidates with specific skills?",
            "Would you like to see our recruitment dashboard overview?",
            "How about I show you some client information or recent hiring trends?",
            "Would you like to explore our candidate database or job pipeline?",
        ]
        
        follow_up = random.choice(follow_up_suggestions)
        
        return f"<div class='ai-response'>{base_response}<br><br>🤔 {follow_up}</div>"

    async def process_query(self, user_query: str) -> str:
        """Process query using direct routing for simple queries and LLM for complex ones."""
        print(f"🔍 Processing query: '{user_query}'")
        
        query_lower = user_query.lower().strip()
        
        # Check if it's a greeting first
        if self._is_greeting(user_query):
            print("👋 Greeting detected, providing friendly welcome")
            return self._generate_greeting_response(user_query)
        
        # Check if user is asking about chat history/previous questions
        if self._is_history_inquiry(user_query):
            print("📚 Chat history inquiry detected")
            return self._generate_history_response(user_query)
        
        # Check if query is OATS-related
        if not self._is_oats_related_query(user_query):
            print("🚫 Query is not OATS-related, providing polite redirection")
            return self._generate_off_topic_response(user_query)
        
        # Priority 0: Check for workflow/process guidance queries
        if self.flow_manager and self.flow_manager.is_flow_query(user_query):
            print("📋 Flow/workflow query detected, analyzing workflows...")
            # Use the enhanced smart response generation
            flow_response = self.flow_manager.generate_response(user_query)
            
            # For logging purposes, still get the analysis details
            flow_ids, confidence = self.flow_manager.analyze_user_query(user_query)
            print(f"✅ Generated workflow response (flows: {flow_ids}, confidence: {confidence:.2f})")
            
            # Add to conversation memory for flow guidance
            self.add_to_conversation_memory(
                user_query=user_query, 
                response_data={'workflow_guidance': flow_ids}, 
                endpoints_used=['flow_manager'], 
                ai_response=flow_response
            )
            return flow_response
        
        # Priority 1: Check conversation memory for context-aware responses
        # This helps with follow-up questions about previously shown data
        conversation_search = self.search_conversation_data(user_query)
        conversation_context = ""
        
        if conversation_search['found_data'] or conversation_search['matching_queries']:
            print(f"💭 Found relevant context in conversation history")
            conversation_context = f"""
CONVERSATION CONTEXT:
{conversation_search['relevant_context']}

RELEVANT PREVIOUS DATA:
"""
            # Add found data to context
            for found_item in conversation_search['found_data'][:5]:  # Limit to 5 most relevant items
                conversation_context += f"- From query '{found_item['source_query']}': {json.dumps(found_item['item'], indent=2)[:300]}...\n"
            
            # If user is asking about specific IDs/codes that were in previous results
            for pattern in [r'(JID\d+)', r'(CID\d+)', r'(CLI\d+)', r'(\w+\d+)']:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match and conversation_search['found_data']:
                    search_id = match.group(1).upper()
                    print(f"🎯 User asking about {search_id} from previous conversation")
                    # Generate response using conversation context
                    return self._generate_conversation_aware_response(user_query, conversation_search, search_id)
        
        # Priority 1: Handle specific IDs directly (proven logic from chatbotv5v1 copy 3)
        cid_match = re.search(r'(CID\d+)', query_lower, re.IGNORECASE)
        if cid_match:
            candidate_id = cid_match.group(1).upper()
            print(f"🎯 Intent Detected: Specific Candidate Details for ID: {candidate_id}")
            template = self.endpoints['candidate_details']
            # Format the URL from the template with the found ID
            specific_url = template.url.format(CidId=candidate_id)
            endpoint_to_call = APIEndpoint(url=specific_url, description=template.description, category=template.category)
            
            print(f"📡 Fetching from: {endpoint_to_call.url}")
            response = self.fetch_data_from_endpoint(endpoint_to_call)
            api_responses = {"candidate_details": response}
            ai_response = self.generate_ai_response(user_query, api_responses)
            
            # Add to conversation memory with AI response
            self.add_to_conversation_memory(
                user_query=user_query,
                response_data=api_responses,
                endpoints_used=['candidate_details'],
                ai_response=ai_response
            )
            
            return ai_response


        
        # Priority 2: Direct routing for simple keyword-based queries (from chatbotv5v1 copy 3)
        # This ensures reliable results for common queries
        direct_endpoints = self._direct_endpoint_routing(query_lower)
        if direct_endpoints:
            print(f"🎯 Direct routing selected endpoints: {', '.join(direct_endpoints)}")
            api_responses = await self.fetch_multiple_endpoints(direct_endpoints, user_query)
            
            # Log what we got back
            successful_endpoints = [k for k, v in api_responses.items() if v.success]
            if successful_endpoints:
                print(f"✅ Successfully fetched from: {successful_endpoints}")
                
                # Generate AI response
                ai_response = self.generate_ai_response(user_query, api_responses)
                
                # Add to conversation memory with AI response
                self.add_to_conversation_memory(
                    user_query=user_query,
                    response_data={k: v for k, v in api_responses.items() if v.success},
                    endpoints_used=direct_endpoints,
                    ai_response=ai_response
                )
                
                return ai_response
        
        # Priority 3: Use AI for complex queries that don't match direct routing
        print("🤖 Using AI to determine the best endpoints...")
        
        try:
            # Get relevant endpoints using LLM
            relevant_endpoint_keys = await self.get_relevant_endpoints(user_query)
            
            if not relevant_endpoint_keys:
                print("🤔 No relevant endpoints found, providing general guidance...")
                return self._generate_fallback_response(user_query)
            
            print(f"🎯 AI selected endpoints: {', '.join(relevant_endpoint_keys)}")
            
            # Fetch data from selected endpoints
            api_responses = await self.fetch_multiple_endpoints(relevant_endpoint_keys, user_query)
            
            # Log what we got back
            successful_endpoints = [k for k, v in api_responses.items() if v.success]
            failed_endpoints = [k for k, v in api_responses.items() if not v.success]
            
            print(f"✅ Successful endpoints: {successful_endpoints}")
            print(f"❌ Failed endpoints: {failed_endpoints}")
            
            if not successful_endpoints:
                print("❌ No data could be retrieved from any endpoints")
                return self._generate_fallback_response(user_query, failed_endpoints)
            
            # Generate AI response based on retrieved data
            print("🤖 Generating intelligent response...")
            ai_response = self.generate_ai_response(user_query, api_responses)
            
            # Add to conversation memory with AI response
            self.add_to_conversation_memory(
                user_query=user_query,
                response_data={k: v for k, v in api_responses.items() if v.success},
                endpoints_used=relevant_endpoint_keys,
                ai_response=ai_response
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return self._generate_fallback_response(user_query, [f"System error: {str(e)}"])
    
    def run_interactive_session(self):
        """Legacy method for command-line interface - use Flask routes instead."""
        print("⚠️  This method is deprecated. Please use the Flask web interface instead.")
        print("🌐 Start the Flask app and navigate to http://localhost:5000")
        return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.logout()
        if self.session:
            self.session.close()

# Global chatbot instance
chatbot_instance = None

def get_chatbot():
    """Get or create chatbot instance"""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = OATSChatbot()
        # Auto-login on first initialization
        try:
            success = chatbot_instance.login()
            if success:
                logger.info("Auto-login successful")
            else:
                logger.warning("Auto-login failed")
        except Exception as e:
            logger.error(f"Auto-login error: {e}")
    return chatbot_instance

# Workflow Button Utilities
def extract_workflow_buttons(response_text: str) -> List[Dict]:
    """Extract workflow-related content from response and create action buttons."""
    try:
        workflow_buttons = []
        response_lower = response_text.lower()
        
        # Only show buttons for VERY specific workflow-related responses
        # Check for the specific multiple workflow response pattern
        if '🎯 Multiple Relevant Processes Found' in response_text or '🎯 Relevant Process Found' in response_text:
            # Default button for general workflows
            workflow_buttons = [
                # {
                #     'workflow_id': "CREATE_JOB_POSTING",
                #     'text': "Response 1 Steps",
                #     'icon': "📝"
                # },
                {
                    'workflow_id': "ADD_NEW_CANDIDATE", 
                    'text': "Response",
                    'icon': "👤"
                }
            ]
            
            # Check if it's specifically about advanced search
            if 'Advanced Job Search in OATS' in response_text:
                workflow_buttons = [
                    {
                        'workflow_id': "ADVANCED_JOB_SEARCH_IN_OATS",
                        'text': "Response",
                        'icon': "🔍"
                    }
                ]
        
        # Check for specific JD/Job creation workflow patterns (only when explicitly asking about creation)
        elif (('how to create' in response_lower and ('jd' in response_lower or 'job' in response_lower)) or
              ('create jd' in response_lower) or 
              ('create job posting' in response_lower) or
              ('steps to create' in response_lower and 'job' in response_lower)):
            
            workflow_buttons = [
                {
                    'workflow_id': "CREATE_JOB_POSTING",
                    'text': "Show Job Creation Steps",
                    'icon': "📝"
                }
            ]
        
        # Check for advanced search workflows
        elif ('advance search' in response_lower or 'advanced search' in response_lower or 
              'advance job search' in response_lower or 'advanced job search' in response_lower):
            
            workflow_buttons = [
                {
                    'workflow_id': "ADVANCED_JOB_SEARCH_IN_OATS",
                    'text': "Response",
                    'icon': "🔍"
                }
            ]
        
        # Check for candidate-specific workflows
        # elif ('create candidate' in response_lower or 'add candidate' in response_lower):
            workflow_buttons = [
                {
                    'workflow_id': "ADD_NEW_CANDIDATE",
                    'button_text': "� Show Candidate Creation Steps", 
                    'keyword': 'create_candidate'
                }
            ]
        
        # Check for other common workflows
        # elif ('search candidates' in response_lower or 'find candidates' in response_lower):
            workflow_buttons = [
                {
                    'workflow_id': "SEARCH_CANDIDATES",
                    'button_text': "🔍 Show Candidate Search Steps",
                    'keyword': 'search_candidates'
                }
            ]
        
        # elif ('schedule interview' in response_lower):
            workflow_buttons = [
                {
                    'workflow_id': "SCHEDULE_INTERVIEW", 
                    'button_text': "� Show Interview Scheduling Steps",
                    'keyword': 'schedule_interview'
                }
            ]
        
        # elif ('dashboard' in response_lower):
            workflow_buttons = [
                {
                    'workflow_id': "POST-LOGIN_DASHBOARD_NAVIGATION",
                    'button_text': "📊 Show Dashboard Navigation Steps",
                    'keyword': 'dashboard'
                }
            ]
            
        return workflow_buttons
        
    except Exception as e:
        logger.error(f"Error extracting workflow buttons: {e}")
        return []

def get_workflow_details(workflow_id: str) -> Dict:
    """Get detailed steps for a specific workflow."""
    try:
        from flow_manager import OATSFlowManager
        
        # Initialize flow manager
        flow_manager = OATSFlowManager()
        
        # Get workflow details
        if workflow_id in flow_manager.flows_data:
            workflow = flow_manager.flows_data[workflow_id]
            return {
                'success': True,
                'workflow': {
                    'id': workflow_id,
                    'title': workflow.get('title', 'Unknown Workflow'),
                    'description': workflow.get('description', ''),
                    'steps': workflow.get('steps', []),
                    'prerequisites': workflow.get('prerequisites', []),
                    'tips': workflow.get('tips', []),
                    'related_flows': workflow.get('related_flows', []),
                    'screenshot': workflow.get('screenshot', '')
                }
            }
        else:
            # Fallback: search for similar workflow names
            for flow_id, flow_data in flow_manager.flows_data.items():
                if workflow_id.lower() in flow_id.lower() or any(workflow_id.lower() in keyword.lower() for keyword in flow_data.get('keywords', '').split(', ')):
                    return {
                        'success': True,
                        'workflow': {
                            'id': flow_id,
                            'title': flow_data.get('title', 'Unknown Workflow'),
                            'description': flow_data.get('description', ''),
                            'steps': flow_data.get('steps', []),
                            'prerequisites': flow_data.get('prerequisites', []),
                            'tips': flow_data.get('tips', []),
                            'related_flows': flow_data.get('related_flows', []),
                            'screenshot': flow_data.get('screenshot', '')
                        }
                    }
            
            return {
                'success': False,
                'error': f'Workflow "{workflow_id}" not found'
            }
    
    except Exception as e:
        logger.error(f"Error getting workflow details: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Flask Routes
@app.route('/')
def index():
    """Main chatbot interface"""
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login_route():
    """Handle login requests"""
    try:
        chatbot = get_chatbot()
        success = chatbot.login()
        
        if success:
            session['logged_in'] = True
            session['access_token'] = chatbot.access_token
            return jsonify({
                'success': True,
                'message': 'Login successful! You can now start asking questions.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Login failed. Please check your credentials.'
            }), 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during login.'
        }), 500

@app.route('/logout', methods=['POST'])
def logout_route():
    """Handle logout requests"""
    try:
        if 'logged_in' in session:
            chatbot = get_chatbot()
            chatbot.logout()
            session.clear()
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully.'
        })
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during logout.'
        }), 500

@app.route('/query', methods=['POST'])
def query_route():
    """Handle chatbot queries"""
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({
                'success': False,
                'message': 'Please log in first to ask questions.'
            }), 401
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'message': 'No query provided.'
            }), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({
                'success': False,
                'message': 'Query cannot be empty.'
            }), 400
        
        chatbot = get_chatbot()
        
        # Ensure chatbot is logged in
        if not chatbot.access_token:
            chatbot.access_token = session.get('access_token')
        
        # Process the query with timeout
        print(f"\n🚀 Processing query: '{user_query}'")
        
        try:
            if asyncio.iscoroutinefunction(chatbot.process_query):
                # Run with timeout to prevent hanging
                async def run_with_timeout():
                    return await asyncio.wait_for(chatbot.process_query(user_query), timeout=25.0)
                
                response = asyncio.run(run_with_timeout())
            else:
                response = chatbot.process_query(user_query)
            
            print("✅ Query processed successfully")
        except asyncio.TimeoutError:
            print("❌ Query timed out after 25 seconds")
            return jsonify({
                'success': False,
                'message': 'Request timed out. The API might be slow or returning too much data. Try a more specific query.'
            }), 504
        
        # Store chat in history (maintain rolling buffer of 30)
        try:
            chat_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_query': user_query,
                'bot_response': response,
                'success': True,
                'workflow_buttons': extract_workflow_buttons(response)
            }
            chatbot.store_chat_history(chat_entry)
        except Exception as e:
            logger.error(f"Error storing chat history: {e}")
        
        return jsonify({
            'success': True,
            'response': response,
            'query': user_query,
            'workflow_buttons': extract_workflow_buttons(response)
        })
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while processing your query. Please try again.'
        }), 500

@app.route('/workflow-details', methods=['POST'])
def workflow_details_route():
    """Get detailed steps for a specific workflow"""
    try:
        # Check if user is logged in
        if not session.get('logged_in'):
            return jsonify({
                'success': False,
                'message': 'Please log in first.'
            }), 401
        
        data = request.get_json()
        if not data or 'workflow_id' not in data:
            return jsonify({
                'success': False,
                'message': 'No workflow ID provided.'
            }), 400
        
        workflow_id = data['workflow_id'].strip()
        if not workflow_id:
            return jsonify({
                'success': False,
                'message': 'Workflow ID cannot be empty.'
            }), 400
        
        # Get workflow details
        result = get_workflow_details(workflow_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'message': result.get('error', 'Workflow not found.')
            }), 404
        
    except Exception as e:
        logger.error(f"Workflow details error: {e}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while fetching workflow details.'
        }), 500

@app.route('/auto-login', methods=['POST'])
def auto_login_route():
    """Auto-login route for seamless authentication"""
    try:
        chatbot = get_chatbot()
        
        # Check if already logged in
        if chatbot.access_token and session.get('logged_in'):
            return jsonify({
                'success': True,
                'message': 'Already logged in',
                'auto_login': True
            })
        
        # Attempt login
        success = chatbot.login()
        
        if success:
            session['logged_in'] = True
            session['access_token'] = chatbot.access_token
            return jsonify({
                'success': True,
                'message': 'Auto-login successful',
                'auto_login': True
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Auto-login failed',
                'auto_login': True
            }), 401
    except Exception as e:
        logger.error(f"Auto-login error: {e}")
        return jsonify({
            'success': False,
            'message': 'Auto-login error occurred',
            'auto_login': True
        }), 500

@app.route('/debug-query', methods=['POST'])
def debug_query_route():
    """Debug route to see raw API responses"""
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        query = data.get('query', 'show me clients')
        
        chatbot = get_chatbot()
        
        # Get relevant endpoints
        if asyncio.iscoroutinefunction(chatbot.get_relevant_endpoints):
            endpoint_keys = asyncio.run(chatbot.get_relevant_endpoints(query))
        else:
            endpoint_keys = chatbot.get_relevant_endpoints(query)
        
        # Fetch raw data
        raw_responses = asyncio.run(chatbot.fetch_multiple_endpoints(endpoint_keys, query))
        
        # Return raw structure for debugging
        debug_info = {}
        for key, response in raw_responses.items():
            if response.success:
                debug_info[key] = {
                    'status': 'success',
                    'data_type': type(response.data).__name__,
                    'data_keys': list(response.data.keys()) if isinstance(response.data, dict) else 'Not a dict',
                    'data_sample': str(response.data)[:500] + '...' if len(str(response.data)) > 500 else str(response.data)
                }
            else:
                debug_info[key] = {
                    'status': 'failed',
                    'error': response.error_message
                }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Debug query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversation-history')
def conversation_history_route():
    """Get conversation history summary"""
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Not logged in'}), 401
        
        chatbot = get_chatbot()
        
        # Return lightweight conversation summary
        history_summary = []
        for interaction in chatbot.conversation_history[-5:]:  # Last 5 interactions
            history_summary.append({
                'timestamp': interaction['timestamp'],
                'query': interaction['user_query'],
                'endpoints_used': interaction['endpoints_used'],
                'data_types': interaction['data_types_returned'],
                'record_counts': interaction['record_counts']
            })
        
        return jsonify({
            'recent_history': history_summary,
            'total_queries': len(chatbot.conversation_history),
            'context': chatbot.user_context
        })
        
    except Exception as e:
        logger.error(f"Conversation history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status_route():
    """Check chatbot and login status with conversation info"""
    try:
        chatbot = get_chatbot()
        is_logged_in = session.get('logged_in', False) and chatbot.access_token is not None
        
        return jsonify({
            'logged_in': is_logged_in,
            'gemini_available': chatbot.gemini_model is not None,
            'session_active': 'access_token' in session,
            'auto_login_enabled': True,
            'conversation_count': len(chatbot.conversation_history) if is_logged_in else 0,
            'session_duration': int((time.time() - chatbot.user_context['session_start']) / 60) if is_logged_in else 0
        })
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({
            'logged_in': False,
            'gemini_available': False,
            'session_active': False,
            'auto_login_enabled': True,
            'conversation_count': 0,
            'session_duration': 0
        })

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint that returns structured JSON data from queries."""
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        chatbot = get_chatbot()
        
        # Process the query and get endpoints
        relevant_endpoints = chatbot.get_relevant_endpoints(query)
        if not relevant_endpoints:
            return jsonify({
                'query': query,
                'error': 'No relevant endpoints found',
                'response': 'Could not determine which data to fetch'
            }), 404
        
        # Fetch data from endpoints
        api_responses = chatbot.fetch_data_from_endpoints(relevant_endpoints)
        
        # Process the responses into structured format
        structured_data = {
            'query': query,
            'endpoints_used': list(relevant_endpoints.keys()),
            'data_summary': {},
            'raw_data': {}
        }
        
        for endpoint_name, response in api_responses.items():
            if response.success and response.data:
                data = response.data
                
                # Extract key information based on data structure
                if isinstance(data, dict) and 'Data' in data:
                    # Paginated response - prioritize TotalItems over Count
                    total_count = data.get('TotalItems') or data.get('Count', 0)
                    
                    structured_data['data_summary'][endpoint_name] = {
                        'total_count': total_count,
                        'current_page': data.get('CurrentPage') or data.get('Page', 1),
                        'items_per_page': data.get('ItemsPerPage') or data.get('PerPage', 10),
                        'total_pages': data.get('TotalPages', 0),
                        'records_returned': len(data.get('Data', [])),
                        'actual_total': total_count  # Explicit field for total
                    }
                    
                    # Debug log
                    if 'candidate' in endpoint_name.lower():
                        print(f"📊 API Query - Candidate Total: {total_count} (from {'TotalItems' if 'TotalItems' in data else 'Count'})")
                    
                    # Process records based on type
                    if 'candidate' in endpoint_name.lower():
                        structured_data['data_summary'][endpoint_name]['type'] = 'candidates'
                        structured_data['data_summary'][endpoint_name]['sample_records'] = []
                        
                        for candidate in data.get('Data', [])[:10]:  # First 10 records
                            structured_data['data_summary'][endpoint_name]['sample_records'].append({
                                'id': candidate.get('CidId'),
                                'name': f"{candidate.get('FirstName', '')} {candidate.get('LastName', '')}".strip(),
                                'job_title': candidate.get('CurrentJobTitle'),
                                'location': candidate.get('Location'),
                                'experience': f"{candidate.get('ExperienceYears', 0)} years",
                                'skills': candidate.get('PrimarySkills') or candidate.get('Skills'),
                                'company': candidate.get('CurrentCompany'),
                                'source': candidate.get('Source')
                            })
                    elif 'job' in endpoint_name.lower():
                        structured_data['data_summary'][endpoint_name]['type'] = 'jobs'
                        structured_data['data_summary'][endpoint_name]['sample_records'] = []
                        
                        for job in data.get('Data', [])[:10]:
                            structured_data['data_summary'][endpoint_name]['sample_records'].append({
                                'id': job.get('job_id') or job.get('JobId'),
                                'title': job.get('job_title') or job.get('JobTitle'),
                                'client': job.get('client_name') or job.get('ClientName'),
                                'location': job.get('location') or job.get('Location'),
                                'type': job.get('job_type') or job.get('JobType'),
                                'status': job.get('status') or job.get('Status')
                            })
                    elif 'client' in endpoint_name.lower():
                        structured_data['data_summary'][endpoint_name]['type'] = 'clients'
                        structured_data['data_summary'][endpoint_name]['sample_records'] = []
                        
                        for client in data.get('Data', [])[:10]:
                            structured_data['data_summary'][endpoint_name]['sample_records'].append({
                                'id': client.get('client_id') or client.get('ClientId'),
                                'name': client.get('client_name') or client.get('ClientName'),
                                'company': client.get('company_name') or client.get('CompanyName'),
                                'email': client.get('email') or client.get('Email'),
                                'location': client.get('location') or client.get('Location')
                            })
                    elif 'vendor' in endpoint_name.lower():
                        structured_data['data_summary'][endpoint_name]['type'] = 'vendors'
                        structured_data['data_summary'][endpoint_name]['sample_records'] = []
                        
                        for vendor in data.get('Data', [])[:10]:
                            structured_data['data_summary'][endpoint_name]['sample_records'].append({
                                'id': vendor.get('vendor_id') or vendor.get('VendorId'),
                                'name': vendor.get('vendor_name') or vendor.get('VendorName'),
                                'company': vendor.get('company_name') or vendor.get('CompanyName'),
                                'email': vendor.get('email') or vendor.get('Email'),
                                'type': vendor.get('vendor_type') or vendor.get('VendorType')
                            })
                    
                    # Include raw data if requested
                    if data.get('include_raw', False):
                        structured_data['raw_data'][endpoint_name] = data
                else:
                    # Non-paginated response
                    structured_data['data_summary'][endpoint_name] = {
                        'type': 'simple',
                        'data': data
                    }
        
        # Add conversation to history
        chatbot.add_to_conversation_memory(query, relevant_endpoints, api_responses)
        
        return jsonify(structured_data)
        
    except Exception as e:
        logger.error(f"API query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/structured-query', methods=['POST'])
def structured_query():
    """New API endpoint that returns structured JSON data for frontend unified handling."""
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        chatbot = get_chatbot()
        
        # Use the main process_query method which includes conversation memory
        if asyncio.iscoroutinefunction(chatbot.process_query):
            ai_response = asyncio.run(chatbot.process_query(query))
        else:
            ai_response = chatbot.process_query(query)
        
        # Return the AI response in the structured format expected by the frontend
        structured_response = {
            'query': query,
            'ai_response': ai_response,
            'endpoints_used': ['conversation_memory'],  # Placeholder since process_query handles routing internally
            'success': True
        }
        
        return jsonify(structured_response)
        
    except Exception as e:
        logger.error(f"Error in structured_query: {e}")
        return jsonify({
            'query': query if 'query' in locals() else '',
            'error': str(e),
            'ai_response': f"<div class='ai-response error'>I encountered an error processing your request: {str(e)}</div>",
            'success': False
        }), 500


# Memory Monitoring Routes
@app.route('/memory-monitor')
def memory_monitor():
    """Memory monitoring dashboard"""
    return render_template('memory_monitor.html')


@app.route('/live_memory.json')
def live_memory_json():
    """Serve live memory data as JSON"""
    try:
        if os.path.exists('live_memory.json'):
            with open('live_memory.json', 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({
                'status': 'not_initialized',
                'last_update': datetime.now().isoformat(),
                'current_memory': {
                    'conversations': 0,
                    'cached_data': 0,
                    'memory_usage_mb': 0,
                    'latest_query': None
                }
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'last_update': datetime.now().isoformat(),
            'current_memory': {}
        }), 500


@app.route('/memory_logs.json')
def memory_logs_json():
    """Serve detailed memory logs as JSON"""
    try:
        if os.path.exists('memory_logs.json'):
            with open('memory_logs.json', 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({
                'tracking_started': datetime.now().isoformat(),
                'total_interactions': 0,
                'memory_snapshots': [],
                'performance_metrics': []
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Global chatbot instance for reuse
chatbot = None


def get_chatbot():
    """Get or create a chatbot instance."""
    global chatbot
    if chatbot is None:
        chatbot = OATSChatbot()
    return chatbot

@app.route('/api/chat-history')
def chat_history_api():
    """API endpoint to serve chat history for memory monitor"""
    try:
        chatbot = get_chatbot()
        chat_history = chatbot.get_chat_history()
        
        return jsonify({
            'success': True,
            'chat_history': chat_history,
            'total_chats': len(chat_history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'chat_history': [],
            'total_chats': 0
        }), 500

@app.route('/api/chat-history-formatted')
def chat_history_formatted_api():
    """API endpoint to serve formatted chat history for memory monitor"""
    try:
        chatbot = get_chatbot()
        chat_history = chatbot.get_chat_history()
        
        # Format chat history for display
        formatted_history = []
        for entry in chat_history:
            formatted_entry = {
                'id': entry.get('timestamp', ''),
                'timestamp': entry.get('timestamp', ''),
                'user_query': entry.get('user_query', ''),
                'bot_response': entry.get('bot_response', ''),
                'success': entry.get('success', True),
                'formatted_time': entry.get('timestamp', '').replace('T', ' ').replace('Z', '') if entry.get('timestamp') else ''
            }
            formatted_history.append(formatted_entry)
        
        return jsonify({
            'success': True,
            'chat_history': formatted_history,
            'total_chats': len(formatted_history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving formatted chat history: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'chat_history': [],
            'total_chats': 0
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


# Flow Management Routes
@app.route('/api/flows', methods=['GET'])
def list_flows():
    """List all available workflows"""
    try:
        chatbot = get_chatbot()
        if not chatbot.flow_manager:
            return jsonify({'error': 'Flow management not available'}), 503
        
        flows_info = []
        for flow_id, flow_data in chatbot.flow_manager.flows_data.items():
            flows_info.append({
                'id': flow_id,
                'title': flow_data['title'],
                'description': flow_data['description'],
                'keywords': flow_data['keywords'],
                'related_flows': flow_data.get('related_flows', [])
            })
        
        return jsonify({
            'flows': flows_info,
            'total_count': len(flows_info)
        })
        
    except Exception as e:
        logger.error(f"Error listing flows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/flows/<flow_id>', methods=['GET'])
def get_flow(flow_id):
    """Get detailed information about a specific flow"""
    try:
        chatbot = get_chatbot()
        if not chatbot.flow_manager:
            return jsonify({'error': 'Flow management not available'}), 503
        
        flow_data = chatbot.flow_manager.get_flow_by_id(flow_id)
        if not flow_data:
            return jsonify({'error': 'Flow not found'}), 404
        
        return jsonify(flow_data)
        
    except Exception as e:
        logger.error(f"Error getting flow {flow_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-flows')
def test_flows():
    """Test page for workflow system"""
    return render_template('test_flows.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main application entry point"""
    print("🚀 Starting OATS Job Management Chatbot Flask App...")
    print("🌐 Access the chatbot at: http://localhost:5000")
    print("📋 Available endpoints:")
    print("   GET  /           - Main chatbot interface")
    print("   POST /login      - Login to the system")
    print("   POST /logout     - Logout from the system")
    print("   POST /query      - Send queries to the chatbot")
    print("   GET  /status     - Check system status")
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"📁 Created templates directory: {templates_dir}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
    