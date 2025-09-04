"""
OATS Chatbot Flow Management System
Handles workflow-based queries and provides step-by-step guidance
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OATSFlowManager:
    """Manages workflow detection and response generation for OATS system processes."""
    
    def __init__(self, gemini_model=None):
        self.gemini_model = gemini_model
        self.flows_file = "flows/oats_workflows.txt"
        self.flows_data = {}
        self.load_flows()
    
    def load_flows(self):
        """Load all workflows from the text file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            flows_path = os.path.join(current_dir, self.flows_file)
            
            if not os.path.exists(flows_path):
                logger.warning(f"Flows file not found: {flows_path}")
                return
            
            with open(flows_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse the flows from the text content
            self.flows_data = self._parse_flows_content(content)
            logger.info(f"Loaded {len(self.flows_data)} workflows successfully")
            
        except Exception as e:
            logger.error(f"Error loading flows: {e}")
            self.flows_data = {}
    
    def _parse_flows_content(self, content: str) -> Dict[str, Dict]:
        """Parse the flows text content into structured data."""
        flows = {}
        
        # Split content by flow separators
        flow_sections = re.split(r'=== FLOW_ID: (.+?) ===', content)
        
        # Process each flow section
        for i in range(1, len(flow_sections), 2):
            if i + 1 < len(flow_sections):
                flow_id = flow_sections[i].strip()
                flow_content = flow_sections[i + 1].strip()
                
                # Parse individual flow
                flow_data = self._parse_single_flow(flow_content)
                flow_data['flow_id'] = flow_id
                flows[flow_id] = flow_data
        
        return flows
    
    def _parse_single_flow(self, content: str) -> Dict:
        """Parse a single flow section into structured data."""
        flow_data = {
            'title': '',
            'keywords': [],
            'description': '',
            'steps': [],
            'related_flows': []
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('TITLE:'):
                flow_data['title'] = line.replace('TITLE:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keywords_text = line.replace('KEYWORDS:', '').strip()
                flow_data['keywords'] = [k.strip() for k in keywords_text.split(',')]
            elif line.startswith('DESCRIPTION:'):
                flow_data['description'] = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('STEPS:'):
                current_section = 'steps'
            elif line.startswith('RELATED_FLOWS:'):
                related_text = line.replace('RELATED_FLOWS:', '').strip()
                flow_data['related_flows'] = [f.strip() for f in related_text.split(',')]
            elif current_section == 'steps' and line:
                # Remove step numbering and clean up
                step_text = re.sub(r'^\d+\.?\s*', '', line)
                if step_text:
                    flow_data['steps'].append(step_text)
        
        return flow_data
    
    def analyze_user_query(self, user_query: str) -> Tuple[List[str], float]:
        """
        Enhanced Gemini analysis for more accurate workflow matching.
        Returns: (list_of_flow_ids, confidence_score)
        """
        if not self.gemini_model:
            return self._enhanced_fallback_detection(user_query)
        
        # Create comprehensive flow context for Gemini
        flow_context = self._create_enhanced_flow_context()
        
        prompt = f"""
You are an expert OATS recruitment system analyst. Your job is to precisely match user queries to the most relevant workflows based on actual screenshot data and real system workflows.

USER QUERY: "{user_query}"

WORKFLOW DATABASE:
{flow_context}

ENHANCED ANALYSIS REQUIREMENTS:
1. **Intent Recognition**: Determine the user's exact intent and goal
2. **Context Awareness**: Consider the full context of OATS recruitment processes
3. **Precision Matching**: Match based on specific actions, not just keywords
4. **Multi-step Process Recognition**: Identify if query relates to part of a larger workflow
5. **User Experience Focus**: Prioritize workflows that directly solve the user's problem

MATCHING CRITERIA (Score 0.0 to 1.0):
- **Exact Process Match** (0.9-1.0): Query directly asks for a specific workflow
- **High Relevance** (0.8-0.89): Query strongly relates to workflow outcomes
- **Moderate Relevance** (0.6-0.79): Query partially matches workflow context
- **Low Relevance** (0.3-0.59): Query has some connection but not primary focus
- **No Match** (0.0-0.29): Query doesn't relate to this workflow

SMART MATCHING RULES:
✅ "how to create JD" → CREATE_JOB_POSTING (0.95)
✅ "add new candidate" → ADD_NEW_CANDIDATE (0.92)
✅ "find java developers" → SEARCH_CANDIDATES (0.88)
✅ "schedule interview" → SCHEDULE_INTERVIEW (0.90)
✅ "advanced search options" → ADVANCED_JOB_SEARCH_IN_OATS (0.85)
✅ "manage clients" → CLIENT_MANAGEMENT (0.87)
✅ "dashboard overview" → DASHBOARD_NAVIGATION (0.83)

CONTEXT-AWARE MATCHING:
- If user asks about "creating" something, prioritize creation workflows
- If user asks about "finding/searching", prioritize search workflows  
- If user asks about "managing", prioritize management workflows
- If user asks about specific OATS features, match to relevant sections

RESPONSE FORMAT (JSON only):
{{
    "matched_flows": ["MOST_RELEVANT_FLOW_ID"],
    "confidence": 0.XX,
    "reasoning": "Specific explanation of match logic",
    "user_intent": "What the user wants to accomplish",
    "suggested_action": "Next step recommendation"
}}

IMPORTANT: 
- Return only ONE primary workflow unless query clearly needs multiple processes
- Be highly selective - better to have one accurate match than multiple irrelevant ones
- Focus on actionable workflows that solve the user's immediate need
- Use confidence scores that reflect actual match quality

Analyze the query and return ONLY the JSON response:
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response to extract JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].strip()
            
            # Parse JSON response
            if result_text.startswith('{') and result_text.endswith('}'):
                result = json.loads(result_text)
                flow_ids = result.get('matched_flows', [])
                confidence = result.get('confidence', 0.0)
                reasoning = result.get('reasoning', '')
                
                # Validate flow IDs exist
                valid_flows = [fid for fid in flow_ids if fid in self.flows_data]
                
                # Log the analysis for debugging
                logger.info(f"Gemini Analysis - Query: '{user_query}' | Flows: {valid_flows} | Confidence: {confidence} | Reasoning: {reasoning}")
                
                return valid_flows, confidence
            else:
                logger.warning(f"Invalid JSON response from Gemini: {result_text}")
                return self._enhanced_fallback_detection(user_query)
                
        except Exception as e:
            logger.error(f"Error in enhanced Gemini analysis: {e}")
            return self._enhanced_fallback_detection(user_query)
    
    def _create_enhanced_flow_context(self) -> str:
        """Create enhanced flow context with more details for better matching."""
        context_parts = []
        
        for flow_id, flow_data in self.flows_data.items():
            # Include first few steps to give Gemini more context
            steps_preview = flow_data['steps'][:5] if flow_data['steps'] else []
            steps_text = '\n  '.join([f"{i+1}. {step}" for i, step in enumerate(steps_preview)])
            
            context_parts.append(f"""
FLOW_ID: {flow_id}
TITLE: {flow_data['title']}
DESCRIPTION: {flow_data['description']}
KEYWORDS: {', '.join(flow_data['keywords'][:10])}
KEY_STEPS_PREVIEW:
  {steps_text}
  {"  ... additional steps available" if len(flow_data['steps']) > 5 else ""}
RELATED_FLOWS: {', '.join(flow_data['related_flows'])}
""")
        
        return '\n'.join(context_parts)
    
    def _enhanced_fallback_detection(self, user_query: str) -> Tuple[List[str], float]:
        """Enhanced fallback detection with better scoring and context awareness."""
        query_lower = user_query.lower()
        flow_scores = {}
        
        # Enhanced keyword mappings for better matching
        action_keywords = {
            'create': ['create', 'add', 'new', 'make', 'build', 'generate'],
            'search': ['search', 'find', 'look', 'locate', 'discover'],
            'manage': ['manage', 'edit', 'update', 'modify', 'change'],
            'schedule': ['schedule', 'book', 'arrange', 'set up', 'plan'],
            'import': ['import', 'upload', 'bulk', 'mass', 'batch'],
            'view': ['view', 'see', 'display', 'show', 'list']
        }
        
        entity_keywords = {
            'job': ['job', 'jd', 'position', 'role', 'posting'],
            'candidate': ['candidate', 'applicant', 'talent', 'resume'],
            'client': ['client', 'customer', 'company', 'employer'],
            'interview': ['interview', 'meeting', 'call', 'discussion'],
            'vendor': ['vendor', 'supplier', 'partner'],
            'dashboard': ['dashboard', 'overview', 'summary', 'home']
        }
        
        for flow_id, flow_data in self.flows_data.items():
            score = 0
            
            # 1. Direct keyword matching (high weight)
            for keyword in flow_data['keywords']:
                if keyword.lower() in query_lower:
                    score += 3
            
            # 2. Title word matching (medium-high weight)
            title_words = flow_data['title'].lower().split()
            for word in title_words:
                if len(word) > 3 and word in query_lower:
                    score += 2
            
            # 3. Action + Entity combinations (very high weight)
            for action, action_synonyms in action_keywords.items():
                for entity, entity_synonyms in entity_keywords.items():
                    action_match = any(synonym in query_lower for synonym in action_synonyms)
                    entity_match = any(synonym in query_lower for synonym in entity_synonyms)
                    
                    if action_match and entity_match:
                        # Check if this flow matches the action+entity combination
                        flow_title_lower = flow_data['title'].lower()
                        flow_keywords_lower = [k.lower() for k in flow_data['keywords']]
                        
                        action_in_flow = any(synonym in flow_title_lower for synonym in action_synonyms)
                        entity_in_flow = any(synonym in flow_title_lower for synonym in entity_synonyms) or \
                                       any(synonym in ' '.join(flow_keywords_lower) for synonym in entity_synonyms)
                        
                        if action_in_flow and entity_in_flow:
                            score += 5
            
            # 4. Description matching (medium weight)
            if flow_data['description']:
                desc_words = flow_data['description'].lower().split()
                query_words = query_lower.split()
                common_words = set(desc_words) & set(query_words)
                score += len(common_words) * 0.5
            
            # 5. Steps content matching (low weight)
            steps_text = ' '.join(flow_data['steps']).lower()
            query_words = [w for w in query_lower.split() if len(w) > 3]
            for word in query_words:
                if word in steps_text:
                    score += 0.5
            
            # 6. Special patterns and phrases
            special_patterns = {
                'advanced search': ['ADVANCED_JOB_SEARCH_IN_OATS'],
                'job posting': ['CREATE_JOB_POSTING'],
                'add candidate': ['ADD_NEW_CANDIDATE'],
                'create candidate': ['ADD_NEW_CANDIDATE'],
                'find candidates': ['SEARCH_CANDIDATES'],
                'schedule interview': ['SCHEDULE_INTERVIEW'],
                'client management': ['CLIENT_MANAGEMENT'],
                'dashboard': ['POST-LOGIN_DASHBOARD_NAVIGATION'],
                'main page': ['POST-LOGIN_DASHBOARD_NAVIGATION']
            }
            
            for pattern, target_flows in special_patterns.items():
                if pattern in query_lower and flow_id in target_flows:
                    score += 4
            
            if score > 0:
                flow_scores[flow_id] = score
        
        # Select best matches
        if not flow_scores:
            return [], 0.0
        
        # Sort by score and take top matches
        sorted_flows = sorted(flow_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence based on score distribution
        max_score = sorted_flows[0][1]
        confidence = min(0.9, max_score / 10.0)  # Cap at 0.9 for fallback
        
        # Return only the best match for focused results
        best_flow = sorted_flows[0][0]
        
        return [best_flow], confidence
    
    def generate_response(self, user_query: str) -> str:
        """Generate an intelligent, context-aware response for user queries."""
        flow_ids, confidence = self.analyze_user_query(user_query)
        
        if not flow_ids:
            return self._generate_no_flow_response(user_query)
        
        if len(flow_ids) == 1:
            # Single workflow - provide detailed, personalized response
            flow_data = self.flows_data[flow_ids[0]]
            return self._generate_smart_single_flow_response(user_query, flow_data, confidence)
        else:
            # Multiple workflows - show relevant options
            flow_list = [self.flows_data[fid] for fid in flow_ids]
            return self._generate_smart_multiple_flow_response(user_query, flow_list, confidence)
    
    def generate_flow_response(self, user_query: str, flow_ids: List[str], confidence: float) -> str:
        """Legacy function - redirects to enhanced generate_response."""
        return self.generate_response(user_query)
    
    def _generate_smart_single_flow_response(self, user_query: str, flow_data: Dict, confidence: float) -> str:
        """Generate a smart, context-aware response for a single workflow."""
        
        # Analyze query intent for personalization
        query_lower = user_query.lower()
        intent_context = self._analyze_query_intent(query_lower)
        
        # Generate personalized introduction
        intro = self._generate_personalized_intro(user_query, flow_data, intent_context)
        
        html_response = f"""
<div class='ai-response flow-guidance'>
<h3>🎯 {flow_data['title']}</h3>
{intro}

<div class='workflow-content'>
<h4>📋 Complete Process:</h4>
<div class='steps-container'>
"""
        
        # Add steps with smart formatting
        for i, step in enumerate(flow_data['steps'], 1):
            # Highlight important steps based on query context
            step_class = self._get_step_importance(step, intent_context)
            html_response += f"""
<div class='step-item {step_class}'>
    <span class='step-number'>{i}</span>
    <span class='step-content'>{step}</span>
</div>
"""
        
        html_response += "</div></div>"
        
        # Add smart recommendations
        recommendations = self._generate_smart_recommendations(flow_data, intent_context)
        if recommendations:
            html_response += f"""
<div class='smart-recommendations'>
<h4>💡 Smart Recommendations:</h4>
{recommendations}
</div>
"""
        
        # Add related flows with context
        if flow_data.get('related_flows'):
            html_response += f"""
<div class='related-flows'>
<h4>🔗 You Might Also Need:</h4>
<p>{', '.join(flow_data['related_flows'])}</p>
</div>
"""
        
        # Smart footer based on confidence and context
        footer_message = self._generate_smart_footer(confidence, intent_context)
        html_response += f"""
<div class='flow-footer'>
{footer_message}
</div>
</div>
"""
        
        return html_response
    
    def _analyze_query_intent(self, query_lower: str) -> Dict:
        """Analyze user query to understand intent and context."""
        intent = {
            'urgency': 'normal',
            'experience_level': 'intermediate',
            'specific_focus': [],
            'action_type': 'general'
        }
        
        # Detect urgency
        if any(word in query_lower for word in ['urgent', 'quickly', 'asap', 'immediately', 'fast']):
            intent['urgency'] = 'high'
        elif any(word in query_lower for word in ['learn', 'understand', 'explore']):
            intent['urgency'] = 'low'
        
        # Detect experience level
        if any(word in query_lower for word in ['new', 'beginner', 'first time', 'never', 'how to start']):
            intent['experience_level'] = 'beginner'
        elif any(word in query_lower for word in ['advanced', 'expert', 'detailed', 'comprehensive']):
            intent['experience_level'] = 'advanced'
        
        # Detect action type
        if any(word in query_lower for word in ['create', 'add', 'new', 'make']):
            intent['action_type'] = 'create'
        elif any(word in query_lower for word in ['find', 'search', 'look']):
            intent['action_type'] = 'search'
        elif any(word in query_lower for word in ['manage', 'edit', 'update']):
            intent['action_type'] = 'manage'
        
        # Detect specific focus areas
        focus_keywords = {
            'technical': ['technical', 'skills', 'programming', 'developer'],
            'process': ['process', 'workflow', 'steps', 'procedure'],
            'management': ['management', 'team', 'client', 'relationship'],
            'reporting': ['report', 'analytics', 'metrics', 'performance']
        }
        
        for focus, keywords in focus_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent['specific_focus'].append(focus)
        
        return intent
    
    def _generate_personalized_intro(self, user_query: str, flow_data: Dict, intent_context: Dict) -> str:
        """Generate a personalized introduction based on user intent."""
        
        if intent_context['urgency'] == 'high':
            intro = f"<p><strong>🚀 Quick Guide:</strong> Here's how to {flow_data['description'].lower()} efficiently:</p>"
        elif intent_context['experience_level'] == 'beginner':
            intro = f"<p><strong>👋 Welcome!</strong> I'll guide you step-by-step through {flow_data['description'].lower()}. Don't worry, it's easier than it looks!</p>"
        elif intent_context['experience_level'] == 'advanced':
            intro = f"<p><strong>🔧 Advanced Guide:</strong> Here's the complete process for {flow_data['description'].lower()} with all the details:</p>"
        else:
            intro = f"<p><strong>📝 Process Guide:</strong> {flow_data['description']}</p>"
        
        return intro
    
    def _get_step_importance(self, step: str, intent_context: Dict) -> str:
        """Determine step importance based on context."""
        step_lower = step.lower()
        
        # Highlight steps related to user's focus
        if intent_context['action_type'] == 'create' and any(word in step_lower for word in ['create', 'add', 'new']):
            return 'highlight-step'
        elif intent_context['action_type'] == 'search' and any(word in step_lower for word in ['search', 'find', 'filter']):
            return 'highlight-step'
        elif any(focus in step_lower for focus in intent_context['specific_focus']):
            return 'highlight-step'
        
        return 'normal-step'
    
    def _generate_smart_recommendations(self, flow_data: Dict, intent_context: Dict) -> str:
        """Generate smart recommendations based on workflow and user intent."""
        recommendations = []
        
        if intent_context['experience_level'] == 'beginner':
            recommendations.append("💡 Take your time with each step - accuracy is more important than speed")
            recommendations.append("📚 Consider bookmarking this workflow for future reference")
        
        if intent_context['urgency'] == 'high':
            recommendations.append("⚡ Focus on the essential steps first, you can refine details later")
            recommendations.append("🎯 Keep all required information ready before starting")
        
        if 'technical' in intent_context['specific_focus']:
            recommendations.append("🔧 Double-check technical specifications and requirements")
        
        if 'management' in intent_context['specific_focus']:
            recommendations.append("👥 Consider notifying relevant team members about this process")
        
        return '<br>'.join([f"<span class='recommendation'>{rec}</span>" for rec in recommendations])
    
    def _generate_smart_footer(self, confidence: float, intent_context: Dict) -> str:
        """Generate an intelligent footer based on confidence and context."""
        
        if confidence >= 0.9:
            confidence_msg = "This is exactly what you're looking for! 🎯"
        elif confidence >= 0.7:
            confidence_msg = "This should help with your request 👍"
        else:
            confidence_msg = "This might be what you need - let me know if you'd like something else 🤔"
        
        if intent_context['urgency'] == 'high':
            help_msg = "Need immediate help with any step? Just ask!"
        else:
            help_msg = "Questions about any step? Feel free to ask for clarification!"
        
        return f"""
<p><strong>{confidence_msg}</strong></p>
<p>💬 {help_msg}</p>
<p class='confidence-indicator'>🤖 Confidence: {confidence:.0%}</p>
"""
        
        return html_response
    
    def _generate_smart_multiple_flow_response(self, user_query: str, flow_list: List[Dict], confidence: float) -> str:
        """Generate smart response for multiple workflows with contextual ranking."""
        
        # Analyze query intent for smart presentation
        query_lower = user_query.lower()
        intent_context = self._analyze_query_intent(query_lower)
        
        html_response = f"""
<div class='ai-response flow-guidance'>
<h3>🎯 Multiple Relevant Processes Found</h3>
<p><strong>Smart Analysis:</strong> Based on your query "<em>{user_query}</em>", I found several relevant workflows. Here's the most suitable one:</p>
"""
        
        # Show only the most relevant workflow (first one) in detail
        if flow_list:
            primary_flow = flow_list[0]
            html_response += f"""
<div class='primary-flow-recommendation'>
<h4>🌟 Recommended: {primary_flow['title']}</h4>
<p><strong>Why this matches:</strong> {primary_flow['description']}</p>

<div class='quick-steps-preview'>
<h5>📋 Key Steps Preview:</h5>
<ol class='condensed-steps'>
"""
            
            # Show first 4 steps with smart highlighting
            for i, step in enumerate(primary_flow['steps'][:4], 1):
                step_class = self._get_step_importance(step, intent_context)
                highlight_class = " class='important-step'" if step_class == 'highlight-step' else ""
                html_response += f"<li{highlight_class}><strong>Step {i}:</strong> {step}</li>\n"
            
            if len(primary_flow['steps']) > 4:
                html_response += f"<li><em>... plus {len(primary_flow['steps']) - 4} more detailed steps</em></li>\n"
            
            html_response += "</ol></div></div>"
            
            # Show alternative workflows if there are more
            if len(flow_list) > 1:
                html_response += f"""
<div class='alternative-workflows'>
<h5>🔄 Alternative Workflows:</h5>
<div class='alternatives-list'>
"""
                for alt_flow in flow_list[1:3]:  # Show max 2 alternatives
                    html_response += f"""
<div class='alternative-item'>
    <h6>{alt_flow['title']}</h6>
    <p class='alt-description'>{alt_flow['description']}</p>
</div>
"""
                html_response += "</div></div>"
        
        # Smart recommendations based on context
        recommendations = self._generate_context_recommendations(intent_context, len(flow_list))
        if recommendations:
            html_response += f"""
<div class='context-recommendations'>
<h5>💡 Smart Suggestions:</h5>
{recommendations}
</div>
"""
        
        # Smart footer with next steps
        footer_message = self._generate_multiple_flow_footer(confidence, intent_context, len(flow_list))
        html_response += f"""
<div class='flow-footer'>
{footer_message}
</div>
</div>
"""
        
        return html_response
    
    def _generate_context_recommendations(self, intent_context: Dict, num_flows: int) -> str:
        """Generate contextual recommendations for multiple flow scenarios."""
        recommendations = []
        
        if intent_context['experience_level'] == 'beginner':
            recommendations.append("🎯 Start with the recommended workflow above - it's the most straightforward approach")
            recommendations.append("📚 Once comfortable, explore the alternative workflows for different scenarios")
        
        if intent_context['urgency'] == 'high':
            recommendations.append("⚡ Focus on the primary recommendation first - it's the fastest path to your goal")
            recommendations.append("⏰ Save alternative workflows for future reference")
        
        if num_flows > 3:
            recommendations.append("🔍 Multiple options available - ask me to narrow down based on your specific needs")
        
        if 'technical' in intent_context['specific_focus']:
            recommendations.append("🔧 Consider the technical requirements of each workflow before choosing")
        
        return '<br>'.join([f"<span class='context-rec'>{rec}</span>" for rec in recommendations])
    
    def _generate_multiple_flow_footer(self, confidence: float, intent_context: Dict, num_flows: int) -> str:
        """Generate smart footer for multiple flow responses."""
        
        if confidence >= 0.8:
            confidence_msg = f"Found {num_flows} relevant workflows - the top recommendation is highly accurate! 🎯"
        elif confidence >= 0.6:
            confidence_msg = f"Found {num_flows} potentially relevant workflows 👍"
        else:
            confidence_msg = f"Found {num_flows} possible matches - let me know if you need something more specific 🤔"
        
        if intent_context['urgency'] == 'high':
            help_msg = "💬 Say 'show me [workflow name]' for immediate detailed steps!"
        else:
            help_msg = "💬 Ask me to 'elaborate on [workflow name]' or 'show alternatives' for more options!"
        
        return f"""
<p><strong>{confidence_msg}</strong></p>
<p>{help_msg}</p>
<p class='confidence-indicator'>🤖 Multi-Flow Analysis | Confidence: {confidence:.0%}</p>
"""
    
    def _generate_single_flow_response(self, user_query: str, flow_data: Dict, confidence: float) -> str:
        """Generate response for a single workflow."""
        html_response = f"""
<div class='ai-response flow-guidance'>
<h3>🎯 {flow_data['title']}</h3>
<p><em>{flow_data['description']}</em></p>

<h4>📝 Step-by-Step Process:</h4>
<ol class='flow-steps'>
"""
        
        for i, step in enumerate(flow_data['steps'], 1):
            html_response += f"<li><strong>Step {i}:</strong> {step}</li>\n"
        
        html_response += "</ol>\n"
        
        # Add related flows if available
        if flow_data.get('related_flows'):
            html_response += f"""
<h4>🔗 Related Processes:</h4>
<p>You might also be interested in: {', '.join(flow_data['related_flows'])}</p>
"""
        
        # Add helpful footer
        html_response += f"""
<div class='flow-footer'>
<p>💡 <strong>Need more help?</strong> Ask me about any specific step or related process!</p>
<p>🤖 <em>Confidence: {confidence:.0%} - This guidance was generated based on your query: "{user_query}"</em></p>
</div>
</div>
"""
        
        return html_response
    
    def _generate_multiple_flow_response(self, user_query: str, flow_list: List[Dict], confidence: float) -> str:
        """Generate response when multiple workflows are relevant."""
        html_response = f"""
<div class='ai-response flow-guidance'>
<h3>🎯 Relevant Process Found</h3>
<p>Based on your query "<em>{user_query}</em>", here is the relevant workflow:</p>
"""
        
        for i, flow_data in enumerate(flow_list, 1):
            # Comment out the first response (i == 1) to hide it
            if i == 1:
                continue  # Skip the first response
            
            html_response += f"""
<div class='flow-summary'>
<h4>{flow_data['title']}</h4>
<p><strong>Description:</strong> {flow_data['description']}</p>
<p><strong>Key Steps:</strong></p>
<ul>
"""
            # Show first 3 steps as preview
            for step in flow_data['steps'][:3]:
                html_response += f"<li>{step}</li>\n"
            
            if len(flow_data['steps']) > 3:
                html_response += f"<li><em>... and {len(flow_data['steps']) - 3} more steps</em></li>\n"
            
            html_response += "</ul>\n</div>\n"
        
        html_response += f"""
<div class='flow-footer'>
<p>💡 <strong>Want detailed steps?</strong> Ask me specifically about any of these processes!</p>
<p>🤖 <em>Confidence: {confidence:.0%} - Ask me to elaborate on any specific workflow.</em></p>
</div>
</div>
"""
        
        return html_response
    
    def _generate_no_flow_response(self, user_query: str) -> str:
        """Generate response when no relevant flows are found."""
        available_flows = list(self.flows_data.keys())
        
        return f"""
<div class='ai-response flow-guidance'>
<h3>🤔 No Specific Workflow Found</h3>
<p>I couldn't find a specific workflow for: "<em>{user_query}</em>"</p>

<h4>📚 Available Workflows:</h4>
<ul>
"""+ '\n'.join([f"<li>{self.flows_data[fid]['title']}</li>" for fid in available_flows[:8]]) + f"""
</ul>

<p>💡 <strong>Try asking:</strong></p>
<ul>
<li>"How to create a job posting"</li>
<li>"How do I search for candidates"</li>
<li>"Show me interview scheduling process"</li>
<li>"Client management workflow"</li>
</ul>

<p>🤖 <em>Or ask me about any specific process you need help with!</em></p>
</div>
"""

    def is_flow_query(self, user_query: str) -> bool:
        """Determine if a user query is asking for workflow guidance."""
        flow_indicators = [
            'how to', 'how do i', 'how can i', 'steps to', 'process for',
            'workflow', 'procedure', 'guide', 'tutorial', 'instructions',
            'create', 'add', 'schedule', 'manage', 'setup', 'configure'
        ]
        
        query_lower = user_query.lower()
        return any(indicator in query_lower for indicator in flow_indicators)
    
    def get_flow_by_id(self, flow_id: str) -> Optional[Dict]:
        """Get a specific flow by its ID."""
        return self.flows_data.get(flow_id)
    
    def search_flows_by_keyword(self, keyword: str) -> List[Dict]:
        """Search flows by keyword."""
        keyword_lower = keyword.lower()
        matching_flows = []
        
        for flow_data in self.flows_data.values():
            # Check if keyword appears in title, keywords, or description
            if (keyword_lower in flow_data['title'].lower() or
                any(keyword_lower in kw.lower() for kw in flow_data['keywords']) or
                keyword_lower in flow_data['description'].lower()):
                matching_flows.append(flow_data)
        
        return matching_flows
