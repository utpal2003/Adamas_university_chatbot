import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """Handles response formatting based on intent"""
    
    @staticmethod
    def format_response(intent: str, content: str) -> str:
        """Format response based on intent"""
        response_templates = {
            # Core Academic Intents
            'admission_requirements': f"🎓 **Admission Requirements:**\n\n{content}",
            'admission_process': f"📝 **Admission Process:**\n\n{content}",
            'course_information': f"📚 **Course Information:**\n\n{content}",
            'program_details': f"🎯 **Program Details:**\n\n{content}",
            'department_info': f"🏛️ **Department Information:**\n\n{content}",
            'academic_calendar': f"📅 **Academic Calendar:**\n\n{content}",

            # Financial Intents
            'fees_structure': f"💰 **Fee Structure:**\n\n{content}",
            'scholarship_info': f"🎗️ **Scholarship Information:**\n\n{content}",
            'financial_aid': f"💳 **Financial Aid:**\n\n{content}",
            'payment_options': f"💵 **Payment Options:**\n\n{content}",

            # Campus & Facilities
            'facilities': f"🏫 **Campus Facilities:**\n\n{content}",
            'campus_information': f"🌳 **Campus Information:**\n\n{content}",
            'building_locations': f"🗺️ **Building Locations:**\n\n{content}",
            'library_info': f"📖 **Library Information:**\n\n{content}",
            'hostel_info': f"🏠 **Hostel Information:**\n\n{content}",
            'sports_facilities': f"⚽ **Sports Facilities:**\n\n{content}",
            'laboratories': f"🔬 **Laboratories:**\n\n{content}",

            # Transportation
            'transport_info': f"🚌 **Transport Information:**\n\n{content}",
            'bus_schedule': f"⏰ **Bus Schedule:**\n\n{content}",
            'campus_transport': f"🚍 **Campus Transport:**\n\n{content}",
            'parking_info': f"🅿️ **Parking Information:**\n\n{content}",

            # Career & Placement
            'placement_info': f"💼 **Placement Information:**\n\n{content}",
            'internship_opportunities': f"🔍 **Internship Opportunities:**\n\n{content}",
            'career_services': f"🎯 **Career Services:**\n\n{content}",
            'placement_records': f"📊 **Placement Records:**\n\n{content}",

            # Contact & Support
            'contact_info': f"📞 **Contact Information:**\n\n{content}",
            'helpdesk_support': f"🛠️ **Helpdesk Support:**\n\n{content}",
            'administration_contacts': f"👔 **Administration Contacts:**\n\n{content}",
            'department_contacts': f"📞 **Department Contacts:**\n\n{content}",
        }

        formatted_response = response_templates.get(intent, f"ℹ️ **Information:**\n\n{content}")
        logger.info(f"📤 Formatted response for {intent}")
        return formatted_response

class IntentRefiner:
    """Handles intent refinement using keyword matching"""
    
    @staticmethod
    def refine_intent(original_intent: str, query: str) -> str:
        """Refine intent using keyword matching for better accuracy"""
        query_lower = query.lower()
        
        # Keyword to intent mapping
        keyword_intent_map = {
            'library': 'library_info',
            'book': 'library_info',
            'study': 'library_info',
            'hostel': 'hostel_info',
            'accommodation': 'hostel_info',
            'room': 'hostel_info',
            'bus': 'transport_info',
            'transport': 'transport_info',
            'route': 'transport_info',
            'fee': 'fees_structure',
            'cost': 'fees_structure',
            'price': 'fees_structure',
            'tuition': 'fees_structure',
            'scholarship': 'scholarship_info',
            'financial': 'scholarship_info',
            'admission': 'admission_requirements',
            'admit': 'admission_requirements',
            'eligibility': 'admission_requirements',
            'course': 'course_information',
            'program': 'course_information',
            'degree': 'course_information',
            'placement': 'placement_info',
            'job': 'placement_info',
            'career': 'placement_info',
            'contact': 'contact_info',
            'phone': 'contact_info',
            'email': 'contact_info'
        }
        
        # Check for keywords in query
        for keyword, intent in keyword_intent_map.items():
            if keyword in query_lower:
                logger.info(f"🔍 Refined intent from '{original_intent}' to '{intent}' based on keyword: {keyword}")
                return intent
        
        return original_intent

class TelecallerEscalation:
    """Handles telecaller escalation logic"""
    
    @staticmethod
    def should_escalate(response_data: Dict[str, Any], user_input: str) -> bool:
        """Determine if conversation should be escalated to telecaller"""
        user_input_lower = user_input.lower()
        
        # Conditions for telecaller escalation
        escalation_conditions = [
            # Low confidence in prediction
            response_data.get('confidence', 0) < 0.4,
            
            # User explicitly asks for human help
            any(word in user_input_lower for word in [
                'human', 'person', 'agent', 'representative', 
                'talk to someone', 'speak with', 'real person',
                'call me', 'contact me'
            ]),
            
            # Complex queries that might need human assistance
            any(phrase in user_input_lower for phrase in [
                'detailed information', 'more information', 'explain in detail',
                'personal guidance', 'counseling', 'career guidance',
                'visit campus', 'campus tour', 'meet'
            ]),
            
            # Admission-related complex queries
            any(phrase in user_input_lower for phrase in [
                'help with admission', 'admission guidance', 'application help',
                'document verification', 'admission process help'
            ]),
            
            # Financial queries that might need personal discussion
            any(phrase in user_input_lower for phrase in [
                'payment plan', 'installment', 'education loan',
                'financial assistance', 'fee concession'
            ])
        ]
        
        should_escalate = any(escalation_conditions)
        logger.info(f"📞 Telecaller escalation check: {should_escalate}")
        
        return should_escalate