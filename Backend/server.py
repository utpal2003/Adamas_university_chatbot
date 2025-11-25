from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from chatbot import UniversityChatbot
    from config import Config
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create mock classes for testing
    class UniversityChatbot:
        def __init__(self):
            self.model = None
            self.knowledge_base = None
            self.label_encoder = None
            logger.info("Mock chatbot initialized")
        
        def process_message(self, message):
            return {
                'response': f"Mock response to: {message}",
                'intent': 'general',
                'confidence': 0.9,
                'source': 'mock'
            }
        
        def should_escalate_to_telecaller(self, response_data, user_message):
            return "fee" in user_message.lower() or "admission" in user_message.lower()

    class Config:
        KNOWLEDGE_INTENTS = ['fees', 'admission', 'courses', 'hostel']

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize chatbot
print("🚀 Initializing University Chatbot...")
try:
    chatbot = UniversityChatbot()
    print("✅ Chatbot initialized successfully")
except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")
    chatbot = UniversityChatbot()  # Fallback to mock

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    """Chat API endpoint for frontend"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required',
                'data': None
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty',
                'data': None
            }), 400
        
        print(f"📨 Received message: {user_message}")
        
        # Process the message
        response_data = chatbot.process_message(user_message)
        
        # Determine if telecaller is needed
        requires_followup = chatbot.should_escalate_to_telecaller(response_data, user_message)
        
        # Format response for frontend
        response = {
            'success': True,
            'data': {
                'response': response_data['response'],
                'intent': response_data.get('intent', 'general'),
                'confidence': response_data.get('confidence', 0.9),
                'source': response_data.get('source', 'model'),
                'requiresFollowup': requires_followup
            },
            'user_message': user_message
        }
        
        print(f"🤖 Bot response: {response_data['response']}")
        print(f"📞 Requires followup: {requires_followup}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat_message: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'data': None
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model_status = chatbot.model is not None
        kb_status = chatbot.knowledge_base is not None
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'components': {
                'model_loaded': model_status,
                'knowledge_base_connected': kb_status
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/intents', methods=['GET'])
def get_intents():
    """Get available intents"""
    try:
        if hasattr(chatbot, 'label_encoder') and chatbot.label_encoder:
            intents = list(chatbot.label_encoder.classes_)
            knowledge_intents = Config.KNOWLEDGE_INTENTS
            
            return jsonify({
                'success': True,
                'intents': intents,
                'knowledge_intents': knowledge_intents,
                'total_intents': len(intents),
                'knowledge_based_intents': len(knowledge_intents)
            })
        else:
            return jsonify({
                'success': True,
                'intents': ['general', 'admission', 'fees', 'courses', 'hostel'],
                'knowledge_intents': Config.KNOWLEDGE_INTENTS,
                'total_intents': 5,
                'knowledge_based_intents': len(Config.KNOWLEDGE_INTENTS),
                'note': 'Using default intents'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    """Get knowledge base statistics"""
    try:
        if hasattr(chatbot, 'knowledge_base') and chatbot.knowledge_base:
            stats = chatbot.knowledge_base.get_stats()
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({
                'success': True,
                'stats': {
                    'total_documents': 0,
                    'status': 'Knowledge base not connected',
                    'note': 'Running in mock mode'
                }
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Telecaller escalation endpoint
@app.route('/api/telecaller/request', methods=['POST'])
def request_telecaller():
    """Handle telecaller connection requests"""
    try:
        data = request.get_json()
        
        if not data or 'user_message' not in data:
            return jsonify({
                'success': False,
                'error': 'User message is required'
            }), 400
        
        user_message = data['user_message']
        user_contact = data.get('contact', 'Not provided')
        
        # Here you would typically:
        # 1. Save to database
        # 2. Send email notification
        # 3. Integrate with CRM system
        
        print(f"📞 TELEALLER REQUEST: User asked: '{user_message}' | Contact: {user_contact}")
        
        return jsonify({
            'success': True,
            'message': 'Telecaller request received successfully. Our team will contact you shortly.',
            'request_id': f"TCL_{os.urandom(4).hex().upper()}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to process telecaller request: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🤖 University Chatbot API Server Starting...")
    print("📍 Endpoints:")
    print("   GET  /api/health          - Health check")
    print("   GET  /api/intents         - List available intents")
    print("   GET  /api/knowledge/stats - Knowledge base statistics")
    print("   POST /api/chat/message    - Send message to chatbot")
    print("   POST /api/telecaller/request - Request telecaller connection")
    print("   GET  /                    - Serve frontend")
    print("\n🚀 Server running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)