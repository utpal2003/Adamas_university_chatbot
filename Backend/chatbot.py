import torch
import numpy as np
from typing import Dict, Any, List
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import Config
from knowledge_base import UniversityKnowledgeBase
from chatbot.extractors import HybridContentExtractor
from chatbot.utils import ResponseFormatter, IntentRefiner, TelecallerEscalation

class UniversityChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.tag_responses = None
        self.knowledge_base = None
        self.content_extractor = HybridContentExtractor()
        self.response_formatter = ResponseFormatter()
        self.intent_refiner = IntentRefiner()
        self.telecaller_escalation = TelecallerEscalation()

        logger.info(f"🚀 Initializing University Chatbot on device: {self.device}")
        
        # Load components
        self.load_model()
        self.setup_knowledge_base()

    def load_model(self):
        """Load the trained model with PyTorch 2.6 compatibility"""
        try:
            from transformers import AutoTokenizer

            # Check if model file exists
            if not os.path.exists(Config.MODEL_SAVE_PATH):
                logger.error(f"Model file not found: {Config.MODEL_SAVE_PATH}")
                logger.info("Please train the model first using: python train.py")
                return False

            logger.info(f"📦 Loading model from: {Config.MODEL_SAVE_PATH}")
            
            # Load checkpoint with weights_only=False for PyTorch 2.6 compatibility
            checkpoint = torch.load(
                Config.MODEL_SAVE_PATH, 
                map_location=self.device, 
                weights_only=False
            )

            # Load metadata
            self.label_encoder = checkpoint['label_encoder']
            self.tag_responses = checkpoint['tag_responses']

            # Import the model class from train.py
            from train import OptimizedUniversityClassifier

            # Initialize model
            n_classes = len(self.label_encoder.classes_)
            self.model = OptimizedUniversityClassifier(n_classes=n_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

            logger.info(f"✅ Model loaded successfully! {n_classes} classes: {list(self.label_encoder.classes_)}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Please train the model first using: python train.py")
            return False

    def setup_knowledge_base(self):
        """Setup knowledge base connection"""
        try:
            self.knowledge_base = UniversityKnowledgeBase()
            stats = self.knowledge_base.get_stats()
            logger.info(f"✅ Knowledge base connected: {stats}")
        except Exception as e:
            logger.warning(f"Knowledge base not available: {e}")
            self.knowledge_base = None

    def predict_intent(self, text: str) -> tuple:
        """Predict intent for given text"""
        if self.model is None or self.tokenizer is None:
            logger.warning("Model or tokenizer not loaded")
            return "error", 0.0

        try:
            # Tokenize input
            encoding = self.tokenizer(
                text.lower(),
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_LENGTH,
                return_tensors='pt'
            )

            # Predict
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                intent = self.label_encoder.inverse_transform([predicted.cpu().item()])[0]
                confidence_score = confidence.cpu().item()

            logger.info(f"🎯 Predicted intent: {intent} (confidence: {confidence_score:.3f})")
            return intent, confidence_score

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "error", 0.0

    def get_predefined_response(self, intent: str) -> str:
        """Get predefined response for intent"""
        if self.tag_responses and intent in self.tag_responses:
            responses = self.tag_responses[intent]
            response = np.random.choice(responses)
            logger.info(f"📝 Using predefined response for {intent}")
            return response
        else:
            logger.warning(f"No predefined response found for intent: {intent}")
            return "I'm not sure how to help with that. Could you please rephrase your question?"

    def summarize_content(self, content: str, query: str, intent: str = None) -> str:
        """Use the hybrid content extractor for smart summarization"""
        logger.info(f"🤖 Using hybrid content extractor for query: '{query}'")
        return self.content_extractor.extract_content(content, query, intent)

    def get_knowledge_response(self, intent: str, query: str) -> str:
        """Get response from knowledge base with hybrid extraction"""
        if self.knowledge_base is None:
            logger.warning("Knowledge base not available, using fallback")
            return self.get_predefined_response(intent)

        # Refine intent using keywords for better accuracy
        refined_intent = self.intent_refiner.refine_intent(intent, query)
        
        logger.info(f"🔍 Searching knowledge base for: '{query}' (refined intent: {refined_intent})")
        
        # Search knowledge base
        results = self.knowledge_base.search(query, n_results=3)

        if not results:
            logger.warning("❌ No results from knowledge base search")
            return self.get_predefined_response(refined_intent)

        logger.info(f"✅ Found {len(results)} results from knowledge base")
        
        # Use the most relevant result
        best_match = results[0]
        content = best_match.get('content', '').strip()
        
        if not content:
            logger.warning("❌ Empty content in best match")
            return self.get_predefined_response(refined_intent)

        logger.info(f"📄 Raw content length: {len(content)} characters")
        
        # Use hybrid content extraction
        summarized_content = self.summarize_content(content, query, refined_intent)
        logger.info(f"📋 Extracted content length: {len(summarized_content)} characters")
        
        # Return the formatted response
        final_response = self.response_formatter.format_response(refined_intent, summarized_content)
        logger.info(f"🤖 Final response ready")
        
        return final_response

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process user message and generate response"""
        logger.info(f"📨 Processing message: '{user_input}'")
        
        # Predict intent
        intent, confidence = self.predict_intent(user_input)

        response_data = {
            'user_input': user_input,
            'intent': intent,
            'confidence': round(confidence, 3),
            'response': '',
            'source': 'model'
        }

        # Generate response based on intent type
        if intent in Config.KNOWLEDGE_INTENTS:
            # Use knowledge base for knowledge intents
            response_data['response'] = self.get_knowledge_response(intent, user_input)
            response_data['source'] = 'knowledge_base'
        else:
            # Use predefined responses for non-knowledge intents
            response_data['response'] = self.get_predefined_response(intent)
        
        # Check for telecaller escalation
        response_data['requires_followup'] = self.telecaller_escalation.should_escalate(response_data, user_input)
        
        logger.info(f"✅ Response generated (source: {response_data['source']})")
        return response_data

    def should_escalate_to_telecaller(self, response_data: Dict[str, Any], user_input: str) -> bool:
        """Determine if conversation should be escalated to telecaller"""
        return self.telecaller_escalation.should_escalate(response_data, user_input)


def main():
    """Main function to run the chatbot"""
    chatbot = UniversityChatbot()
    
    # Test the chatbot
    print("\n🤖 University Chatbot Ready!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            break
            
        if user_input:
            response = chatbot.process_message(user_input)
            print(f"\nBot: {response['response']}")
            print(f"   [Intent: {response['intent']}, Confidence: {response['confidence']}, Source: {response['source']}]")
            print()


if __name__ == "__main__":
    main()