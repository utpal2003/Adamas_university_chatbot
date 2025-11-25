import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 🎯 OPTIMIZED MODEL CONFIG
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 8  # Increased for better convergence
    LEARNING_RATE = 1e-4  # Higher learning rate
    NUM_EPOCHS = 15  # More epochs for better learning
    HIDDEN_DROPOUT = 0.3

    # Paths
    MODEL_SAVE_PATH = "./models/model.pth"
    TOKENIZER_PATH = "./models/tokenizer"
    INTENTS_PATH = "./data/intents.json"
    KNOWLEDGE_PATH = "./knowledge"

    # ChromaDB Configuration
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    COLLECTION_NAME = "university_knowledge"

    # Training Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Response Configuration
    CONFIDENCE_THRESHOLD = 0.7

    KNOWLEDGE_INTENTS = [
        # Core Academic Intents
        "admission_requirements",
        "admission_process",
        "course_information",
        "program_details",
        "department_info",
        "academic_calendar",

        # Financial Intents
        "fees_structure",
        "scholarship_info",
        "financial_aid",
        "payment_options",

        # Campus & Facilities
        "facilities",
        "campus_information",
        "building_locations",
        "library_info",
        "hostel_info",
        "sports_facilities",
        "laboratories",

        # Faculty & Staff
        "faculty_info",
        "staff_directory",
        "department_heads",

        # Transportation
        "transport_info",
        "bus_schedule",
        "campus_transport",
        "parking_info",

        # Career & Placement
        "placement_info",
        "internship_opportunities",
        "career_services",
        "placement_records",

        # Contact & Support
        "contact_info",
        "helpdesk_support",
        "administration_contacts",
        "department_contacts",

        # Rules & Policies
        "university_rules",
        "academic_policies",
        "code_of_conduct",
        "attendance_policy",

        # Examination
        "exam_process",
        "exam_schedule",
        "grading_system",
        "result_declaration",

        # General Information
        "university_overview",
        "college_location",
        "campus_tour",
        "events_activities",

        # Student Life
        "student_clubs",
        "cultural_activities",
        "campus_events",

        # Technical Support
        "it_services",
        "wifi_info",
        "computer_labs",

        # Emergency & Health
        "emergency_contacts",
        "health_services",
        "counselling_services"
    ]



# docker run -d -p 8000:8000 --name chroma-db chromadb/chroma