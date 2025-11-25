import re
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SmartContentExtractor:
    def __init__(self):
        self.query_patterns = {
            'fee': r'(₹|\$|rs?\.?)\s*(\d+[,\.]?\d*)',
            'duration': r'(\d+)\s*(year|semester|month|week|day)',
            'eligibility': r'(eligibility|requirement|qualification|criteria)',
            'contact': r'(\+?\d{10,12}|@|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)',
            'date': r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        }
    
    def extract_relevant_content(self, content: str, query: str, intent: str) -> str:
        """AI-powered content extraction using multiple techniques"""
        
        # Technique 1: Semantic similarity scoring
        semantic_content = self._semantic_extraction(content, query)
        
        # Technique 2: Pattern-based extraction
        pattern_content = self._pattern_based_extraction(content, query)
        
        # Technique 3: Section-based extraction
        section_content = self._section_based_extraction(content, query)
        
        # Combine and rank results
        combined = self._rank_and_combine(
            [semantic_content, pattern_content, section_content], 
            query
        )
        
        return combined if combined else "I don't have specific information about that."
    
    def _semantic_extraction(self, content: str, query: str) -> str:
        """Extract content based on semantic similarity"""
        sentences = self._split_into_sentences(content)
        
        if len(sentences) <= 1:
            return content
        
        try:
            # Calculate similarity between query and each sentence
            vectorizer = TfidfVectorizer().fit_transform([query] + sentences)
            vectors = vectorizer.toarray()
            query_vector = vectors[0]
            sentence_vectors = vectors[1:]
            
            similarities = cosine_similarity([query_vector], sentence_vectors)[0]
            
            # Get top 3 most relevant sentences
            top_indices = np.argsort(similarities)[-3:][::-1]
            relevant_sentences = [sentences[i] for i in top_indices if similarities[i] > 0.1]
            
            return ' '.join(relevant_sentences)
        except Exception as e:
            logger.warning(f"Semantic extraction failed: {e}")
            return ""
    
    def _pattern_based_extraction(self, content: str, query: str) -> str:
        """Extract content based on patterns in the query"""
        lines = content.split('\n')
        relevant_lines = []
        
        query_lower = query.lower()
        
        # Define extraction patterns based on query type
        extraction_rules = [
            # Fee queries
            (['fee', 'cost', 'price', 'tuition'], 
             lambda l: '₹' in l or any(word in l.lower() for word in ['fee', 'cost', 'price'])),
            
            # Duration queries  
            (['duration', 'how long', 'years', 'semester'],
             lambda l: any(word in l.lower() for word in ['year', 'semester', 'month', 'duration'])),
            
            # Eligibility queries
            (['eligibility', 'requirement', 'qualification'],
             lambda l: any(word in l.lower() for word in ['eligibility', 'requirement', 'qualification', 'criteria'])),
            
            # Contact queries
            (['contact', 'phone', 'email', 'address'],
             lambda l: any(word in l.lower() for word in ['contact', 'phone', 'email', 'address']) or 
                      re.search(r'\b\d{10}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', l)),
            
            # Hostel queries
            (['hostel', 'accommodation', 'room', 'stay'],
             lambda l: any(word in l.lower() for word in ['hostel', 'room', 'accommodation', 'mess'])),
            
            # Library queries
            (['library', 'book', 'study'],
             lambda l: any(word in l.lower() for word in ['library', 'book', 'study', 'reading'])),
            
            # Transport queries
            (['transport', 'bus', 'route', 'commute'],
             lambda l: any(word in l.lower() for word in ['bus', 'transport', 'route', 'schedule'])),
            
            # Placement queries
            (['placement', 'job', 'career', 'company'],
             lambda l: any(word in l.lower() for word in ['placement', 'job', 'career', 'company'])),
            
            # Scholarship queries
            (['scholarship', 'financial', 'aid'],
             lambda l: any(word in l.lower() for word in ['scholarship', 'financial', 'aid', 'grant']))
        ]
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            for keywords, condition in extraction_rules:
                if any(keyword in query_lower for keyword in keywords):
                    if condition(line_clean):
                        relevant_lines.append(line_clean)
                        break
        
        return '\n'.join(relevant_lines[:5])
    
    def _section_based_extraction(self, content: str, query: str) -> str:
        """Extract entire relevant sections"""
        sections = self._split_into_sections(content)
        query_keywords = set(query.lower().split())
        
        scored_sections = []
        for section in sections:
            score = self._calculate_section_relevance(section, query_keywords)
            if score > 0:
                scored_sections.append((score, section))
        
        # Return top 2 most relevant sections
        scored_sections.sort(reverse=True)
        return '\n\n'.join([section for _, section in scored_sections[:2]])
    
    def _split_into_sections(self, content: str) -> list:
        """Split content into logical sections"""
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                continue
            
            # Detect section headers
            if (line_clean.isupper() or 
                line_clean.startswith('#') or 
                (len(line_clean) < 100 and not line_clean.startswith('-'))):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line_clean]
            else:
                current_section.append(line_clean)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _calculate_section_relevance(self, section: str, query_keywords: set) -> float:
        """Calculate how relevant a section is to the query"""
        section_lower = section.lower()
        section_words = set(section_lower.split())
        
        # Keyword overlap
        keyword_overlap = len(query_keywords.intersection(section_words))
        
        # Presence of important markers
        important_markers = ['₹', 'important', 'required', 'must', 'deadline']
        marker_score = sum(1 for marker in important_markers if marker in section_lower)
        
        # Length penalty (prefer concise sections)
        length_penalty = min(len(section.split()) / 100, 1.0)
        
        return keyword_overlap + (marker_score * 0.5) - (length_penalty * 0.1)
    
    def _split_into_sentences(self, text: str) -> list:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _rank_and_combine(self, candidates: list, query: str) -> str:
        """Rank candidate extractions and combine the best ones"""
        scored_candidates = []
        
        for candidate in candidates:
            if not candidate.strip():
                continue
            
            # Score based on length (prefer medium length)
            length_score = 1.0 - abs(len(candidate.split()) - 50) / 100
            length_score = max(0.1, min(1.0, length_score))
            
            # Score based on query term presence
            query_terms = set(query.lower().split())
            candidate_terms = set(candidate.lower().split())
            term_score = len(query_terms.intersection(candidate_terms)) / len(query_terms) if query_terms else 0
            
            total_score = (term_score * 0.7) + (length_score * 0.3)
            scored_candidates.append((total_score, candidate))
        
        if not scored_candidates:
            return ""
        
        # Return the best candidate
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1]

class ExpertContentExtractor:
    def __init__(self):
        self.expert_rules = {
            'fee_queries': self._extract_fee_info,
            'admission_queries': self._extract_admission_info,
            'course_queries': self._extract_course_info,
            'facility_queries': self._extract_facility_info,
            'placement_queries': self._extract_placement_info,
            'scholarship_queries': self._extract_scholarship_info,
            'contact_queries': self._extract_contact_info,
            'general_queries': self._extract_general_info
        }
        
        self.query_classifier = {
            'fee': ['fee', 'cost', 'price', 'tuition', 'how much'],
            'admission': ['admission', 'admit', 'apply', 'application', 'eligibility'],
            'course': ['course', 'program', 'curriculum', 'syllabus', 'subject'],
            'facility': ['hostel', 'library', 'transport', 'bus', 'lab', 'sports', 'facility'],
            'placement': ['placement', 'job', 'career', 'company', 'recruitment'],
            'scholarship': ['scholarship', 'financial', 'aid', 'grant'],
            'contact': ['contact', 'phone', 'email', 'address', 'location']
        }
    
    def extract_content(self, content: str, query: str) -> str:
        """Expert system for content extraction"""
        query_type = self._classify_query(query)
        extractor = self.expert_rules.get(query_type, self.expert_rules['general_queries'])
        
        return extractor(content, query)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        for query_type, keywords in self.query_classifier.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        return 'general'
    
    def _extract_fee_info(self, content: str, query: str) -> str:
        """Smart fee extraction from structured fees file"""
        lines = content.split('\n')
        
        # Find which program user is asking about
        programs = ['mca', 'bca', 'bba', 'btech', 'engineering', 'bsc', 'ba', 'mba']
        target_program = None
        query_lower = query.lower()
        
        for program in programs:
            if program in query_lower:
                target_program = program
                break
        
        logger.info(f"🎯 Looking for fees for program: {target_program}")
        
        # Extract program-specific fees
        program_data = self._extract_program_data(lines, target_program)
        
        if program_data:
            return self._format_program_fees(program_data, target_program)
        else:
            return self._extract_general_fees(lines, query)
    
    def _extract_program_data(self, lines: list, target_program: str) -> dict:
        """Extract specific program data from structured file"""
        if not target_program:
            return None
        
        program_data = {}
        in_program_section = False
        current_program = None
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Check if this is a program section
            if line_clean.startswith('PROGRAM:'):
                current_program = line_clean.replace('PROGRAM:', '').strip().lower()
                in_program_section = (target_program in current_program)
                if in_program_section:
                    program_data['program'] = line_clean.replace('PROGRAM:', '').strip()
            
            # If we're in the target program section, collect data
            elif in_program_section and current_program and target_program in current_program:
                if line_clean.startswith('TUITION_FEE:'):
                    program_data['tuition'] = line_clean.replace('TUITION_FEE:', '').strip()
                elif line_clean.startswith('TOTAL_TUITION:'):
                    program_data['total'] = line_clean.replace('TOTAL_TUITION:', '').strip()
                elif line_clean.startswith('DURATION:'):
                    program_data['duration'] = line_clean.replace('DURATION:', '').strip()
                elif line_clean.startswith('ADDITIONAL_FEES:'):
                    program_data['additional'] = line_clean.replace('ADDITIONAL_FEES:', '').strip()
                elif line_clean.startswith('HOSTEL_OPTIONAL:'):
                    program_data['hostel'] = line_clean.replace('HOSTEL_OPTIONAL:', '').strip()
                
                # Stop if we hit next program section
                if line_clean.startswith('PROGRAM:') and target_program not in line_clean.lower():
                    break
        
        return program_data if len(program_data) > 1 else None
    
    def _format_program_fees(self, program_data: dict, program_name: str) -> str:
        """Format program-specific fee information"""
        parts = [f"💰 **{program_data['program']} Fee Structure:**"]
        
        if 'duration' in program_data:
            parts.append(f"📅 **Duration:** {program_data['duration']}")
        
        if 'tuition' in program_data:
            parts.append(f"🎓 **Tuition Fee:** {program_data['tuition']}")
        
        if 'total' in program_data:
            parts.append(f"💵 **Total Tuition:** {program_data['total']}")
        
        if 'additional' in program_data:
            parts.append(f"📋 **Additional Fees:** {program_data['additional']}")
        
        if 'hostel' in program_data:
            parts.append(f"🏠 **Hostel (Optional):** {program_data['hostel']}")
        
        parts.append("\n💳 **Payment Options Available**")
        parts.append("📚 **Scholarships: Up to 50% fee waiver**")
        
        return '\n'.join(parts)
    
    def _extract_general_fees(self, lines: list, query: str) -> str:
        """Extract general fee information when no specific program"""
        general_info = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Skip comments and section headers
            if line_clean.startswith('#') or line_clean.startswith('##'):
                continue
                
            # Include key fee information
            if any(keyword in line_clean.lower() for keyword in ['fee:', 'total:', '₹']):
                if len(general_info) < 6:
                    general_info.append(line_clean)
        
        if general_info:
            return "💰 **General Fee Information:**\n" + '\n'.join(general_info[:6])
        else:
            return "I don't have specific fee information. Please contact admissions office."
    
    def _extract_admission_info(self, content: str, query: str) -> str:
        """Expert admission information extraction"""
        lines = content.split('\n')
        admission_info = []
        
        # Find which program user is asking about
        programs = ['mca', 'bca', 'bba', 'btech', 'engineering', 'bsc', 'ba', 'mba']
        target_program = None
        query_lower = query.lower()
        
        for program in programs:
            if program in query_lower:
                target_program = program
                break
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Look for admission-related information
            if any(keyword in line_clean.lower() for keyword in [
                'eligibility', 'requirement', 'process', 'procedure', 
                'document', 'deadline', 'admission', 'apply'
            ]):
                if target_program:
                    if target_program in line_clean.lower():
                        admission_info.append(line_clean)
                else:
                    admission_info.append(line_clean)
                
                if len(admission_info) >= 5:
                    break
        
        if admission_info:
            return "🎓 **Admission Information:**\n" + '\n'.join(admission_info[:5])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_course_info(self, content: str, query: str) -> str:
        """Expert course information extraction"""
        lines = content.split('\n')
        course_info = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Extract course structure information
            if any(keyword in line_clean.lower() for keyword in [
                'duration', 'semester', 'year', 'credit', 
                'subject', 'curriculum', 'specialization', 'course', 'program'
            ]):
                course_info.append(line_clean)
                
                if len(course_info) >= 6:
                    break
        
        if course_info:
            return "📚 **Course Information:**\n" + '\n'.join(course_info[:6])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_facility_info(self, content: str, query: str) -> str:
        """Expert facility information extraction"""
        lines = content.split('\n')
        facility_info = []
        
        query_lower = query.lower()
        
        # Determine facility type
        facility_types = {
            'hostel': ['hostel', 'room', 'accommodation', 'stay', 'mess'],
            'library': ['library', 'book', 'study', 'research', 'reading'],
            'transport': ['transport', 'bus', 'route', 'schedule', 'commute'],
            'sports': ['sports', 'gym', 'ground', 'play', 'facility']
        }
        
        target_facility = None
        for facility, keywords in facility_types.items():
            if any(keyword in query_lower for keyword in keywords):
                target_facility = facility
                break
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            if target_facility:
                if any(keyword in line_clean.lower() for keyword in facility_types[target_facility]):
                    facility_info.append(line_clean)
            else:
                # General facility query
                if any(keyword in line_clean.lower() for keyword in ['facility', 'available', 'provide', 'include']):
                    facility_info.append(line_clean)
            
            if len(facility_info) >= 5:
                break
        
        if facility_info:
            facility_name = target_facility.upper() if target_facility else "Campus Facilities"
            return f"🏫 **{facility_name}:**\n" + '\n'.join(facility_info[:5])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_placement_info(self, content: str, query: str) -> str:
        """Expert placement information extraction"""
        lines = content.split('\n')
        placement_info = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Extract placement-related information
            if any(keyword in line_clean.lower() for keyword in [
                'placement', 'job', 'career', 'company', 'recruitment',
                'package', 'salary', 'opportunity', 'internship'
            ]):
                placement_info.append(line_clean)
                
                if len(placement_info) >= 4:
                    break
        
        if placement_info:
            return "💼 **Placement Information:**\n" + '\n'.join(placement_info[:4])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_scholarship_info(self, content: str, query: str) -> str:
        """Expert scholarship information extraction"""
        lines = content.split('\n')
        scholarship_info = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Extract scholarship-related information
            if any(keyword in line_clean.lower() for keyword in [
                'scholarship', 'financial', 'aid', 'grant', 'waiver',
                'merit', 'discount', 'concession'
            ]):
                scholarship_info.append(line_clean)
                
                if len(scholarship_info) >= 4:
                    break
        
        if scholarship_info:
            return "🎗️ **Scholarship Information:**\n" + '\n'.join(scholarship_info[:4])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_contact_info(self, content: str, query: str) -> str:
        """Expert contact information extraction"""
        lines = content.split('\n')
        contact_info = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Extract contact-related information
            if any(keyword in line_clean.lower() for keyword in [
                'contact', 'phone', 'email', 'address', 'number',
                'location', 'call', 'visit'
            ]) or re.search(r'\b\d{10}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line_clean):
                contact_info.append(line_clean)
                
                if len(contact_info) >= 3:
                    break
        
        if contact_info:
            return "📞 **Contact Information:**\n" + '\n'.join(contact_info[:3])
        else:
            return self._extract_general_info(content, query)
    
    def _extract_general_info(self, content: str, query: str) -> str:
        """General information extraction for any query type"""
        lines = content.split('\n')
        relevant_info = []
        query_keywords = set(query.lower().split())
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean or len(line_clean) < 10:
                continue
                
            line_lower = line_clean.lower()
            line_words = set(line_lower.split())
            
            # Check if line contains query keywords
            if query_keywords.intersection(line_words):
                relevant_info.append(line_clean)
            
            # Also include lines with important markers
            elif any(marker in line_lower for marker in ['important', 'required', 'must', 'note']):
                relevant_info.append(line_clean)
            
            if len(relevant_info) >= 4:
                break
        
        if relevant_info:
            return "ℹ️ **Information:**\n" + '\n'.join(relevant_info[:4])
        else:
            # Fallback: return first meaningful part of content
            for line in lines:
                if len(line.strip()) > 20:
                    return f"ℹ️ **Information:**\n{line.strip()}"
            
            return "I don't have specific information about that. Please contact the relevant department for more details."
    
    def _split_into_sections(self, content: str) -> list:
        """Split content into sections"""
        return [section.strip() for section in content.split('\n\n') if section.strip()]

class HybridContentExtractor:
    def __init__(self):
        self.semantic_extractor = SmartContentExtractor()
        self.expert_extractor = ExpertContentExtractor()
    
    def extract_content(self, content: str, query: str, intent: str) -> str:
        """Hybrid extraction using both AI and rules"""
        
        # SPECIAL HANDLING: For specific intents, prefer expert system
        if intent in ['fees_structure', 'admission_requirements', 'course_information']:
            logger.info(f"🎯 Using expert extractor for {intent}")
            return self.expert_extractor.extract_content(content, query)
        
        # For other intents, use hybrid approach
        semantic_result = self.semantic_extractor.extract_relevant_content(content, query, intent)
        
        if len(semantic_result.split()) < 10 or self._is_low_quality(semantic_result, query):
            expert_result = self.expert_extractor.extract_content(content, query)
            
            if self._score_content(expert_result, query) > self._score_content(semantic_result, query):
                return expert_result
        
        return semantic_result
    
    def _is_low_quality(self, content: str, query: str) -> bool:
        """Check if extracted content is low quality"""
        if not content or len(content.strip()) < 20:
            return True
        
        # Check if content contains any query terms
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms.intersection(content_terms):
            return True
        
        return False
    
    def _score_content(self, content: str, query: str) -> float:
        """Score content quality"""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer 30-150 words)
        word_count = len(content.split())
        if word_count > 150:
            length_score = 0.3
        else:
            length_score = 1.0 - (word_count / 200)
        score += max(0.1, length_score) * 0.4
        
        # Relevance score
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        relevance_score = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        score += relevance_score * 0.4
        
        # Information density score
        info_density = len([c for c in content if c.isdigit() or c in '₹$']) / len(content)
        score += info_density * 0.2
        
        return score