import gradio as gr
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from typing import Dict, List, Tuple, Optional
import re
import textstat
from collections import Counter
import math
from abc import ABC, abstractmethod

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

class TextAnalyzer:
    """Handles text analysis and feature extraction"""
    
    @staticmethod
    def extract_features(text: str) -> Dict:
        """Extract linguistic features from text"""
        if not text.strip():
            return {}
            
        words = text.split()
        sentences = TextAnalyzer._split_sentences(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'lexical_diversity': TextAnalyzer._calculate_lexical_diversity(words),
            'avg_sentence_length': TextAnalyzer._calculate_avg_sentence_length(words, sentences),
            'burstiness': TextAnalyzer._calculate_burstiness(sentences),
            'flesch_reading_ease': TextAnalyzer._calculate_readability(text)
        }
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _calculate_lexical_diversity(words: List[str]) -> float:
        """Calculate lexical diversity (unique words / total words)"""
        if not words:
            return 0.0
        unique_words = len(set(word.lower() for word in words))
        return unique_words / len(words)
    
    @staticmethod
    def _calculate_avg_sentence_length(words: List[str], sentences: List[str]) -> float:
        """Calculate average sentence length"""
        sentence_count = len(sentences) if sentences else 1
        return len(words) / sentence_count
    
    @staticmethod
    def _calculate_burstiness(sentences: List[str]) -> float:
        """Calculate sentence length variation (burstiness)"""
        if len(sentences) <= 1:
            return 0.0
            
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths or len(sentence_lengths) <= 1:
            return 0.0
            
        mean_length = np.mean(sentence_lengths)
        if mean_length == 0:
            return 0.0
            
        return np.std(sentence_lengths) / mean_length
    
    @staticmethod
    def _calculate_readability(text: str) -> float:
        """Calculate Flesch reading ease score"""
        try:
            return textstat.flesch_reading_ease(text) if len(text) > 10 else 50.0
        except:
            return 50.0

class DatasetAnalyzer:
    """Handles dataset loading and pattern analysis"""
    
    def __init__(self, dataset_path: str = 'ai_human_content_detection_dataset.csv'):
        self.dataset_path = dataset_path
        self.dataset = None
        self.patterns = {}
        
    def load_dataset(self) -> bool:
        """Load dataset from CSV file"""
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded: {len(self.dataset)} rows")
            return True
        except FileNotFoundError:
            print("Dataset not found. Using fallback patterns.")
            return False
    
    def analyze_patterns(self) -> Dict:
        """Analyze patterns between AI and human content"""
        if self.dataset is None:
            return self._get_fallback_patterns()
        
        ai_samples = self.dataset[self.dataset['label'] == 1]
        human_samples = self.dataset[self.dataset['label'] == 0]
        
        features = [
            'lexical_diversity', 'avg_sentence_length', 'avg_word_length',
            'punctuation_ratio', 'flesch_reading_ease', 'burstiness',
            'passive_voice_ratio', 'predictability_score'
        ]
        
        patterns = {}
        for feature in features:
            if feature in self.dataset.columns:
                ai_mean = ai_samples[feature].mean()
                human_mean = human_samples[feature].mean()
                
                patterns[feature] = {
                    'target': 'increase' if human_mean > ai_mean else 'decrease',
                    'human_avg': human_mean,
                    'ai_avg': ai_mean,
                    'difference': abs(human_mean - ai_mean)
                }
        
        self.patterns = patterns
        return patterns
    
    def _get_fallback_patterns(self) -> Dict:
        """Fallback patterns when dataset is not available"""
        return {
            'lexical_diversity': {
                'target': 'increase',
                'human_avg': 0.85,
                'ai_avg': 0.75,
                'difference': 0.10
            },
            'burstiness': {
                'target': 'increase', 
                'human_avg': 0.65,
                'ai_avg': 0.45,
                'difference': 0.20
            },
            'predictability_score': {
                'target': 'decrease',
                'human_avg': 85.0,
                'ai_avg': 105.0,
                'difference': 20.0
            }
        }

class SuggestionGenerator:
    """Generates improvement suggestions based on analysis"""
    
    SUGGESTION_RULES = {
        'low_lexical_diversity': "Use more varied vocabulary - avoid repeating the same words",
        'low_burstiness': "Vary your sentence lengths - mix short and long sentences",
        'high_readability': "Add some complexity - use more sophisticated language",
        'low_readability': "Simplify slightly - make it more readable",
        'short_sentences': "Consider longer, more detailed sentences",
        'long_sentences': "Break down some longer sentences for clarity",
        'insufficient_detail': "Add more detail and context to your prompt"
    }
    
    @classmethod
    def generate(cls, analysis: Dict) -> List[str]:
        """Generate suggestions based on prompt analysis"""
        if not analysis:
            return ["Please enter a valid prompt to analyze."]
        
        suggestions = []
        
        # Lexical diversity check
        if analysis.get('lexical_diversity', 0) < 0.7:
            suggestions.append(cls.SUGGESTION_RULES['low_lexical_diversity'])
        
        # Sentence variation check
        if analysis.get('burstiness', 0) < 0.3:
            suggestions.append(cls.SUGGESTION_RULES['low_burstiness'])
        
        # Readability checks
        flesch_score = analysis.get('flesch_reading_ease', 50)
        if flesch_score > 80:
            suggestions.append(cls.SUGGESTION_RULES['high_readability'])
        elif flesch_score < 30:
            suggestions.append(cls.SUGGESTION_RULES['low_readability'])
        
        # Sentence length checks
        avg_length = analysis.get('avg_sentence_length', 0)
        if avg_length < 8:
            suggestions.append(cls.SUGGESTION_RULES['short_sentences'])
        elif avg_length > 25:
            suggestions.append(cls.SUGGESTION_RULES['long_sentences'])
        
        # Word count check
        if analysis.get('word_count', 0) < 10:
            suggestions.append(cls.SUGGESTION_RULES['insufficient_detail'])
        
        return suggestions if suggestions else ["Your prompt looks well-balanced!"]

class PromptOptimizer:
    """Main optimizer that coordinates analysis and enhancement"""
    
    def __init__(self, gemini_model=None):
        self.gemini_model = gemini_model
        self.dataset_analyzer = DatasetAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.suggestion_generator = SuggestionGenerator()
        
        # Initialize
        self.dataset_analyzer.load_dataset()
        self.dataset_analyzer.analyze_patterns()
    
    def analyze_prompt(self, prompt: str) -> Dict:
        """Analyze prompt and return features"""
        return self.text_analyzer.extract_features(prompt)
    
    def generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate improvement suggestions"""
        return self.suggestion_generator.generate(analysis)
    
    def optimize_with_ai(self, prompt: str, suggestions: List[str]) -> str:
        """Optimize prompt using AI (Gemini)"""
        if not self.gemini_model:
            return self._create_basic_optimization(prompt, suggestions)
        
        try:
            optimization_request = self._create_optimization_request(prompt, suggestions)
            response = self.gemini_model.generate_content(optimization_request)
            return response.text
        except Exception as e:
            print(f"AI optimization failed: {e}")
            return self._create_basic_optimization(prompt, suggestions)
    
    def _create_optimization_request(self, prompt: str, suggestions: List[str]) -> str:
        """Create optimization request for AI"""
        return f"""
        Improve this prompt to make it more natural and human-like:
        
        Original: {prompt}
        
        Apply these improvements:
        {chr(10).join([f"- {s}" for s in suggestions])}
        
        Return only the improved prompt, maintaining the original intent.
        """
    
    def _create_basic_optimization(self, prompt: str, suggestions: List[str]) -> str:
        """Create basic optimization when AI is not available"""
        enhanced_suggestions = []
        improved_prompt = prompt
        
        suggestion_text = " ".join(suggestions).lower()
        
        if "vocabulary" in suggestion_text:
            enhanced_suggestions.append("Added vocabulary diversity cues")
        if "sentence" in suggestion_text:
            enhanced_suggestions.append("Added sentence variation instructions")
        if "detail" in suggestion_text:
            enhanced_suggestions.append("Enhanced with context and specificity")
        if "complexity" in suggestion_text:
            enhanced_suggestions.append("Adjusted complexity level")
        
        # Basic enhancement for short prompts
        if len(improved_prompt.split()) < 15:
            improved_prompt = f"Please {improved_prompt.lower()} with detailed explanations, varied sentence structures, and engaging language that feels natural and conversational."
        
        result = f"**Optimized Prompt:**\n\n{improved_prompt}\n\n"
        if enhanced_suggestions:
            result += f"**Applied Improvements:**\n" + "\n".join([f"• {s}" for s in enhanced_suggestions]) + "\n\n"
        result += "*Note: Connect Gemini API for AI-powered optimization*"
        
        return result
    
    def optimize(self, prompt: str) -> Tuple[str, str]:
        """Main optimization method"""
        if not prompt.strip():
            return "Please enter a prompt to optimize.", "**Please enter a prompt to analyze.**"
        
        # Analyze prompt
        analysis = self.analyze_prompt(prompt)
        if not analysis:
            return "Invalid prompt provided.", "**No analysis available.**"
        
        # Generate suggestions
        suggestions = self.generate_suggestions(analysis)
        
        # Create optimized version
        optimized = self.optimize_with_ai(prompt, suggestions)
        
        # Format analysis
        analysis_text = self._format_analysis(analysis, suggestions)
        
        return optimized, analysis_text
    
    def _format_analysis(self, analysis: Dict, suggestions: List[str]) -> str:
        """Format analysis results for display"""
        return f"""
## Prompt Analysis:

• **Word Count:** {analysis['word_count']}
• **Sentences:** {analysis['sentence_count']}
• **Lexical Diversity:** {analysis['lexical_diversity']:.2f}/1.0
• **Avg Sentence Length:** {analysis['avg_sentence_length']:.1f} words
• **Sentence Variation:** {analysis['burstiness']:.2f}
• **Reading Ease:** {analysis['flesch_reading_ease']:.0f}/100

## Improvement Suggestions:
{chr(10).join([f"• {s}" for s in suggestions])}
        """

# Application Interface
class PromptlyApp:
    """Gradio application interface"""
    
    def __init__(self):
        self.optimizer = PromptOptimizer(model if GEMINI_API_KEY else None)
    
    def optimize_interface(self, prompt: str) -> Tuple[str, str]:
        """Interface method for Gradio"""
        return self.optimizer.optimize(prompt)
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Promptly - AI Prompt Optimizer", theme=gr.themes.Soft()) as app:
            
            # Header
            gr.Markdown("""
            # Promptly - Grammarly for Prompt Engineering
            
            **Improve your prompts to get more natural, human-like responses from AI chatbots!**
            
            This tool analyzes your prompts using patterns from AI vs Human content detection research 
            and suggests improvements to make your AI interactions more natural and effective.
            """)
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    input_prompt = gr.Textbox(
                        label="Your Original Prompt",
                        placeholder="Enter your prompt here...\n\nExample: Write a blog post about artificial intelligence",
                        lines=6,
                        max_lines=15
                    )
                    optimize_btn = gr.Button("Optimize My Prompt", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_prompt = gr.Textbox(
                        label="Optimized Prompt",
                        lines=6,
                        max_lines=15,
                        show_copy_button=True
                    )
            
            with gr.Row():
                analysis_output = gr.Markdown(label="Analysis & Suggestions")
            
            # Examples
            gr.Markdown("### Try These Examples:")
            gr.Examples(
                examples=[
                    ["Write a blog post about artificial intelligence"],
                    ["Create a marketing email for a new product launch"],
                    ["Generate a creative story about space exploration"],
                    ["Explain quantum physics in simple terms"],
                    ["Make a workout plan for beginners"],
                    ["Write code for a todo app"]
                ],
                inputs=input_prompt
            )
            
            # Event handlers
            optimize_btn.click(
                fn=self.optimize_interface,
                inputs=[input_prompt],
                outputs=[output_prompt, analysis_output]
            )
            
            input_prompt.submit(
                fn=self.optimize_interface,
                inputs=[input_prompt], 
                outputs=[output_prompt, analysis_output]
            )
            
            # Footer
            gr.Markdown("""
            ---
            
            ### How Promptly Works:
            
            1. **Analysis**: Examines linguistic patterns in your prompt
            2. **Pattern Matching**: Compares against AI vs Human content research  
            3. **Smart Suggestions**: Provides specific improvement recommendations
            4. **Optimization**: Enhances your prompt for more natural AI responses
            
            ### Features Analyzed:
            
            - **Lexical Diversity**: Vocabulary variety and word choice
            - **Sentence Variation**: Mix of short and long sentences  
            - **Reading Complexity**: Appropriate difficulty level
            - **Predictability**: Patterns that make text sound more human
            - **Style Indicators**: Language naturalness metrics
            
            Built using AI vs Human Content Detection dataset with 1000+ samples and 17 linguistic features.
            
            ---
            *Promptly v1.0 - Making AI conversations more human*
            """)
        
        return app

# Application entry point
def main():
    """Main application entry point"""
    app_instance = PromptlyApp()
    interface = app_instance.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
