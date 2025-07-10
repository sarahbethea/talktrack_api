# Topic extraction logic
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import json
import time
import torch
import os
import re #regex module

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")



class TopicAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the topic analyzer with a local Llama model.
        
        Args:
            model_name (str): Hugging Face model name
        """
        print(f"\t*** Loading topic analysis model: {model_name} ***")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            use_auth_token=token
        )

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        print("\t*** Model loaded successfully ***")
    

    def extract_themes(self, transcript):
        """Extract key themes from interview transcript."""
        prompt = self._themes_build_prompt(transcript)
        
        # Generate with pipeline 
        start_time = time.time()
        result = self.generator(
            prompt,
            max_new_tokens=600,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False  # Only return the generated part
        )
        end_time = time.time()
        inference_time = end_time - start_time

        print(f"\t*** Inference completed in {inference_time}s ***")
        
        # Extract generated text
        response_text = result[0]['generated_text']
        
        # Parse themes
        return {
            "parsed_themes": self._parse_themes(response_text),
            "raw_response": response_text,
            "inference_time": inference_time
        }
    
    def classify_segment(self, segment_text, themes):
        """
        Classify single segment into one of provided themes and return a short summary

        Args:
            segment_text (str): segment text.
            themes (???): list of themes.
        
        Returns:
            ???
        """

        prompt = self._build_classification_prompt(segment_text, themes)

        result = self.generator(
            prompt,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            return_full_text=False
        )

        # Parse JSON response (can add error handling)
        response_text = result[0]["generated_text"]
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON:\n{response_text}")
            return {"theme_title": "Uncategorized", "summary": "Could not parse model response."}


    

    def _themes_build_prompt(self, transcript):
        # Same prompt as before
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are an expert at analyzing interview content. Your task is to identify the major themes discussed in interview transcripts.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Analyze this interview transcript and identify 3-5 major themes/topics discussed.

        For each theme, provide:
        1. A brief title (2-4 words)
        2. A short description (1-2 sentences)  
        3. 3-5 key phrases that indicate this theme

        Respond only with the JSON array, and do not include any explanation or introduction.
        Return your response as a JSON array in this exact format:
        [
        {{
            "title": "Career Background",
            "description": "Discussion about professional experience and work history",
            "keywords": ["work", "job", "experience", "career", "company"]
        }},
        {{
            "title": "Technical Skills", 
            "description": "Conversation about programming languages and technical abilities",
            "keywords": ["programming", "code", "development", "technical", "software"]
        }}
        ]

        Transcript:
        {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
    

    def _build_classification_prompt(self, text, themes):
        return f"""
        Here are the interview themes:

        {json.dumps(themes, indent=2)}

        Classify the following interview segment into the most relevant theme, and summarize it in 1–2 sentences.

        Segment:
        {text}

        Respond **only** in this JSON format:
        {{
        "theme_title": "...",
        "summary": "..."
        }}
        """.strip()
    
    
    def _parse_themes(self, response_text):
        """Parse themes from model response."""
        try:
            # Look for JSON array in the response
            json_match = re.search(r'\[\s*{.*?}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                themes = json.loads(json_str)
                return themes
            else:
                print("No JSON found in response")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            print(f"Unexpected error parsing themes: {e}")
            return []
        

