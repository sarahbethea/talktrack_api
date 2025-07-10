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
        prompt = self._build_prompt("extract_themes", {"transcript": transcript})
        
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
            themes (list): list of themes.
        
        Returns:
            ???
        """

        prompt = self._build_prompt("classify_segment", {
            "themes": themes, 
            "segment_text": segment_text
        })

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
        

    def _build_prompt(self, task: str, inputs: dict) -> str:
        """
        Build a prompt for a given task using input values.
        
        Args:
            task (str): Either "extract_themes" or "classify_segment"
            inputs (dict): Data required for the task (e.g. transcript, themes, segment_text)
        
        Returns:
            str: A fully formatted prompt string
        """
        if task == "extract_themes":
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
            {inputs["transcript"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """

        elif task == "classify_segment":
            return f"""
            Here are the themes:

            {json.dumps(inputs["themes"], indent=2)}

            Classify the following segment into one of these themes and summarize it in 1-2 sentences.

            Segment:
            {inputs["segment_text"]}

            Respond only with:
            {{
                "theme_title": "...",
                "summary": "..."
            }}
            """.strip()

        else:
            raise ValueError(f"Unknown prompt task: {task}")
            

