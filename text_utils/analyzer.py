# Topic extraction logic
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from text_utils.default_themes import get_default_themes
import json
import time
import torch
import textwrap
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

        print("\t*** Model loaded successfully")
    

    def extract_themes(self, transcript) -> dict:
        """
        Extract key themes from interview transcript and append
        default themes. 

        Args:
            transcript (str)
        
        Returns:
            dict: A dictionary with the following fields:
            - "parsed_themes" (str): Themes in an easy to read format.
            - "raw_response" (str): Themes in JSON format.  
            - "inference_time" (float)
        """
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
        inference_time = round(end_time - start_time, 2)

        print(f"\t*** Inference completed in {inference_time}s")
        
        # Extract generated text
        raw_response = result[0]['generated_text']
        
        # Parse themes
        parsed_themes = self._parse_themes(raw_response)

        # Format JSON string and add theme ID and default themes
        try:
            json_themes = json.loads(raw_response)
            for idx, theme in enumerate(json_themes):
                theme["theme_id"] = idx
            # Append default themes (such as "interviewer", "none", etc.)
            json_themes.extend(get_default_themes())
        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse raw_response: {e}")
            json_themes = [] 
                
        return {
            "parsed_themes": parsed_themes,
            "json_themes": json_themes,
            "inference_time": inference_time
        }
    

    def classify_segment(self, segment_text: str, themes: list[dict]) -> dict:
        """
        Classify a single transcript segment into one of the provided themes 
        and generate a short summary of its content.

        This method uses the loaded LLaMA model to evaluate the segment's 
        content and return both a classification (based on the provided themes) 
        and a 1-2 sentence summary suitable for annotation or labeling.

        Args:
            segment_text (str): The raw text of a single segment (e.g., from one speaker block).
            themes (list of dict): A list of themes previously extracted from the full transcript.
                Each theme dict should include at least a "title" and "description".

        Returns:
            dict: A dictionary with the following fields:
                - "theme_title" (str): The best-matching theme for this segment.
                - "summary" (str): A short description or summary of what is said in the segment.
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

        # Parse JSON response 
        response_text = result[0]["generated_text"]
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON:\n{response_text}")
            return {"theme_title": "Uncategorized", "summary": "Could not parse model response."}
        


    def classify_all_segments(self, segments: list[dict], themes: list[dict], batch_size=8) -> list[dict]:
        """
        Classify and summarize a list of segments based on provided themes. 

        Args:
            segments (list of dict): Segments to classify. Each must include a "text" field.
            themes (list of dict): List of themes extracted from the full transcript. 
            batch_size (int): Number of segments to classify in each batch. 
        
        Returns:
            list of dict: The same segments, with added fields for "theme_title" and "summary".
        """
        print(f"\t*** Classifying {len(segments)} segments in batches of {batch_size}")

        # Initialize stats dict and failed segments list for tracking
        stats = {
            "total": len(segments),
            "successful": 0,
            "retried": 0,
            "failed": 0,
            "batch_failures": 0
        }

        failed_segments = []

        # Define function to make prompt for each segment
        def make_prompt(text):
            return self._build_prompt("classify_segment", {
                "themes": themes,
                "segment_text": text
            })
        
        # Build prompts for all non empty segments
        segments_to_classify = []
        prompts = []

        for segment in segments:
            # If segment is empty, give it "None" theme
            if self._is_empty_segment(segment["text"]):
                segment["theme_title"] = "None",
                segment["summary"] = "No relevant speech content.",
                segment["theme_id"] = -1
                continue

            # Add prompts for non empty segments
            prompt = make_prompt(segment["text"])
            prompts.append(prompt)
            segments_to_classify.append(segment)

        start_time = time.time()

        # Process themes in batches
        for i in range(0, len(segments_to_classify), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_segments = segments_to_classify[i:i + batch_size]

            batch_start = time.time()
            try:
                results = self.generator(
                    batch_prompts,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    return_full_text=False
                )
            except Exception as e:
                print(f"[ERROR] Failed to run batch {i // batch_size}: {e}")
                stats["batch_failures"] += 1
                failed_segments.extend(batch_segments)
                continue

            # Loop through batch results
            for j, result_list in enumerate(results):
                segment = batch_segments[j]
                result = result_list[0]
                raw = result["generated_text"]

                try:
                    parsed = json.loads(raw)
                    matched_title = parsed.get("theme_title", "Uncategorized")
                    summary = parsed.get("summary", "")
                    stats["successful"] += 1
                except Exception:
                    print(f"\t*** [!] Parsing failed for segment {i+j}, retrying individually")
                    stats["retried"] += 1

                    # If parsing fails, call classify_segment() for that segment
                    retry_result = self.classify_segment(segment["text"], themes)
                    matched_title = retry_result.get("theme_title", "Uncategorized")
                    summary = retry_result.get("summary", "[Retry failed]")

                    if matched_title == "Uncategorized":
                        stats["failed"] += 1
                        failed_segments.append(segment)
                
                # Get theme_id
                matched_theme = next((t for t in themes if t["title"] == matched_title), None)
                theme_id = matched_theme["theme_id"] if matched_theme else -1 # -1 means unmatched

                # Update segment
                segment["theme_title"] = matched_title
                segment["summary"] = summary
                segment["theme_id"] = theme_id

                print(f"\t*** [{i + j + 1}/{len(segments)}] Theme: {matched_title} (ID: {theme_id})")
            
            batch_duration = round(time.time() - batch_start, 2)
            print(f"\t*** Batch {i // batch_size + 1} processed in {batch_duration}s")

        total_duration = round(time.time() - start_time, 2)
        print(f"\t*** All segments classified in {total_duration}s")

        # Print stats summary
        print("\t*** Classification Summary:")
        print(f"  Total segments       : {stats['total']}")
        print(f"  Parsed successfully  : {stats['successful']}")
        print(f"  Retried              : {stats['retried']}")
        print(f"  Failed after retry   : {stats['failed']}")
        print(f"  Batch failures       : {stats['batch_failures']}")
        print(f"  Total time           : {total_duration}s")

        self.last_classification_stats = stats
        self.failed_segments = failed_segments

        return segments


    def _parse_themes(self, response_text: str):
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
        

    def _is_empty_segment(self, text:str) -> bool:
        stripped = text.strip().lower()
        # if stripped text is empty or only punctuation, return true
        return not stripped or re.fullmatch(r"[.,!?()\[\]\"'\s]+", stripped)
    

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
            return textwrap.dedent(f"""\
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an expert at analyzing interview content. Your task is to classify this segment of text into one of the provided themes, and then summarize the segment.
                <|eot_id|><|start_header_id|>user<|end_header_id|>

                Here are the themes:

                {json.dumps(inputs["themes"], indent=2)}

                Classify the following segment into one of the provided themes based on its content and speaker.

                - If the segment appears to be spoken by the interviewer, classify it as "Interviewer", even if it includes introductory or transitional phrasing. For example:
                {{
                    "theme_title": "Interviewer",
                    "summary": "The interviewer asks the participant to introduce themselves."
                }}
                - Only classify a segment as "Introduction" if the **interviewee** is introducing themselves.
                - Do not modify the theme titles.

                Segment:
                {inputs["segment_text"]}

                Respond only with a single JSON object, and do not include any explanation or introduction.

                If the segment contains no meaningful speech, only background noise, or is unintelligible or silent, classify it as:
                {{
                "theme_title": "None",
                "summary": "No relevant speech content"
                }}

                Return your response in this exact format:
                {{
                "theme_title": "Career Background",
                "summary": "Discussion about professional experience and work history"
                }}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """)

        else:
            raise ValueError(f"Unknown prompt task: {task}")
            

