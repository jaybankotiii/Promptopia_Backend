import time

import pandas as pd
print("✅ Pandas imported successfully!")
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tiktoken
import re
import requests
import anthropic
import google.generativeai as genai
import sys
import json
import os
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['GRPC_VERBOSITY'] = 'ERROR'    # Suppress GRPC warnings
os.environ['GRPC_CPP_ENABLE_LOGGING'] = '0'


# Constants for API keys

# Access the API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,  # This is the default and can be omitted
)

# Initialize Hugging Face model for semantic similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define available models
models = [
    {"name": "gpt-4", "engine": "gpt-4", "provider": "openai"},
    {"name": "gpt-4-turbo", "engine": "gpt-4-turbo", "provider": "openai"},
    {"name": "gpt-3.5-turbo", "engine": "gpt-3.5-turbo", "provider": "openai"},
    {"name": "gpt-3.5-turbo-16k", "engine": "gpt-3.5-turbo-16k", "provider": "openai"},
    {"name": "claude-2.1", "engine": "claude-2.1", "provider": "anthropic"},
    {"name": "claude-2.0", "engine": "claude-2.0", "provider": "anthropic"},
    {"name": "gemini-1.5-flash", "engine": "gemini-1.5-flash", "provider": "google"}
]

# Utility function: Sanitize sheet names for Excel/PDF
def sanitize_sheet_name(prompt):
    sheet_name = re.sub(r"[\[\]:*?/\\]", "", prompt)
    return sheet_name[:30]

def normalize_model_name(model_name):
    return model_name.strip().lower().replace(" ", "-")


# Count tokens using tiktoken
def count_tokens(text, model="gpt-4"):
    try:
        if model in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Token counting failed for model {model}: {str(e)}")
        return "N/A"

# Measure latency
def measure_latency(model_call):
    start_time = time.time()
    response = model_call()
    latency = time.time() - start_time
    return response, latency * 1000  # Convert to milliseconds


# Generic function to fetch model responses
def get_model_response_by_provider(prompt, model_name, provider, max_retries=10, delay=2):
    """
    Fetch the model response with retry logic for handling API errors.

    Args:
        prompt (str): The input prompt.
        model_name (str): The model to use.
        provider (str): The provider ("openai", "anthropic", or "google").
        max_retries (int): Number of retry attempts on failure.
        delay (int): Delay between retries in seconds.

    Returns:
        str: The model's response.
    """

    retries = 0

    while retries <= max_retries:
        try:
            if provider == "openai":
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()

            elif provider == "anthropic":
                client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
                response = client.completions.create(
                    model=model_name,
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=10000
                )
                return response.completion.strip()

            elif provider == "google":
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if hasattr(response, "candidates") and response.candidates:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    raise ValueError("No candidates returned from Google Gemini.")

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"⚠️ Error with {provider} API: {str(e)}. Retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(delay * retries)  # Exponential backoff

    raise ValueError(f"❌ {provider} API failed after {max_retries} retries.")

# Semantic similarity using Sentence-BERT
def calculate_semantic_similarity(standard, response):
    embeddings = similarity_model.encode([standard, response], convert_to_tensor=True)
    from torch.nn.functional import cosine_similarity
    similarity = cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    return max(0, min(similarity, 1))  # Clipping the range


# Token efficiency and economy gain
def evaluate_token_economy(prompt, response, model_name):
    tokens_prompt = count_tokens(prompt, model_name)
    tokens_response = count_tokens(response, model_name)
    if tokens_prompt == "N/A" or tokens_response == "N/A":
        return "N/A"
    return tokens_response / tokens_prompt  # Token efficiency ratio

def calculate_token_economy_gain(base_efficiency, model_efficiency):
    if base_efficiency == "N/A" or model_efficiency == "N/A":
        return "N/A"
    return ((base_efficiency - model_efficiency) / base_efficiency) * 100

# Accuracy and scoring logic
def calculate_unified_accuracy(gold_standard, response):
    semantic_similarity = calculate_semantic_similarity(gold_standard, response) * 100
    task_accuracy = task_specific_validation(response, gold_standard)
    coherence_score = calculate_coherence_score(response)
    unified_accuracy = (0.5 * semantic_similarity + 0.3 * task_accuracy + 0.2 * coherence_score)
    return min(100, unified_accuracy)  # Clipping the range


# Task-Specific Validation Using GPT API
def task_specific_validation(response, gold_standard):
    try:
        system_prompt = (
            "You are an expert AI evaluator tasked with scoring the correctness of a model's response. "
            "The response should be evaluated against the provided gold standard. "
            "Score the response for correctness, factual alignment, and adherence to the gold standard on a scale from 1 to 100. "
            "Consider the following strictly:\n"
            "1. If the response directly contradicts or misrepresents the gold standard, score it below 50.\n"
            "2. If the response is incomplete but partially aligns with the gold standard, score it between 50 and 70.\n"
            "3. If the response is accurate but lacks depth, score it between 70 and 90.\n"
            "4. If the response is completely accurate, well-aligned, and comprehensive, score it between 90 and 100.\n"
            "Your evaluation should reflect these strict guidelines.\n"
            "Score the response strictly on a scale from 1 to 100.\n"
            "Do not provide explanations or text. Respond with a single numeric score only."
        )
        user_prompt = f"Gold Standard: {gold_standard}\nResponse: {response}\n\nWhat is the correctness score?"
        completion = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="gpt-4-turbo",
            max_tokens=10
        )
        # Extract response content using JSON-like syntax
        return float(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in task-specific validation: {e}")
        return 0

# Coherence Scoring Using GPT API
def calculate_coherence_score(response):
    try:
        system_prompt = (
            "You are an expert AI evaluator tasked with scoring the coherence, fluency, and relevance of a response. "
            "Score the response on a scale from 1 to 100. Use the following strict criteria:\n"
            "1. If the response is incoherent, off-topic, or irrelevant, score it below 50.\n"
            "2. If the response is grammatically flawed but still somewhat logical, score it between 50 and 70.\n"
            "3. If the response is logical, relevant, and mostly fluent, score it between 70 and 90.\n"
            "4. If the response is perfectly coherent, highly fluent, and completely relevant, score it between 90 and 100.\n"
            "Ensure your evaluation strictly adheres to these rules.\n"
            "Score the response strictly on a scale from 1 to 100.\n"
            "Do not provide explanations or text. Respond with a single numeric score only."
        )
        user_prompt = f"Response: {response}\n\nWhat is the coherence, fluency, and relevance score?"
        completion = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="gpt-4-turbo",
            max_tokens=10
        )
        # Extract response content using JSON-like syntax
        return float(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error in coherence scoring: {e}")
        return 0



# Weighted score calculation
def calculate_weighted_score(metrics, baseline_metrics):
    accuracy_gain = metrics["Accuracy"] - baseline_metrics["Accuracy"]
    latency_gain = (baseline_metrics["Latency (ms)"] - metrics["Latency (ms)"]) / baseline_metrics["Latency (ms)"] * 100
    token_economy_gain = metrics["Token Economy Gain (%)"]
    return (0.2 * accuracy_gain + 0.4 * latency_gain + 0.4 * token_economy_gain)

# Main evaluation function
def evaluate_from_spreadsheet(json_data):
    if not isinstance(json_data, list):
        raise ValueError("Input must be a list of prompts with their models.")

    for item in json_data:
        if 'prompt' not in item or 'goldenModel' not in item:
            raise ValueError("Each item must contain 'prompt' and 'goldenModel' fields.")

    # ✅ Convert JSON to pandas DataFrame with 'prompt' and 'model' columns
    data = pd.DataFrame(json_data)
    data.rename(columns={'goldenModel': 'model'}, inplace=True)  # Rename 'goldenModel' to 'model'
    data.columns = data.columns.str.strip().str.lower()


    results_by_prompt = {}

    for _, row in data.iterrows():
        prompt = row['prompt']
        gold_model = normalize_model_name(row['model'])
        gold_model_data = next((model for model in models if model['name'] == gold_model), None)
        if not gold_model_data:
            raise ValueError(f"Gold model '{gold_model}' not found in available models.")

        provider = gold_model_data["provider"]

        # Gold Standard Evaluation
        gold_response, gold_latency = measure_latency(lambda: get_model_response_by_provider(prompt, gold_model, provider))
        gold_efficiency = evaluate_token_economy(prompt, gold_response, gold_model)

        gold_metrics = {
            "Model": gold_model,
            "Latency (ms)": gold_latency,
            "Token Efficiency Ratio": gold_efficiency,
            "Accuracy": 100.0,
            "Response": gold_response
        }

        comparisons = []
        for model in models:
            if model["name"] != gold_model:
                try:
                    response, latency = measure_latency(lambda: get_model_response_by_provider(prompt, model["engine"], model["provider"]))
                    model_efficiency = evaluate_token_economy(prompt, response, model["engine"])
                    token_economy_gain = calculate_token_economy_gain(gold_efficiency, model_efficiency)
                    accuracy = calculate_unified_accuracy(gold_response, response)
                    semantic_similarity = calculate_semantic_similarity(gold_response, response) * 100
                    coherence = calculate_coherence_score(response)

                    metrics = {
                        "Latency (ms)": latency,
                        "Token Economy Gain (%)": calculate_token_economy_gain(gold_efficiency, evaluate_token_economy(prompt, response, model["engine"])),
                        "Semantic Similarity": semantic_similarity,
                        "Coherence": coherence,
                        "Accuracy": calculate_unified_accuracy(gold_response, response),
                        "Response": response,
                    }

                    # Determine the verdict for models below 95% accuracy
                    if accuracy < 95:
                        comparisons.append({
                            "Model": model["name"],
                            "Metrics": metrics,
                            "Verdict": f"Retain the previous model ({gold_model})",
                            "Reasoning": f"Accuracy of this model is not up to par (below 95%)."
                        })
                    else:
                        comparisons.append({
                            "Model": model["name"],
                            "Metrics": metrics,
                            "Verdict": None,
                            "Reasoning": None
                        })

                except Exception as e:
                    comparisons.append({
                        "Model": model["name"],
                        "Metrics": {
                            "Latency (ms)": "N/A",
                            "Token Economy Gain (%)": "N/A",
                            "Accuracy": "N/A",
                            "Response": f"Error: {str(e)}"
                        },
                        "Verdict": f"Retain the previous model ({gold_model})",
                        "Reasoning": f"Error occurred while processing this model: {str(e)}"
                    })

        # Select models preferred for latency and token efficiency
        eligible_models = [m for m in comparisons if m["Metrics"]["Accuracy"] != "N/A" and m["Metrics"]["Accuracy"] >= 95]
        best_latency_model = min(eligible_models, key=lambda x: x["Metrics"]["Latency (ms)"], default=None)
        best_token_model = max(eligible_models, key=lambda x: x["Metrics"]["Token Economy Gain (%)"], default=None)

        # Update verdicts and reasoning for eligible models
        for comparison in comparisons:
            if comparison["Verdict"] is None:  # Eligible models with accuracy >= 95
                if comparison == best_latency_model:
                    comparison["Verdict"] = f"Preferred for Latency"
                    comparison["Reasoning"] = f"This model has the best latency."
                elif comparison == best_token_model:
                    comparison["Verdict"] = f"Preferred for Token Efficiency"
                    comparison["Reasoning"] = f"This model has the best token efficiency."
                else:
                    comparison["Verdict"] = f"Retain the previous model ({gold_model})"
                    comparison["Reasoning"] = f"Another model is preferred for specific metrics."

        # Final Decision
        final_decision = {
            "Best Model": gold_model if not eligible_models else max(eligible_models, key=lambda x: x["Metrics"]["Accuracy"])["Model"],
            "Preferred for Latency": best_latency_model["Model"] if best_latency_model else "None",
            "Preferred for Token Efficiency": best_token_model["Model"] if best_token_model else "None",
        }

        results_by_prompt[prompt] = {
            "Gold Standard": gold_metrics,
            "Comparisons": comparisons,
            "Final Decision": final_decision
        }

    return results_by_prompt


if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()

        if not input_data:
            print(json.dumps({"error": "No data received"}))
            sys.exit(1)

        prompts = json.loads(input_data)
        results = evaluate_from_spreadsheet(prompts)
        print(json.dumps(results))


    except Exception as e:
        import traceback
        error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
        print(json.dumps({"error": error_message}))
        sys.exit(1)