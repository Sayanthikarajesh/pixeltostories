import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import torch
import requests
from TTS.api import TTS  # Added TTS import from fullwithaudio.py

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, RegexpParser

# Load models, feature extractors, and tokenizers
# For captioning - using GIT model instead of VIT-GPT2
git_model_name = "microsoft/git-base-coco" # You can also use "microsoft/git-large" for better quality
processor = AutoProcessor.from_pretrained(git_model_name)
model = AutoModelForCausalLM.from_pretrained(git_model_name)

# Function to classify individual words
def classify_word(word, pos_tag):
    if pos_tag.startswith("NN"):
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        
    if pos_tag.startswith("VB"):
        return "Activity"
    return "Unknown"

# Function to classify the caption
def classify_caption(caption):
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)
    classifications = {}
    for word, pos in pos_tags:
        classifications[word] = classify_word(word, pos)
    return classifications

# Function to extract noun and verb phrases
def extract_phrases(caption):
    tokens = word_tokenize(caption)
    pos_tags = pos_tag(tokens)
    grammar = r"""
        NP: {<NN.*>}             
        VP: {<VB.><VBN|VBG|VB.>*}         
        VP: {<VB.*>}                         
        PP: {<IN>}       
    """
    chunk_parser = RegexpParser(grammar)
    chunked_tree = chunk_parser.parse(pos_tags)

    def extract_chunks(tree, label):
        return [" ".join(word for word, pos in subtree.leaves()) for subtree in tree.subtrees(filter=lambda t: t.label() == label)]

    noun_phrases = extract_chunks(chunked_tree, "NP")
    verb_phrases = extract_chunks(chunked_tree, "VP")
    prep_phrases = extract_chunks(chunked_tree, "PP")

    return noun_phrases, verb_phrases, prep_phrases

# Function to generate a caption using GIT model
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Process the image with GIT processor
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate the caption
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error processing image {image_path}: {e}"

# Function to process multiple images
def process_images(image_paths):
    results = {}
    for idx, path in enumerate(image_paths):
        caption = generate_caption(path)
        classifications = classify_caption(caption)
        noun_phrases, verb_phrases, prep_phrases = extract_phrases(caption)
        results[f"Image {idx + 1}"] = {
            "caption": caption,
            "classifications": classifications,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "prep_phrases": prep_phrases
        }
    return results

def build_knowledge_graph_nx(results):
    graph = nx.DiGraph()

    for Image, data in results.items():
        noun_phrases = data["noun_phrases"]
        verb_phrases = data["verb_phrases"]
        prep_phrases = data["prep_phrases"]

        # Add nodes (noun phrases)
        for noun in noun_phrases:
            graph.add_node(noun, color="blue")

        # Add edges (verb phrases as relationships)
        if verb_phrases:
            for verb in verb_phrases:
                if len(noun_phrases) > 1:
                    for i in range(len(noun_phrases) - 1):
                        graph.add_edge(noun_phrases[i], noun_phrases[i + 1], relationship=verb)
        elif prep_phrases:
            for prep in prep_phrases:
                if len(noun_phrases) > 1:
                    for i in range(len(noun_phrases) - 1):
                        graph.add_edge(noun_phrases[i], noun_phrases[i + 1], relationship=prep)

    return graph

def visualize_knowledge_graph_nx(graph, output_path):
    plt.figure(figsize=(12, 10))

    # Extract edge labels for relationships
    edge_labels = nx.get_edge_attributes(graph, 'relationship')

    # Assign node colors
    node_colors = [data["color"] for _, data in graph.nodes(data=True)]

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42, k=1.5)
    nx.draw_networkx_edges(graph, pos, edge_color="orange", arrows=True, width=2)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=2500)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", font_size=8)

    plt.title("Knowledge Graph", fontsize=20, fontweight='bold', pad=20)

    # Save the graph
    plt.savefig(output_path)
    plt.close()

# Function to generate a story using Llama 3 via Ollama
def generate_story(entities, story_output_path):
    entity_string = entities if isinstance(entities, str) else ', '.join(entities)
    print("entities - ", entity_string)
    
    # Provide a structured story prompt for Llama 3
    prompt = (f"Write a short story which has a meaningful plot with a good beginning, middle part and also a satisfactory ending. also give a heading to the story. "
              f"The story must incorporate the following entities and concepts from images: {entity_string}. "
              f"Make the story coherent, engaging, and logical, connecting all the entities in a natural way.")
    
    try:
        # Connect to Ollama API (make sure Ollama is running locally with Llama 3 model)
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3:latest",  # Make sure you have pulled this model with: ollama pull llama3
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_tokens": 600
                }
            }
        )
        
        if response.status_code == 200:
            story = response.json().get("response", "Error: No story generated")
        else:
            story = f"Error connecting to Ollama API: Status code {response.status_code}"
    except Exception as e:
        story = f"Error generating story: {str(e)}"
        print(story)
        
        # Fallback option: Use a different Llama model if available
        # try:
        #     response = requests.post(
        #         'http://localhost:11434/api/generate',
        #         json={
        #             "model": "mistral",  # Alternative model
        #             "prompt": prompt,
        #             "stream": False
        #         }
        #     )
        #     if response.status_code == 200:
        #         story = response.json().get("response", "Error: No story generated (fallback also failed)")
        # except:
        #     pass

    # Save the story to a file
    with open(story_output_path, 'w') as story_file:
        story_file.write(story)
    
    return story

# Function to process story generation from file
def process_story_generation(file_path, output_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Initialize variables to store noun phrases and relations
        noun_phrases = []
        relations = []

        for line in lines:
            if line.startswith("Noun Phrases:"):
                # Extract noun phrases and add them to the list
                noun_phrase_text = line.split("Noun Phrases:")[1].strip()
                # Handle different formats (list representation in text)
                if noun_phrase_text.startswith("[") and noun_phrase_text.endswith("]"):
                    try:
                        extracted_phrases = eval(noun_phrase_text)
                        if isinstance(extracted_phrases, list):
                            noun_phrases.extend(extracted_phrases)
                        else:
                            noun_phrases.append(noun_phrase_text)
                    except:
                        noun_phrases.append(noun_phrase_text)
                else:
                    noun_phrases.append(noun_phrase_text)
            elif line.startswith("Relation:"):
                # Extract relations and add them to the list
                relation_text = line.split("Relation:")[1].strip()
                # Handle different formats
                if relation_text.startswith("[") and relation_text.endswith("]"):
                    try:
                        extracted_relations = eval(relation_text)
                        if isinstance(extracted_relations, list):
                            relations.extend(extracted_relations)
                        else:
                            relations.append(relation_text)
                    except:
                        relations.append(relation_text)
                else:
                    relations.append(relation_text)

        # Combine noun phrases and relations into a single entity string
        all_entities = noun_phrases + relations
        
        # Filter out empties and "None"
        filtered_entities = [e for e in all_entities if e and e != "None"]
        
        entity_string = ', '.join(filtered_entities)

        # Print entity_string to check
        print("Entity String:", entity_string)

        # Generate story with the entity string
        generate_story(entity_string, output_path)

    except Exception as e:
        print(f"Error processing story generation: {e}")
        with open(output_path, 'w') as error_file:
            error_file.write(f"Error generating story: {str(e)}")

def generate_voice_from_story(
    story_path="outputs/generated_story.txt",
    speaker_wav="static/audio/rk.wav",
    output_path="outputs/audio.wav",
    language="en"
):
    # # Check if the story file exists
    # if not os.path.exists(story_path):
    #     print("[✗] Story file not found. Skipping XTTS voice generation.")
    #     return

    # Read the story text
    with open(story_path, "r", encoding="windows-1252") as file:
        text = file.read().strip()

    if not text:
        print("[✗] Story text is empty. Skipping XTTS voice generation.")
        return

    print("Generating voice using XTTS...")

    try:
        # Load XTTS model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path
        )

        print(f"[✓] Voice output saved to: {output_path}")
    except Exception as e:
        print(f"[✗] Error in XTTS voice generation: {e}")


# Main function to process all images in the static/uploads folder and clean up
def process_and_cleanup():
    upload_folder = 'static/uploads'
    output_folder = 'outputs'  # Changed from 'graphs' to 'outputs'
    
    # Create output directory if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image paths in the 'static/uploads' folder
    image_paths = [os.path.join(upload_folder, f) for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    
    if not image_paths:
        print("No images found in the 'static/uploads' folder.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Process images and save results
    results = process_images(image_paths)

    # Save captions and entities in a text file
    captions_entities_path = os.path.join(output_folder, "captions_and_entities.txt")
    with open(captions_entities_path, "w") as file:
        for img, result in results.items():
            file.write(f"{img}:\n")
            file.write(f"Caption: {result['caption']}\n")
            file.write(f"Noun Phrases: {result['noun_phrases']}\n")
            
            # Determine the relationship
            if result['verb_phrases']:
                relation = result['verb_phrases']
            elif result['prep_phrases']:
                relation = result['prep_phrases']
            else:
                relation = "None"

            file.write(f"Relation: {relation}\n\n")

    # Build and save the knowledge graph
    knowledge_graph = build_knowledge_graph_nx(results)
    visualize_knowledge_graph_nx(knowledge_graph, os.path.join(output_folder, "knowledge_graph.jpg"))

    # Generate and save story
    story_output_path = os.path.join(output_folder, "generated_story.txt")
    process_story_generation(captions_entities_path, story_output_path)
    
    # Generate audio from the story (new feature from fullwithaudio.py)
    audio_output_path = os.path.join(output_folder, "audio.wav")
    audio_path = generate_voice_from_story(
        story_path=story_output_path,
        output_path=audio_output_path
    )

    print(f"Captions and entities saved in '{captions_entities_path}'.")
    print(f"Knowledge graph saved as '{os.path.join(output_folder, 'knowledge_graph.jpg')}'.")
    print(f"Generated story saved in '{story_output_path}'.")
    if audio_path:
        print(f"Generated audio saved in '{audio_path}'.")

    # Create a summary file for easy frontend display
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, "w") as summary_file:
        for img, result in results.items():
            summary_file.write(f"{img}: {result['caption']}\n")

    # Delete all images from the 'static/uploads' folder
    for img_path in image_paths:
        os.remove(img_path)
        print(f"Deleted {img_path} from 'static/uploads'.")
    
    # Return the paths of generated files (useful for app.py)
    return {
        "captions_file": captions_entities_path,
        "graph_file": os.path.join(output_folder, "knowledge_graph.jpg"),
        "story_file": story_output_path,
        "audio_file": audio_path,  # New audio file path
        "summary_file": summary_path
    }

# Function to check if Ollama is running
def check_ollama_available(model_name="llama3:latest"):
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not any(model["name"] == model_name for model in models):
                print(f"WARNING: {model_name} model not found in Ollama. Please run: ollama pull {model_name}")
            return True
        return False
    except:
        print("WARNING: Ollama server not running. Please start Ollama before running this script.")
        return False

if __name__ == "__main__":
    # Check if Ollama is available before processing
    check_ollama_available()
    process_and_cleanup()