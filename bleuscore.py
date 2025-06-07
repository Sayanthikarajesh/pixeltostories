from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

# Load image from local path
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Load ViT-GPT2
def vit_gpt2_caption(image):
    vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values
    output_ids = vit_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Load GIT model
def git_caption(image):
    git_processor = AutoProcessor.from_pretrained("microsoft/git-base")
    git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    inputs = git_processor(images=image, return_tensors="pt")
    generated_ids = git_model.generate(**inputs, max_new_tokens=50)
    caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# BLEU Score
def compute_bleu_score(candidate, reference):
    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = [word_tokenize(reference.lower())]
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

# ROUGE Score
def compute_rouge_scores(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def compute_meteor_score(candidate, reference):
    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = word_tokenize(reference.lower())
    return meteor_score([reference_tokens], candidate_tokens)

# Main
if __name__ == "__main__":
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    image_path = "F:/S8/final/flicker8kimages/27782020_4dab210360.jpg"  # ✅ Replace this with your image path
    reference_caption = "A street vending machine is parked while people walk by"

    image = load_image(image_path)

    # 🔹 Generate captions
    vit_result = vit_gpt2_caption(image)
    git_result = git_caption(image)

    # 📢 Print generated captions
    print(f"\n🔷 ViT-GPT2 Caption: {vit_result}")
    print(f"🔶 GIT Caption     : {git_result}")
    print(f"✅ Reference Caption: {reference_caption}")

    # 🔸 BLEU
    vit_bleu = compute_bleu_score(vit_result, reference_caption)
    git_bleu = compute_bleu_score(git_result, reference_caption)

    # 🔸 METEOR
    vit_meteor = compute_meteor_score(vit_result, reference_caption)
    git_meteor = compute_meteor_score(git_result, reference_caption)

    # 🔸 ROUGE
    vit_rouge = compute_rouge_scores(vit_result, reference_caption)
    git_rouge = compute_rouge_scores(git_result, reference_caption)

    # 📊 Print scores
    print(f"\n📊 BLEU Score (ViT-GPT2): {vit_bleu:.4f}")
    print(f"📊 BLEU Score (GIT)     : {git_bleu:.4f}")

    print(f"\n🔷 METEOR Score (ViT-GPT2): {vit_meteor:.4f}")
    print(f"🔶 METEOR Score (GIT)     : {git_meteor:.4f}")

    print(f"\n🔷 ROUGE-1 (ViT-GPT2): {vit_rouge['rouge1'].fmeasure:.4f}")
    print(f"🔶 ROUGE-1 (GIT)     : {git_rouge['rouge1'].fmeasure:.4f}")

    print(f"\n🔷 ROUGE-L (ViT-GPT2): {vit_rouge['rougeL'].fmeasure:.4f}")
    print(f"🔶 ROUGE-L (GIT)     : {git_rouge['rougeL'].fmeasure:.4f}")
