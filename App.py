import re
from typing import List, Dict, Any

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel



# Configuration

MODEL_NAME = "bert-base-uncased"
SIMILARITY_THRESHOLD = 0.36

TASK_LABELS = {
    "create presentation": "Presentation",
    "write report": "Writing",
    "include introduction": "Section Requirement",
    "include methodology": "Section Requirement",
    "include conclusion": "Section Requirement",
    "include references": "Section Requirement",
    "analyze topic": "Analysis",
    "explain concept": "Explanation",
    "submit file": "Submission",
    "record video": "Media",
    "build program": "Programming",
    "answer questions": "Questions",
    "summarize article": "Summary",
}



# Model loading

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


# Text helpers

def split_sentences(text: str) -> List[str]:
    """
    Split prompt into simple sentences or line-based instructions.
    """
    raw_parts = re.split(r'[\n\r]+|(?<=[.!?])\s+', text)
    parts = [p.strip(" -•\t") for p in raw_parts if p.strip()]
    return parts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings using the attention mask.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_text(text: str) -> torch.Tensor:
    """
    Convert text to a normalized BERT embedding.
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        output = model(**encoded)

    pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.squeeze(0)


@st.cache_data
def precompute_label_embeddings() -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for task labels.
    """
    return {label: embed_text(label) for label in TASK_LABELS.keys()}


LABEL_EMBEDDINGS = precompute_label_embeddings()


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    """
    return torch.dot(vec1, vec2).item()


# Explainability

def merge_wordpieces(tokens: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Merge BERT wordpieces back into readable words.
    """
    merged = []
    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, scores):
        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
        else:
            if current_word:
                merged.append({"word": current_word, "score": current_score})
            current_word = token
            current_score = score

    if current_word:
        merged.append({"word": current_word, "score": current_score})

    return merged


def explain_prediction(sentence: str, target_label: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple occlusion-based explainability:
    remove one token at a time and measure how much the similarity drops.
    Larger drop = more important word.
    """
    encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]

    original_vec = embed_text(sentence)
    label_vec = LABEL_EMBEDDINGS[target_label]
    original_score = cosine_similarity(original_vec, label_vec)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_scores = []
    for i, token in enumerate(tokens):
        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            token_scores.append(0.0)
            continue

        masked_ids = torch.cat([input_ids[:i], input_ids[i + 1:]]).unsqueeze(0)
        masked_attention = torch.cat([attention_mask[:i], attention_mask[i + 1:]]).unsqueeze(0)

        with torch.no_grad():
            output = model(input_ids=masked_ids, attention_mask=masked_attention)

        pooled = mean_pool(output.last_hidden_state, masked_attention)
        masked_vec = F.normalize(pooled, p=2, dim=1).squeeze(0)
        masked_score = cosine_similarity(masked_vec, label_vec)

        importance = original_score - masked_score
        token_scores.append(max(0.0, importance))

    merged = merge_wordpieces(tokens, token_scores)

    # Filter out punctuation-like fragments and very small scores
    cleaned = []
    for item in merged:
        word = item["word"].strip()
        if re.fullmatch(r"[^a-zA-Z0-9]+", word):
            continue
        cleaned.append(item)

    cleaned.sort(key=lambda x: x["score"], reverse=True)
    return cleaned[:top_k]



# Prediction logic

def predict_tasks(text: str) -> List[Dict[str, Any]]:
    """
    Predict likely tasks from the input prompt.
    """
    sentences = split_sentences(text)
    results = []

    for sentence in sentences:
        sent_vec = embed_text(sentence)

        scores = []
        for label, label_vec in LABEL_EMBEDDINGS.items():
            sim = cosine_similarity(sent_vec, label_vec)
            scores.append((label, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = scores[0]

        if best_score >= SIMILARITY_THRESHOLD:
            explanation = explain_prediction(sentence, best_label, top_k=5)

            results.append(
                {
                    "sentence": sentence,
                    "predicted_task": best_label,
                    "category": TASK_LABELS[best_label],
                    "score": round(best_score, 4),
                    "important_words": [x["word"] for x in explanation],
                }
            )

    return results


# Streamlit UI

st.set_page_config(page_title=" Assignment Helper", layout="wide")

st.title("Assignment Helper Bot")
st.write(
    "Paste an assignment prompt below. The app uses BERT embeddings to detect likely tasks "
    "and shows the most influential words behind each prediction."
)

default_text = """Create a PowerPoint presentation with an introduction, methodology, expected results, and conclusion.
Include references at the end.
Write a short summary of the article.
Submit the final file as a PDF before Friday."""

user_text = st.text_area("Assignment Prompt", value=default_text, height=220)

if st.button("Analyze Prompt"):
    if not user_text.strip():
        st.warning("Please enter an assignment prompt.")
    else:
        predictions = predict_tasks(user_text)

        if not predictions:
            st.info("No strong task predictions were found. Try lowering the threshold or expanding the task labels.")
        else:
            st.subheader("Detected Tasks")

            for i, item in enumerate(predictions, start=1):
                st.markdown(f"### Task {i}")
                st.write(f"**Sentence:** {item['sentence']}")
                st.write(f"**Predicted Task:** {item['predicted_task']}")
                st.write(f"**Category:** {item['category']}")
                st.write(f"**Confidence Score:** {item['score']}")
                st.write(f"**Important Words:** {', '.join(item['important_words'])}")
                st.divider()

st.sidebar.header("About")
st.sidebar.write(f"Model: {MODEL_NAME}")
st.sidebar.write(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
st.sidebar.write(
    
)
