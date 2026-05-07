import anthropic
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

def load_artifacts(output_dir="output"):
    with open(f"{output_dir}/chunks.json") as f:
        chunks = json.load(f)
    index = faiss.read_index(f"{output_dir}/index.faiss")
    return chunks, index

def retrieve_relevant_chunks(query, chunks, index, model, top_k=TOP_K):
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["relevance_score"] = float(distances[0][i])
            results.append(chunk)
    return results

def answer_question(client, question, guidance, relevant_chunks):
    context = "\n\n".join([
        f"[Page {c['page']}]\n{c['text']}"
        for c in relevant_chunks
    ])

    prompt = f"""You are a model validation analyst reviewing model documentation.

Your task is to answer the following validation question using ONLY the document excerpts provided.

VALIDATION QUESTION:
{question}

GUIDANCE:
{guidance}

DOCUMENT EXCERPTS:
{context}

Instructions:
- Answer based strictly on evidence found in the excerpts
- If the answer is clearly present, set status to "passed"
- If the answer is partially present or unclear, set status to "needs_review"
- If there is no evidence at all, set status to "not_found"
- Always cite the page number where evidence was found
- Be concise but specific

Respond in this exact JSON format:
{{
  "status": "passed or needs_review or not_found",
  "answer": "your answer here",
  "evidence_quote": "direct quote from document or null",
  "page_reference": null,
  "notes": "any additional observations or null"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {
            "status": "needs_review",
            "answer": response_text,
            "evidence_quote": None,
            "page_reference": None,
            "notes": "JSON parsing failed"
        }

def run_extraction(questions_path="prompts/validation_questions.json",
                   output_dir="output"):
    print("Loading artifacts...")
    chunks, index = load_artifacts(output_dir)
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with open(questions_path) as f:
        question_sets = json.load(f)

    results = {}
    total = sum(len(s["questions"]) for s in question_sets.values())
    count = 0

    for section_key, section in question_sets.items():
        print(f"\nProcessing section: {section['sheet']}")
        results[section_key] = {
            "sheet": section["sheet"],
            "answers": []
        }

        for q in section["questions"]:
            count += 1
            print(f"  [{count}/{total}] {q['id']}: {q['question'][:60]}...")

            relevant = retrieve_relevant_chunks(
                q["question"], chunks, index, embed_model
            )

            answer = answer_question(
                client, q["question"], q["guidance"], relevant
            )

            results[section_key]["answers"].append({
                "id": q["id"],
                "question": q["question"],
                **answer
            })

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/extraction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/extraction_results.json")
    return results

if __name__ == "__main__":
    run_extraction()


def ingest_and_extract(pdf_path, output_dir="output", api_key=None):
    import anthropic
    from documents.ingest import extract_text_from_pdf, chunk_pages, embed_chunks, build_index
    import faiss, numpy as np
    from sentence_transformers import SentenceTransformer

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    embeddings, embed_model = embed_chunks(chunks)
    index = build_index(embeddings)

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    with open("prompts/validation_questions.json") as f:
        question_sets = json.load(f)

    results = {}
    for section_key, section in question_sets.items():
        results[section_key] = {"sheet": section["sheet"], "answers": []}
        for q in section["questions"]:
            relevant = retrieve_relevant_chunks(q["question"], chunks, index, embed_model)
            answer = answer_question(client, q["question"], q["guidance"], relevant)
            results[section_key]["answers"].append({
                "id": q["id"],
                "question": q["question"],
                **answer
            })
    return results
