import time
from foundry_local_sdk.exception import FoundryLocalException

SYSTEM_PROMPT = """You are a professional financial document analysis assistant. Answer ONLY using the provided context from the documents below.

CRITICAL RULES (follow strictly, no exceptions):
1. If the question asks for a price prediction, future stock value, or "should I buy/invest" — respond ONLY with: "I cannot provide investment advice or price predictions. This assistant only summarizes information explicitly stated in the documents." Do not attempt to speculate or reason toward an investment verdict.
2. If the answer is not clearly stated in the context, say so directly — do not guess, extrapolate, or combine unrelated details.
3. Pay close attention to WHO or WHAT is associated with each fact. A person who signs a document is not necessarily the authority that approved it. Double-check the exact relationship stated in the text before answering.
4. If a numeric or table-derived figure looks uncertain or ambiguous, say so and recommend checking the original document.
5. Answer ENTIRELY in the same language as the user's question — never mix languages within one response. If the question is in English, the answer must be in English. If in Turkish, the answer must be in Turkish.
6. Do NOT include page numbers, source citations, or references inside your answer text — this is handled separately by the application UI. Just give a clean, direct answer."""

def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {source}, Page {page}]\n{content}"
        for score, source, page, content in chunks
    )
    return f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""

def answer_query(question, embed_client, chat_client, conn, retries=2):
    from retrieval import get_top_chunks
    chunks = get_top_chunks(question, embed_client, conn, top_k=4)

    if not chunks:
        return "No relevant context found. Please ensure a document is uploaded first.", []

    prompt = build_prompt(question, chunks)

    for attempt in range(retries + 1):
        try:
            response = chat_client.complete_chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            return response.choices[0].message.content, chunks
        except FoundryLocalException as e:
            print(f"Error (attempt {attempt+1}): {e}")
            time.sleep(1)

    return "An error occurred while generating the response. Please try again.", chunks