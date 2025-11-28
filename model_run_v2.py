import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import json
import numpy as np
import requests
import textwrap
import string
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


EMBED_MODEL_NAME = r"Models/paraphrase-multilingual-MiniLM-L12-v2" 


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" 

EMBED_FILE = "tbk_chunks_embeddings.npy"
META_FILE = "tbk_chunks_metadata.json"


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


print("Modeller ve veriler yükleniyor...")


embed_model = SentenceTransformer(EMBED_MODEL_NAME)


reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)


embeddings = np.load(EMBED_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def simple_tokenizer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

tokenized_corpus = [simple_tokenizer(doc["text_preview"]) for doc in metadata]
bm25 = BM25Okapi(tokenized_corpus)

print("Sistem hazır!\n")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def search_semantic(query, top_k=10):
    """Vektör benzerliğine göre arama yapar."""
    q_emb = embed_model.encode(query, convert_to_numpy=True)
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((i, score)) # (index, score)
    
    # Puana göre sırala
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search_bm25(query, top_k=10):
    """Kelime eşleşmesine (BM25) göre arama yapar."""
    tokenized_query = simple_tokenizer(query)
    
    doc_scores = bm25.get_scores(tokenized_query)
    
    
    scores = [(i, score) for i, score in enumerate(doc_scores)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def hybrid_search_and_rerank(query, top_k_final=3):
    """
    1. Semantic Search yap (Top 10)
    2. BM25 Search yap (Top 10)
    3. Sonuçları birleştir (Tekilleştir)
    4. Cross-Encoder ile yeniden puanla (Rerank)
    5. En iyi 'top_k_final' sonucu döndür
    """
    
    
    semantic_results = search_semantic(query, top_k=10)
    bm25_results = search_bm25(query, top_k=10)
    
    
    unique_indices = set()
    for idx, _ in semantic_results:
        unique_indices.add(idx)
    for idx, _ in bm25_results:
        unique_indices.add(idx)
        
    candidate_indices = list(unique_indices)
    
    
    if not candidate_indices:
        return []

   
    cross_inp = []
    for idx in candidate_indices:
        text = metadata[idx]["text_preview"]
        cross_inp.append([query, text])
        

    rerank_scores = reranker_model.predict(cross_inp)
    
    
    final_results = []
    for i, idx in enumerate(candidate_indices):
        final_results.append({
            "chunk_id": idx,
            "score": float(rerank_scores[i]), 
            "text": metadata[idx]["text_preview"]
        })
        
    
    final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
    
    return final_results[:top_k_final]



def ask_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"API Hatası: {e}"

def rag_answer(question):
    
    retrieved = hybrid_search_and_rerank(question, top_k_final=3)

    if not retrieved:
        return "İlgili bilgi bulunamadı.", []

    context_text = ""
    for r in retrieved:
        context_text += (
            f"[CHUNK {r['chunk_id']} | Rerank Score={r['score']:.4f}]\n"
            f"{r['text']}\n\n"
        )

    prompt = f"""
Aşağıdaki bağlam kira hukuku ile ilgili kanun maddelerinden oluşmaktadır. 
Görevin: Sadece bağlamdaki bilgilere dayanarak soruyu yanıtlamaktır.

Kurallar:
1) Eğer soru bağlamdaki maddelerle ilgili değilse:
   - "Bu soru kira mevzuatıyla ilgili değildir, bu nedenle cevap veremem." diye yanıtla.
2) Eğer bağlamda ilgili bilgi yoksa:
   - "Bu konuda bağlamda bilgi bulunmamaktadır." de.
3) Bağlam dışı MADDE numarası uydurma.
4) Gereksiz uzun açıklamalar yapma.

Cevap formatı:

1) İlgili Maddeler:
(bağlamda gerçekten bulunan maddeleri listele, yoksa 'Yok' yaz)

2) Kısa Hukuki Değerlendirme:
(bağlamda varsa kısa yorum; yoksa 'Yok')

3) Sorunun Net Yanıtı:
(sadece bağlama dayalı cevap)

4) Sonuç:
(1 paragraf)

---

BAĞLAM:
{context_text}

SORU: {question}

Kısa, net ve tamamen bağlama dayalı cevap ver.
"""
    
    print("\n--- SYSTEM PROMPT GÖNDERİLİYOR ---\n")
    
    answer = ask_ollama(prompt)
    return answer, retrieved


if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        q = input("Soru (Çıkış için 'q'): ").strip()
        if q.lower() == 'q': break
        
        answer, retrieved = rag_answer(q)

        print("\n--- HYBRID & RERANK SONUÇLARI ---")
        for r in retrieved:
           
            print(f"Chunk {r['chunk_id']} | Score: {r['score']:.4f}")
            print(r['text'][:150].replace("\n", " ") + "...")
            print("-" * 30)

        print("\n--- MODEL CEVABI ---\n")
        print(answer)