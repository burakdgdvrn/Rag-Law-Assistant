
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        
os.environ["TOKENIZERS_PARALLELISM"] = "false"   
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"        
os.environ["AUTOGRAPH_VERBOSITY"] = "0"         
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import requests
import textwrap



MODEL_NAME = "Models\paraphrase-multilingual-MiniLM-L12-v2"


EMBED_FILE = "tbk_chunks_embeddings.npy"
META_FILE = "tbk_chunks_metadata.json"


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def semantic_search(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(query, convert_to_numpy=True)

    embeddings = np.load(EMBED_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scores[:top_k]:
        results.append({
            "chunk_id": idx,
            "score": score,
            "text": meta[idx]["text_preview"]
        })
    return results


def ask_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code != 200:
        raise Exception(f"Ollama hatası: {response.text}")
    return response.json()["response"]


def rag_answer(question):
    retrieved = semantic_search(question, top_k=3)

    
    context_text = ""
    for r in retrieved:
        context_text += (
            f"[CHUNK {r['chunk_id']} | score={r['score']:.4f}]\n"
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



    print("\n--- OLLAMA PROMPT ---\n")
    print(textwrap.shorten(prompt, width=1200))

    answer = ask_ollama(prompt)
    return answer, retrieved


if __name__ == "__main__":
    q = input("Soru: ").strip()
    answer, retrieved = rag_answer(q)

    print("\n--- EN BENZER CHUNK'LAR ---")
    for r in retrieved:
        print(f"Chunk {r['chunk_id']} | score={r['score']:.4f}")
        print(r['text'][:300] + "...")
        print("----------------------------")

    print("\n--- OLLAMA CEVAP ---\n")
    print(answer)
