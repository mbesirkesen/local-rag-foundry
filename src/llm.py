import hashlib
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any

HAS_FOUNDRY_LOCAL = False
try:
    from foundry_local_sdk import Configuration, FoundryLocalManager
    HAS_FOUNDRY_LOCAL = True
except ImportError:
    HAS_FOUNDRY_LOCAL = False

FOUNDRY_INIT_TIMEOUT_SEC = 300

class LLMEngine:
    def __init__(self, model_id: str = "Phi-4-mini-instruct-generic-cpu:5"):
        self.model_id = model_id
        self.client = None
        self.foundry_model = None
        self.is_foundry_active = False

        if not HAS_FOUNDRY_LOCAL:
            print("Foundry Local SDK yok, Fallback mod aktif.")
            return

        print("Foundry Local başlatılıyor (model indirme/yükleme sürebilir)...")
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._try_start_foundry)
            model = future.result(timeout=FOUNDRY_INIT_TIMEOUT_SEC)
            if model is not None:
                self.foundry_model = model
                self.is_foundry_active = True
                print(f"Foundry Local SDK aktif: {self.model_id}")
            else:
                print("Foundry Local model bulunamadı, Fallback mod aktif.")
        except FuturesTimeoutError:
            print("Foundry Local zaman aşımı, Fallback mod aktif.")
        except Exception as e:
            print(f"Foundry Local ilklendirilemedi, Fallback mod aktif: {e}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _try_start_foundry(self):
        config = Configuration("VerifiableLocalRAG")
        if getattr(FoundryLocalManager, "instance", None) is not None:
            manager = FoundryLocalManager.instance
        else:
            manager = FoundryLocalManager(config)

        models = manager.catalog.list_models()
        target_model = None
        for m in models:
            if m.id == self.model_id:
                target_model = m
                break

        if not target_model:
            print(f"Katalogda model yok: {self.model_id}")
            return None

        if not target_model.is_cached:
            print(f"Foundry Local model indiriliyor: {self.model_id}...")
            last = [-1]

            def _progress(pct):
                step = int(pct // 10)
                if step != last[0]:
                    last[0] = step
                    print(f"İndirme: %{int(pct)}")

            target_model.download(progress_callback=_progress)
        print(f"Model yükleniyor: {self.model_id}")
        target_model.load()
        return target_model

    def generate_embedding(self, text: str, engine: str = "auto") -> List[float]:
        """
        Metin için 384 boyutlu vektör üretir.
        engine: auto | foundry | fallback
        """
        use_foundry = engine != "fallback" and self.is_foundry_active and self.foundry_model
        if use_foundry:
            try:
                emb_client = self.foundry_model.get_embedding_client()
                res = emb_client.create(input=text)
                return res.data[0].embedding
            except Exception:
                pass
        return self._simple_hash_embedding(text)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Süreç yeniden başlasa da aynı vektörü üreten determinist fallback."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> str:
        if not context_chunks:
            return "Yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunmamaktadır."

        extractive = self._extractive_answer(context_chunks)

        if self.is_foundry_active and self.foundry_model:
            try:
                context_str = self._clean_chunk(context_chunks[0])[:900]
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Metin:\n{context_str}\n\n"
                            f"Soru: {query}\n"
                            "Sadece metindeki maddeleri kısa liste olarak yaz."
                        ),
                    },
                ]
                chat_client = self.foundry_model.get_chat_client()
                chat_client.settings.temperature = 0.1
                chat_client.settings.max_tokens = 280
                chat_client.settings.frequency_penalty = 1.1
                chat_client.settings.presence_penalty = 0.6
                chat_client.settings.top_p = 0.8
                response = chat_client.complete_chat(messages=messages)
                ans = self._clip_repetition((response.choices[0].message.content or "").strip())
                if self._is_usable_answer(ans, query):
                    return f"{ans}\n\n{self._source_line(context_chunks[0])}"
            except Exception as e:
                print(f"Foundry Local LLM yanıt hatası: {e}")

        return extractive

    @staticmethod
    def _clean_chunk(chunk: Dict[str, Any]) -> str:
        text = re.sub(r"\|+", " ", chunk.get("content") or "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _source_line(chunk: Dict[str, Any]) -> str:
        return f"(Kaynak: {chunk['source_file']}, Sayfa {chunk['page_number']})"

    def _extractive_answer(self, context_chunks: List[Dict[str, Any]]) -> str:
        chunk = context_chunks[0]
        text = self._clean_chunk(chunk)[:900]
        if not text:
            return "Yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunmamaktadır."
        return f"{text}\n\n{self._source_line(chunk)}"

    @staticmethod
    def _is_usable_answer(text: str, query: str) -> bool:
        if not text or len(text) < 40:
            return False
        low = text.lower().strip()
        leaked = (
            "uydurma",
            "belgedeki bilgiden",
            "kisa turkce",
            "kısa türkçe",
            "cevap ver",
            "sadece metindeki",
            "kaynak:",
            "soru:",
            "system",
        )
        if any(j in low for j in leaked):
            return False
        if low.startswith("kaynak") or low.startswith("[1]"):
            return False
        words = [w for w in re.findall(r"\w+", low) if len(w) > 2]
        if len(words) < 8:
            return False
        if len(set(words)) / max(len(words), 1) < 0.55:
            return False
        return True

    @staticmethod
    def _clip_repetition(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"(.{10,}?)(\s*[,:]?\s*\1){2,}", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
        parts = re.split(r"(?<=[.!?])\s+|,\s+", text)
        seen = set()
        out = []
        for part in parts:
            key = re.sub(r"\s+", " ", part.strip().lower())
            if len(key) < 4 or key in seen:
                continue
            seen.add(key)
            out.append(part.strip())
        clipped = ". ".join(out).strip() or text.strip()
        if len(clipped) > 800:
            clipped = clipped[:800].rsplit(" ", 1)[0] + "…"
        return clipped
