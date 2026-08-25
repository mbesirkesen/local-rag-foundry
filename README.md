# Local RAG (Foundry)

Yerelde çalışan, sayfa kaynaklı bir belge soru-cevap sistemi. PDF/TXT yükler, parçalara ayırır, soruya göre ilgili sayfayı bulur ve yanıtı Türkçe verir. Harici API zorunlu değildir: Microsoft Foundry Local varsa Phi-4 kullanılır, yoksa hash tabanlı yedek motorla arama yine çalışır.

**Yazar:** Muhammed Beşir Kesen

---

## Ne işe yarar?

Bir veya birden fazla belgeyi tarayıcıdan yükleyip doğal dilde soru sorarsınız. Sistem:

- yanıtı belgedeki ilgili sayfaya dayandırır
- `(Kaynak: dosya.pdf, Sayfa N)` satırı ekler
- belgede olmayan özel isim / hayali proje uydurmaz
- iki ayrı belgedeki konuları zorla birleştiren tuzak soruları reddeder

Arayüz Türkçedir. Ana uygulama FastAPI + `static/` web arayüzüdür (`http://127.0.0.1:8000`).

---

## Özellikler

- **Foundry Local (opsiyonel):** `Phi-4-mini-instruct-generic-cpu` ile çevrimdışı üretim. SDK yoksa hash-384 gömme ve çıkarımsal (extractive) yanıt devreye girer.
- **PDF/TXT ayrıştırma:** `pdfplumber` ile metin + tablo; kelime hizalı Markdown tablolar sayısal satırları korur. Yedek: `pypdf`.
- **Anlamsal parçalama:** Cümle / konu sınırına göre chunk; tablolar bütün tutulur.
- **Hibrit arama:** BM25 + sözcük eşlemesi + konu/varlık puanı + yeniden sıralama. Belge filtresi (tek dosya veya tüm belgeler).
- **Kısa bellek:** Takip sorularını önceki turla genişletir; yeni bir özel isim gelince önceki reddi yapıştırmaz.
- **Korumalar:** Çapraz belge (mixed-domain) reddi, belgede geçmeyen proje/özel isim, sayısal ve isimli olgular için çıkarıcı yanıtlar, yanıtın Türkçe tutulması.
- **Doğrulama:** Cümle düzeyinde Jaccard örtüşmesi (`src/verifier.py`).
- **Kaynak bağlantısı:** Arayüzde kaynak satırı ilgili PDF sayfasını açar.

---

## Mimari

```
[PDF / TXT]
    │
    ▼
[Ayrıştırıcı]  pdfplumber / pypdf + Markdown tablo
    │
    ▼
[Parçalayıcı]  anlamsal chunk + sayfa numarası
    │
    ▼
[Gömme]        Foundry embedding veya hash-384
    │
    ▼
[SQLite]       data/vector_store.db
    │
    ▼
[Getirici]     BM25 + sözcük + rerank + belge yönlendirme
    │
    ▼
[Yanıt]        korumalar → extractive / Foundry Phi-4 (Türkçe)
    │
    ▼
[Doğrulayıcı]  cümle-kaynak örtüşmesi
```

---

## Kurulum

Python 3.10+ gerekir.

```powershell
cd verifiable-local-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Foundry Local kuruluysa ilk çalıştırmada model indirilebilir; yoksa uygulama yedek motorda açılır.

---

## Çalıştırma

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Tarayıcı: [http://127.0.0.1:8000](http://127.0.0.1:8000)

1. Soldan PDF veya TXT yükleyin.
2. İsterseniz belge seçin (`Tüm belgeler` veya tek dosya).
3. Soruyu yazıp gönderin.

Yüklenen dosyalar `data/` altında kalır ve git’e **eklenmez**.

---

## Testler

```powershell
python tests/test_database.py
python tests/test_ingest.py
python tests/test_retriever.py
python tests/test_llm.py
python tests/test_verifier.py
```

---

## Proje yapısı

```
verifiable-local-rag/
├── api.py              FastAPI: yükleme, sohbet, belgeler
├── app.py              Eski Streamlit arayüzü (opsiyonel)
├── static/             Türkçe web arayüzü
├── src/
│   ├── ingest.py       PDF/TXT, tablo, anlamsal chunk
│   ├── database.py     SQLite vektör deposu
│   ├── retriever.py    Hibrit arama, konu puanı, koruma kalıpları
│   ├── memory.py       Takip sorusu, belge yönlendirme
│   ├── llm.py          Foundry / yedek motor, extractive yanıt, korumalar
│   └── verifier.py     Cümle-kaynak doğrulama
├── tests/              Birim testleri
├── data/               Yerel belgeler ve SQLite (gitignore)
├── requirements.txt
└── LICENSE             MIT
```

---

## Bağımlılıklar

`requirements.txt`: FastAPI, Uvicorn, pdfplumber, pypdf, NumPy, python-multipart, Foundry Local SDK.

---

## Lisans

MIT License — © 2026 Muhammed Beşir Kesen
