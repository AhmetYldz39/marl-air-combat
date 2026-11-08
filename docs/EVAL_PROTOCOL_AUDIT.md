# T2.5 — Evaluation Protocol Audit

**Tarih:** 2026-09-09
**Kapsam:** T2.5 Step 1b / 1c / 1d
**Durum:** ✅ **KÖK NEDEN BULUNDU VE KANITLANDI**

---

## 1. Karar

Yayımlanan `docs/evals/01–06` sayıları ile T2.2 unified-runner sayıları
arasındaki farkın tek nedeni **inference cihazıdır (CPU vs CUDA)**.

- **Ortam değişmedi.** Eski eval script'leri, bugünkü ortam ve bugünkü
  checkpoint'lerle çalıştırıldığında kendi yayımlanan sayılarını **birebir**
  üretiyor.
- **Protokol değişmedi.** Seeding şeması, episode döngüsü, rakip kurulumu,
  kazanma/berabere sınıflandırması ve politika yolu eski script'lerle unified
  runner arasında bit düzeyinde aynı.
- **Runner'da hata yok.** Aynı cihazda unified runner ile eski script
  bit-aynı sonuç veriyor (410/87/3 — üç metrik de aynı).
- Yayımlanan tüm sayılar **CUDA**'da üretilmiş; T2.2 **CPU**'da koşmuş.

Bu doküman `ENV_CHANGE_AUDIT.md` adıyla planlanmıştı. Denetlenecek bir ortam
değişikliği yok — kanıt bu çerçeveyi çürüttü — bu yüzden adı
`EVAL_PROTOCOL_AUDIT.md`.

---

## 2. Belirleyici Deney

Eski eval script'leri hâlâ repoda. Bunları **bugünkü kodda, bugünkü ortamda,
bugünkü checkpoint'lerle** çalıştırmak testin tamamı: eğer kendi yayımlanan
sayısını üretiyorlarsa ortam değişmemiştir.

| # | Hat | Cihaz | n | seed | win% | kill/ep | W/D/L | Referans | Sonuç |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `evaluation/eval_facmac.py` | CUDA | 500 | 42 | **82.0** | 1.730 | 410/87/3 | yayım. 82.0 | ✅ birebir |
| 2 | `evaluation/eval_facmac.py` (OM+R) | CUDA | 500 | 42 | **84.2** | 1.780 | 421/76/3 | yayım. 84.2 | ✅ birebir |
| 3 | `evaluation/eval.py` (MAPPO) | CUDA | 500 | 42 | **46.4** | 1.056 | — | yayım. 46.4 | ✅ birebir |
| 4 | `aircombat_bench.eval.runner` | CUDA | 500 | 42 | **82.0** | 1.730 | 410/87/3 | satır 1 | ✅ legacy ile bit-aynı |
| 5 | `aircombat_bench.eval.runner` | CPU | 500 | 42 | **84.0** | 1.746 | 420/74/6 | T2.2 84.0 | ✅ birebir |

Satır 1–3: ortam değişmedi.
Satır 1 vs 4: unified runner ile eski script aynı cihazda ayırt edilemez.
Satır 4 vs 5: **tek değişken cihaz, sonuç 2.0 pp kayıyor.**

Ayrıca `eval_facmac.py`'nin OM+R çıktısı yalnızca win rate'i değil intent ve
rol dağılımlarını da CLAUDE.md'deki değerlerle birebir üretti
(`sniper=0.247 pursuit=0.166 def=0.109 sup=0.478`) — ortamın ve OM yığınının
değişmediğinin bağımsız teyidi.

---

## 3. Elenen Hipotezler

Her biri okumayla değil ölçümle elendi.

| # | Hipotez | Yöntem | Sonuç |
|---|---|---|---|
| H1 | Runner non-deterministik (Case A) | Aynı seed'le iki koşu, CPU + CUDA, 6 model | ❌ Elendi — bit-aynı (`timestamp_utc` hariç). Case B. |
| H2 | MAPPO delta'sı seed=0 vs seed=42 | seed=0, n=500, üç MAPPO modeli | ❌ Elendi — 43.4/68.2/69.4; hiçbiri yayımlananı vermedi. Legacy script seed=**42** ile 46.4'ü tam verdi. |
| H3 | Ortam davranışı değişti (spawn/OOB/WEZ/termination/dinamik) | `git log --follow` + commit-commit diff; sonra eski script'i bugünkü ortamda koşturma | ❌ Elendi — commit'lenmiş davranış değişikliği yok; eski script bugünkü ortamda yayımlananı birebir üretti. |
| H4 | Reward ağırlıkları / config değişti | `config.yaml` diff, Mayıs 24 → bugün | ❌ Elendi — yalnızca training parametreleri (`total_episodes`, `pool_update_interval`, QMIX embed satırları). Eval'i etkileyen tek parametre yok. |
| H5 | Seeding şeması değişti (per-episode vs tek seferlik) | `eval.py` seeding'i 4 commit'te + runner'da incelendi; `--seeding` anahtarı eklenip ölçüldü | ❌ Elendi — **hepsi legacy**: döngü öncesi tek `env.seed(seed)`, döngü içinde yeniden seed yok. İki taraf da aynı şema. |
| H6 | FACMAC mimarisi değişti (öğrenilmiş `fire_head` → kural-tabanlı fire) | Checkpoint anahtarları okundu | ❌ Elendi — `facmac_base_ep2000.pt` içinde `fire_head` yok; checkpoint zaten yeni mimariyle eğitilmiş. |
| H7 | `MAPPOActor` paketleme sırasında değişti | 2d0ebc8 `train_mappo.py` vs paket `algorithms/mappo.py`, sınıf bazlı diff | ❌ Elendi — `MAPPOActor` ve `GATMAPPOActor` birebir aynı. |
| H8 | Runner'ın aksiyon yolu farklı davranıyor | Aynı env üzerinde iki yol adım adım karşılaştırıldı (3 ep, ~2000 adım) | ❌ Elendi — ağırlıklar max\|Δ\|=0.0; aksiyonlar bit-aynı. |
| H9 | `get_all_states()` yan etkili (runner adım başına 2 kez çağırıyor) | Kaynak incelemesi | ❌ Elendi — saf getter, `{aid: s.copy()}`, RNG tüketmiyor. |
| **H10** | **Cihaz (CPU vs CUDA) yörüngeyi değiştiriyor** | Aynı hat, aynı seed, iki cihaz; sonra n=500 doğrulama | ✅ **DOĞRULANDI** |

---

## 4. Kök Neden

FACMAC ve MAPPO politikaları **sürekli kontrol** çıktısı üretiyor
(aileron/elevator/rudder/throttle). Bir `nn.Linear` yığınının CPU ve CUDA
çıktıları son bitlerde farklıdır — cuBLAS farklı toplama sırası ve farklı
çekirdek seçer. Bu ULP düzeyindeki fark:

1. `clamp` sonrası kontrol yüzeyi komutuna geçer,
2. RK4 dinamiğiyle her adımda büyür,
3. bir eşiği geçtiğinde (WEZ konisi kenarı, `cooldown==0` anı, ölüm/timeout
   sınırı) **ayrık** bir sonuç farkına dönüşür,
4. 300–1000 adımlık episode'larda yörüngeyi tamamen ayırır.

12 episode'luk kontrollü karşılaştırma bunu doğrudan gösteriyor
(aynı seed, aynı kod, yalnızca cihaz farklı):

| ep | CPU | CUDA |
|---|---|---|
| 1 | draw / 1 kill / 1000 adım | blue / 2 kill / 281 adım |
| 3 | draw / 0 kill / 1000 adım | blue / 2 kill / 439 adım |
| 5 | blue / 2 kill / 459 adım | blue / 2 kill / 543 adım |
| 9 | blue / 2 kill / 524 adım | blue / 2 kill / 561 adım |
| **12 ep toplam** | **9 galibiyet** | **11 galibiyet** |

Fark rastgele değil ama yönü de sistematik değil — bu yüzden Step 1c'de her
model kaydı, hiçbiri tutarlı yönde kaymadı ve tüm kaymalar CI içinde kaldı.

### Neden QMIX etkilenmedi

QMIX 162 ayrık aksiyon üzerinden `q.argmax()` seçiyor. `argmax` ULP
düzeyindeki farklara karşı **kararlı**: iki Q değeri neredeyse eşit olmadıkça
seçilen indeks değişmez. Bu yüzden QMIX 65.0%'ı CPU'da ve CUDA'da aynı
üretiyor — ve bu yüzden T2.2'de tek birebir eşleşen model oldu.

QMIX'in eşleşmesi başlangıçta "ortam deterministik, öyleyse diğerleri de
eşleşmeliydi" çıkarımına yol açtı ve bu çıkarım araştırmayı yanlış yöne
(ortam değişikliği arayışına) sürükledi. Doğru okuma: QMIX'in eşleşmesi
ortamın değişmediğini gösteriyor — ki doğruydu — ama sürekli-kontrol
politikalarının cihaz duyarlılığı hakkında hiçbir şey söylemiyor.

---

## 5. Şemadaki Boşluk

`docs/evals/*.json` (eski format) ve `docs/evals/v0.1/*.json` (schema v1)
dosyalarının **hiçbiri cihazı kaydetmiyor.** Kayıtlı alanlar: `seed`,
`deterministic`, `n_episodes`, `env_version`, `git_commit`,
`checkpoint.sha256` — ama `device` yok.

Bu boşluk bu araştırmanın uzamasının doğrudan nedeni: sayıların hangi cihazda
üretildiği kayıtlı olsaydı fark ilk bakışta görülürdü. Sonuç
yeniden-üretilebilirliği için `device` (ve tercihen `torch.__version__`,
GPU adı) schema v1'e eklenmelidir.

---

## 6. Sonuçlar ve Öneri

**Yayımlanan sayılar geçerli ve yeniden üretilebilir** — CUDA'da, seed=42,
n=500, bugünkü ortamda. Hiçbir yayımlanan değer hatalı değil.

**T2.2 sayıları da geçerli** — CPU'da, aynı seed ve ortamda. İki set aynı
şeyin iki farklı cihazdaki ölçümü; hiçbiri diğerinin düzeltmesi değil.

Karşılaştırılabilirlik kuralı: **cihaz, seed ile aynı statüde bir eval
koşuludur.** Tek tabloda karıştırılamaz.

Kanonik tablo (T2.5 Step 4, n=2000) için öneri:

1. **Cihazı sabitle ve kaydet.** Tek cihazda koş, `device` alanını schema v1'e
   ekle, manifest ve senaryo dosyalarına yaz.
2. **CPU'yu tercih et.** Gerekçe: CUDA'da `torch.use_deterministic_algorithms`
   ayarlanmadığı sürece bit-tekrarlanabilirlik donanım ve sürücü sürümüne
   bağlıdır; CPU sonucu makineler arası taşınabilir. Maliyet: CPU inference
   daha yavaş (n=2000 için tek modelde ~1 saat mertebesi).
   Alternatif: CUDA'da kal ama `torch.use_deterministic_algorithms(True)`,
   `cudnn.deterministic=True`, `cudnn.benchmark=False` ekle ve GPU modelini
   kaydet — tez yeniden üretilebilirliği için CPU daha güvenli.
3. **Yayımlanan v0.1 sayılarını CUDA olarak etiketle.** Tezde ve
   `docs/evals/README_evals.md`'de "500 ep, seed=42, deterministic, **CUDA**"
   olarak geçmeli.
4. `ENV_VERSION` **değişmemeli** — ortam değişmedi. T2.5 kısıtı korunuyor.

Sıralama etkisi yok: her iki cihazda da FACMAC+OM+R en iyi, ardından pure
FACMAC, sonra MAPPO+GAT+OM+Role / MAPPO+GAT, QMIX, MAPPO baseline. Tezin
karşılaştırmalı iddiaları etkilenmiyor.

---

## 7. Kod Değişikliği

`aircombat_bench/eval/runner.py` — `--seeding legacy|per_episode` anahtarı
eklendi (varsayılan `legacy`).

- `legacy`: döngü öncesi tek `env.seed(seed)`. Eski script'lerin ve tüm
  yayımlanan sayıların şeması. **Varsayılan olduğu için runner'ın mevcut
  davranışı değişmedi** — T2.2 sayıları etkilenmiyor.
- `per_episode`: her episode öncesi `env.seed(seed + ep)`. T2.3 episode
  replay'i için gerekli; legacy ile bit-aynı olmaz (n=60'ta ölçüldü:
  legacy 0.8167 vs per_episode 0.8333, episode kayıtları farklı).

`seeding` alanı çıktı şemasına eklendi. `ep_seed` yorumu düzeltildi: legacy
modda bu değer env'e verilmiş bir seed değil, yalnızca etiket.

---

## 8. Düzeltilmesi Gereken Önceki Belge

`docs/T2.5_diagnosis.md` §4.7, MAPPO delta'larını seed=0 vs seed=42'ye ve
FACMAC delta'larını bir ortam değişikliğine bağlıyor. **İkisi de yanlış**;
Step 1c ve Step 1d bunları çürüttü. O bölüm bu dokümana atıfla
güncellenmelidir.

---

## 9. Step 4a — Cihaz Etkisinin n ile Ölçeklenmesi

Soru: cihaz farkı örnekleme gürültüsü mü, yoksa sistematik sapma mı?
"Aynı dağılım, farklı çekiliş" yorumu doğruysa fark 1/√n ile küçülmeli.

Pure FACMAC, seed=42, diğer her şey aynı:

| n | CPU (1 thread) | CUDA | fark |
|---|---|---|---|
| 500 | 84.00% | 82.00% | 2.00 pp |
| 2000 | **83.75%** [82.07, 85.30] | **82.85%** [81.14, 84.44] | **0.90 pp** |

1/√n tahmini: 2.00 / √4 = 1.00 pp. Gözlenen 0.90 pp.

Aynı p'nin iki bağımsız kestirimi için farkın standart hatası
`sqrt(2p(1−p)/n)`: n=500'de 2.38 pp, n=2000'de 1.19 pp. Gözlenen iki fark da
sıfırdan 1 SE içinde ve her nokta tahmini diğerinin %95 aralığında.

**Sonuç: gürültü gibi davranıyor, sistematik sapma kanıtı yok. CPU kanonik
olarak adlandırılabilir.**

**Sınır — fazlası iddia edilmemektedir:** bu her n için tek çiftlik bir
karşılaştırmadır. 1/√n ile uyum gürültüyle *tutarlıdır* ama küçük bir
sistematik sapmayı dışlamaz; bunun için çoklu seed gerekir. Doküman yalnızca
"gözlenen ölçekleme gürültü modeliyle tutarlı" der.

---

## 10. Step 4b — Kanonik Konfigürasyon

**Kanonik cihaz: CPU, tek thread.**

Thread pin runner'ın kendisi tarafından uygulanır — kullanıcının shell'i
tarafından değil:
- Modül import'unda `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` = `1`
- `run_eval` içinde `pin_threads()` → `torch.set_num_threads(1)` +
  `set_num_interop_threads(1)`

**Ölçüm — pin sayıları değiştirmiyor.** Altı modelin tamamında T2.2 (CPU,
6 thread) ile v1.0 (CPU, 1 thread) ilk 500 episode kayıtları **bit-aynı**
(§11 first-500 kontrolü). Sebep: batch=1'de küçük MLP'ler zaten tek thread
koşuyor, paralel indirgeme oluşmuyor. Pin bir garanti, sayı değişimi değil.

**`runtime` bloğu** schema v1'e eklendi ve `validate_schema_v1` tarafından
zorunlu kılındı: `device`, `torch_version`, `blas_backend`, `num_threads`,
`cpu_model`, `os`, `python_version`; CUDA koşularında ayrıca `gpu_model`,
`cuda_version`.

15 mevcut dosyaya geriye dönük dolduruldu (`v0.1/*` → CPU,
`docs/evals/01–09` → CUDA). Metriklerin değişmediği assert ile doğrulandı.
O dönemde kaydedilmemiş alanlar uydurulmadı, `"unknown"`; her blok
`reconstructed: true` ve `reconstruction_basis` taşıyor.

**Yeniden üretilebilirlik ifadesi (fazlası iddia edilmiyor):** Sonuçlar
eşleşen bir runtime konfigürasyonunda bit-tekrarlanabilirdir. Farklı
konfigürasyonlar arasında sonuçlar bildirilen güven aralığı içinde uyar —
birebir değil.

---

## 11. Kanıt Dosyaları

| Dosya | İçerik |
|---|---|
| `logs/legacy_facmac_base_500ep.json` | eski script, CUDA, 82.0% |
| `logs/legacy_facmac_omr_500ep.json` | eski script, CUDA, 84.2% |
| `logs/legacy_mappo_500ep.json` | eski script, CUDA, 46.4% |
| `logs/seed0_eval_mappo_500ep.json` | seed=0 denemesi, 43.4% |
| `logs/seed0_eval_mappo_gat_500ep.json` | seed=0 denemesi, 68.2% |
| `logs/seed0_eval_mappo_gat_om_500ep.json` | seed=0 denemesi, 69.4% |
| `docs/evals/v0.1/05_facmac_base_500ep.json` | T2.2, CPU, 84.0% |
| `docs/evals/05_facmac_base_500ep.json` | yayımlanan, CUDA, 82.0% |
