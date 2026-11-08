# Evaluation Methodology

**aircombat-bench v0.1** — `ENV_VERSION 1.0.0`, opponent `heuristic-v1`

Bu doküman benchmark'ın ölçüm protokolünü tanımlar. Amaç, farklı katkıcıların
ürettiği sayıların **birbiriyle karşılaştırılabilir** olmasıdır; bu alandaki
çalışmaların çoğu yeniden üretilebilir bir değerlendirme protokolü hiç
bildirmez.

Kanonik sonuçlar: [`docs/evals/v1.0/`](evals/v1.0/) — n=2000, seed 42, CPU tek
thread. Robustluk eki: [`docs/evals/robustness_cuda/`](evals/robustness_cuda/).
Tam kanıt zinciri: [`EVAL_PROTOCOL_AUDIT.md`](EVAL_PROTOCOL_AUDIT.md).

---

## 1. Eşleştirilmiş tasarım (matched design)

**Her yöntem aynı episode akışıyla değerlendirilir.** Episode *i* her modele
aynı başlangıç koşulunu sunar. Modeller arası karşılaştırma bu yüzden bağımsız
iki örneklem değil, **eşleştirilmiş** gözlemlerdir.

**Örtüşen Wilson aralıkları bir fark testi DEĞİLDİR.** Bu, sonuç tablosunun en
olası yanlış okumasıdır. Wilson aralıkları her modelin sabit rakibe karşı
*kendi* oranını tanımlar — ayrı ve doğru bir nicelik, ama iki modeli
karşılaştırmaz.

Eşleştirilmiş ikili sonuç için doğru test **McNemar**'dır (süreklilik
düzeltmeli). İki-oran z-testi, iki modelin aynı sonucu verdiği episode'ları —
verinin çoğunu — atar ve bu tasarımda ciddi biçimde güçsüzdür.

Ne kadar bilgi atıldığı ölçülebilir:

| Karşılaştırma | bağımsızlık altında beklenen uyuşma | gözlenen |
|---|---|---|
| 02 ↔ 03 | 58.0% | 85.2% |
| 05 ↔ 06 | 74.3% | 85.3% |
| 01 ↔ 02 | 49.2% | 74.9% |

Somut sonuç: iki-oran testiyle MAPPO+GAT → +OM+Role farkı p=0.067 ile
"anlamsız" görünürken, eşleştirilmiş testte p=0.00255'tir.

**Araç.** [`aircombat_bench/eval/compare.py`](../aircombat_bench/eval/compare.py):

```bash
python -m aircombat_bench.eval.compare \
    docs/evals/v1.0/02_mappo_gat_2000ep.json \
    docs/evals/v1.0/03_mappo_gat_om_role_2000ep.json
```

McNemar, tam binom işaret testi (sağlama olarak), episode indeksleri üzerinden
paired bootstrap CI'lar, uyuşma oranı ve sonucu değişen episode sayısını verir.

**Önkoşullar uyarı değil hatadır.** Senaryo, rakip, n, seed, seeding,
`env_version`, episode kimliği ve `runtime.device` eşit olmak zorundadır. Bir
CPU dosyası ile bir CUDA dosyası karşılaştırılamaz — cihaz etkisi ölçülen
etkilerle aynı mertebededir (§2).

---

## 2. Cihaz duyarlılığı — ölçülmüş

CPU ile CUDA aynı checkpoint ve aynı tohumlarla farklı sonuç verir. Neden:
sürekli kontrol çıktısında `nn.Linear` yığınının iki cihazdaki sonucu son
bitlerde ayrılır, fark RK4 dinamiğiyle büyür ve bir eşiği (WEZ konisi kenarı,
cooldown anı, timeout sınırı) geçtiğinde ayrık sonuç farkına dönüşür.

n=2000, aynı checkpoint, aynı tohumlar:

| Model | CPU | CUDA | Δ | paired CI95 | McNemar p | sonucu değişen |
|---|---|---|---|---|---|---|
| MAPPO baseline | 47.80% | 48.35% | +0.55 | [−1.15, +2.25] | 0.563 | 299 |
| MAPPO+GAT | 68.75% | 69.50% | +0.75 | [−0.75, +2.25] | 0.359 | 233 |
| MAPPO+GAT+OM+Role | 71.40% | 72.15% | +0.75 | [−0.70, +2.20] | 0.340 | 215 |
| QMIX | 64.05% | 64.05% | +0.00 | [−0.15, +0.15] | 1.000 | **2** |
| Pure FACMAC | 83.75% | 82.85% | −0.90 | [−2.25, +0.50] | 0.229 | 200 |
| FACMAC+OM+R | 86.05% | 84.85% | −1.20 | [−2.75, +0.35] | 0.151 | 256 |

Ortalama |Δ| = **0.69 pp**. Hiçbir model McNemar altında anlamlı değil.

**İşaretlerin yapısı — "karışık" değil, aile-içi tutarlı.** Üç MAPPO türevi
aynı yöne (+), iki FACMAC türevi diğer yöne (−) kayıyor; QMIX hiç kaymıyor.
Doğru ifade: **aile içinde tutarlı, aileler arasında zıt, hiçbiri tek başına
anlamlı değil.** İşaretleri "yönsüz" ya da "karışık" diye tanımlamak ölçümü
yanlış aktarır.

**QMIX bağışıklığı argmax kararlılığını bağımsız olarak doğrular.** 162 ayrık
aksiyon üzerinde `argmax`, ULP düzeyindeki farklara kararlıdır: QMIX 2000
episode'da **2** sonuç değiştiriyor, sürekli kontrol yöntemleri 200–299.

**Ölçeklenme.** Pure FACMAC'te CPU–CUDA farkı n=500'de 2.00 pp, n=2000'de
0.90 pp. 1/√n tahmini 1.00 pp. Örnekleme değişkenliğiyle tutarlı. **Dürüst
sınır:** her n için tek çiftlik karşılaştırma küçük bir sistematik etkiyi
dışlayamaz; bunun için çoklu seed gerekir.

**Sıralama ve anlamlılık hükümleri cihazdan bağımsızdır** (n=2000'de
doğrulandı): altı modelin sıralaması iki cihazda aynı ve beş yöntem
karşılaştırmasının tamamı — marjinal olan 02↔03 ve 05↔06 dâhil — iki cihazda
aynı anlamlılık hükmünü veriyor.

**Cihaz etkisini ölçmek için** `--device-comparison` gerekir (opt-in). O mod
cihaz farkına izin verir ama checkpoint hash eşitliğini ayrıca zorunlu kılar ve
çıktıyı cihaz karşılaştırması olarak etiketler; bir yöntem karşılaştırmasıyla
karıştırılamaz.

---

## 3. Yeniden üretilebilirlik — dürüstçe sınırlanmış

**Sonuçlar eşleşen bir runtime konfigürasyonunda bit-tekrarlanabilirdir.
Farklı konfigürasyonlar arasında sonuçlar bildirilen güven aralığı içinde
uyuşur — birebir değil.** Bundan fazlası iddia edilmemektedir.

**Kanonik konfigürasyon:**

| Alan | Değer |
|---|---|
| device | cpu |
| num_threads | 1 — runner tarafından sabitlenir |
| seeding | `legacy` (döngü öncesi tek `env.seed(seed)`) |
| seed | 42 |
| n_episodes | 2000 |
| deterministic | true |
| ENV_VERSION | 1.0.0 |
| opponent | `heuristic-v1` |

Thread sayısı `pin_threads()` ile runner içinde sabitlenir (`OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` modül import'unda `1`), **kullanıcının shell'i
tarafından değil** — böylece bir sonuç sessizce değiştirilemez. Ölçüm: bu
modellerde pin sayıları değiştirmiyor (batch=1'de küçük MLP'ler zaten tek
thread koşuyor); altı modelin tamamında 6-thread ve 1-thread ilk-500 episode
kayıtları bit-aynı.

**`runtime` bloğu** her çıktıda zorunludur: `device`, `torch_version`,
`blas_backend`, `num_threads`, `cpu_model`, `os`, `python_version`; CUDA'da
ayrıca `gpu_model`, `cuda_version`.

**Provenance.** `git_commit`, `git_dirty` ve `git_dirty_files` koşunun
**başında** yakalanır. Kanonik bir dosyada `git_dirty: true` doğrulayıcı
tarafından reddedilir.

**Replay garantisi — legacy seeding altında.** Env döngü öncesi bir kez
seed'lenir; episode *i*'nin başlangıç koşulu kendisinden önceki *i* episode'un
RNG çekilişlerine bağlıdır. Bu yüzden `ep_seed` **null**'dır ve episode'u
tanımlayan alan `stream_position`'dır. `seed + i` o episode'u izole yeniden
üretmez — ölçüldü: `env.seed(47)` ile legacy akışındaki 5. episode'un spawn'ı
20 km'den fazla ayrışıyor. **Kanonik dosyaların taşıdığı garanti: tüm akışın
yeniden üretilebilirliği, tek episode'un izole replay'i değil.** İzole replay
için `seeding="per_episode"` gerekir ve o mod legacy ile bit-aynı olmaz.

---

## 4. Dondurulmuş referans rakip

Tüm kanonik değerlendirmeler `heuristic-v1`'e karşıdır: kural tabanlı,
öğrenmeyen, sürüm etiketiyle sabitlenmiş (`HEURISTIC_VERSION`).

**Karşılaştırılabilirlik kuralı:** iki sayı yalnızca
**(`ENV_VERSION`, `opponent_version`, `runtime.device`)** üçlüsü eşleştiğinde
aynı tabloda kullanılabilir. Bu üçlünün herhangi biri değişirse sayılar farklı
bir ölçümün sonucudur.

---

## 5. Bilinen asimetriler

### 5.1 Kurtarma desteği — ölçülen yön

[`dogfight_env.py:308`](../aircombat_bench/envs/dogfight_env.py#L308) OOB/zemin
kurtarmasını yalnızca `blue_ids`'e uygular. Scripted red kendi kurtarmasına
sahiptir; **checkpoint sürülen red hiçbirini almaz.**

Tetiklenme sıklığı (150 ep, ajan-adımı başına):

| Dal | env (blue) | heuristic (scripted red) |
|---|---|---|
| zemin `h < 300 m` | 7.886% | 1.457% |
| stall `\|α\| > 22°` | **yok** (gerekseydi 5.011%) | 20.651% |
| düşük hız `v < 80 m/s` | **yok** (gerekseydi 8.181%) | 23.261% |
| harita kenarı | 0.000% | 0.000% |
| **toplam** | **7.886%** | **45.369%** |

**`01`–`06`'da kurtarma desteği scripted red lehinedir** — red, blue'nun
**5,8×'i** kadar destek alır ve blue'da stall ile düşük hız koruması hiç yoktur.
Dolayısıyla **raporlanan blue kazanma oranları bu asimetri bakımından
muhafazakârdır**. Altı yöntemin hepsi birebir aynı asimetriyle karşılaştığı
için `01`–`06` kendi içinde geçerlidir.

Asimetri blue lehine **yalnızca head-to-head koşularda** (`07a`/`07b`) döner,
çünkü orada checkpoint sürülen red hiç kurtarma almaz. Ölçülen yan etkisi:
FACMAC 73.30% (blue) / 69.80% (red); MAPPO 16.55% / 13.35%. İki model de blue
tarafından yaklaşık aynı miktarda kazanıyor (+3.50 ve +3.20 pp) — **iki-yönlü
ortalamayı savunulabilir kılan budur** ve head-to-head sayıları tam bu etkiyi
telafi etmek için iki yanın ortalaması olarak raporlanır.
`compare.combine_head_to_head()` aynalanmamış bir çifti reddeder; tek yönlü bir
sonuç raporda yan-yanlı olarak etiketlenir.

### 5.2 Sınır işleme kanonik rejimde atıldır

Yukarıdaki asimetri **zemin, stall ve düşük hız** dallarına ilişkindir.
Harita-kenarı dalı hiç tetiklenmez: ±25 km'lik arenada blue'nun sınıra en
yakın yaklaşması 14.036 m (marjının 2,8× uzağı), red'inki 9.194 m (3,1×);
300 episode / 359k ajan-adımında marj içinde sıfır adım. Koşul yanlış değil —
uçaklar kenara yaklaşmıyor. Gerçek OOB de olmuyor: altı kanonik koşunun
hepsinde `metrics.oob_rate = 0.0`.

Yüksek OOB oranları (0,34–0,42) **self-play eğitimine** aittir,
değerlendirmeye değil. Bu bir hata değildir; alt sistem kanonik değerlendirme
rejiminde tamamen atıldır.

---

## 6. Çoklu karşılaştırma disiplini — bir doğrulayıcı aile, gerisi keşifsel

Benchmark tek bir koşu kümesinden birden çok ikili karşılaştırma bildirir.
Yapı **tek bir doğrulayıcı aile** üzerine kuruludur; bir karşılaştırma **tam
olarak bir aileye** aittir ve iki kez düzeltilemez.

### 6.1 Doğrulayıcı (confirmatory) aile — üyeliği sabit, 5 karşılaştırma

Bildirinin manşet iddialarını taşıyan yöntem karşılaştırmaları. **Holm
düzeltmesi yalnızca burada uygulanır.**


> **Bildiri metninde p-değerleri 0.001'de kesilir** (`p < 0.001`).
> Aşağıdaki tam değerler bilerek korunmuştur: bir yeniden üretilebilirlik
> sağlamasıdır — `compare.py`'yi koşturan biri aynı sayıyı aldığını
> doğrulayabilmelidir. 10⁻⁷⁷ mertebesindeki bir sayı gerçek bir olasılık
> değildir; asimptotik yaklaşım çok öncesinde geçerliliğini yitirir.
> Ayrıntı: `PAPER_DELTA_v0.1.md` §A4f.

| # | Karşılaştırma | Δ win | ham p | Holm p |
|---|---|---|---|---|
| 1 | MAPPO → MAPPO+GAT | +20.95 pp | 1.6e-77 | 7.9e-77 |
| 2 | MAPPO+GAT → MAPPO+GAT+OM+Role | +2.65 pp | 0.00255 | **0.0051** |
| 3 | QMIX → MAPPO+GAT | +4.70 pp | 6.5e-08 | 1.9e-07 |
| 4 | FACMAC → FACMAC+OM+R | +2.30 pp | 0.00868 | **0.0087** |
| 5 | MAPPO+GAT → FACMAC | +15.00 pp | 9.0e-48 | 3.6e-47 |

Beşi de Holm sonrası anlamlıdır. Üyelik
[`compare.CONFIRMATORY_FAMILY`](../aircombat_bench/eval/compare.py) içinde
sabittir ve bir test bunu doğrular — üyelik değişirse yukarıdaki Holm
değerleri geçersiz olur.

### 6.2 Keşifsel (exploratory) — düzeltme uygulanmaz

Rol mekanizması serisi: yapısal ablasyonlar (06↔08, 03↔09) ve rol müdahale
deneyleri (C0–C6). Bunlar bir performans iddiasını taşımaz; **mekanizma
araştırmasıdır**. Ham p-değerleri ve CI'larla raporlanırlar ve doğrulayıcı
aileyle rekabet eden ikinci bir düzeltilmiş aile oluşturulmaz.

**Sonuçları betimleyicidir, doğrulayıcı değildir.** Seri ön-kayıtlı yorumlarla
yürütülmüştür ([`PREREGISTRATION_T2.10.md`](PREREGISTRATION_T2.10.md)) ve
ön-kayıtlı okumaların hiçbiri gerçekleşmemiştir — bu, sonucun kendisidir.

**Serinin ana bulgusu iki kümeli bir yapıdır, bir aralık değil.** Rol
asimetrisi sürekli değişmez; ajana hizalanmış girdilerde 0.34–0.44, diğer her
koşulda 0.02–0.13 olmak üzere **arasında hiçbir koşul bulunmayan** iki kümeye
düşer (boşluk 0.21, paired bootstrap CI'larla da kapanmaz). Yöneten değişken
ayrıklık değil **hizalanmadır**: atayıcının kendi çıktısı ajanlar arasında
takas edildiğinde (C4) vektörler hâlâ ayrıktır ama asimetri alt kümeye çöker.
Herhangi bir yerde aralık olarak yazılmamalıdır.

| Karşılaştırma | Δ win | paired CI95 | ham p |
|---|---|---|---|
| 06 ↔ 08 rol sıfırlanmış (FACMAC) | −2.25 pp | [−4.05, −0.50] | 0.01436 |
| 03 ↔ 09 rol sıfırlanmış (MAPPO) | +1.65 pp | [+0.00, +3.25] | 0.05452 |

**Bu yapı neden böyle.** Ablasyonlar daha önce doğrulayıcı aileye dâhildi ve
aynı karşılaştırma (06↔08) hangi aile içinde düzeltildiğine göre iki farklı
hüküm alıyordu — kanonik 7'li ailede anlamlı, rol deneyi ailesinde anlamsız.
Bu durum dürüstçe raporlanabilir ama okuyucuya karışıklık olarak görünür.
Ablasyonları ait oldukları yere — keşifsele — taşımak sorunu **kaynağında**
çözer.

### 6.3 Cihaz karşılaştırmaları — ayrı, yöntem ailesi değil

`--device-comparison` ile üretilen karşılaştırmalar (§2) ayrı bir kategoridir:
aynı model, farklı donanım. Bir yöntem karşılaştırması değildir ve yöntem
aileleriyle birlikte düzeltilmez.

### 6.4 Araçta zorunlu kılınması

`compare.py` aileyi **açık bir argüman** olarak alır (`--family` /
`family=...`) ve çıktıya kaydeder. Belirtilmezse `"unspecified"` yazılır.
`holm_confirmatory()` üyeliği doğrular: doğrulayıcı ailede olmayan bir
karşılaştırma Holm'a sokulursa `FamilyError` fırlatır. Böylece hiçbir
karşılaştırma sessizce iki kez düzeltilemez.

---

## 7. Bildiri için sıkıştırılmış hâl (experimental setup)

> Tüm yöntemler ortak bir protokolle değerlendirilir: `heuristic-v1`
> dondurulmuş rakibe karşı, `ENV_VERSION 1.0.0` altında, 2000 deterministik
> episode, seed 42, CPU tek thread (thread sayısı değerlendirme koşucusu
> tarafından sabitlenir). Model başına oranlar Wilson %95 aralığıyla verilir.
>
> Yöntemler **aynı episode tohumlarıyla** değerlendirildiğinden modeller arası
> karşılaştırmalar eşleştirilmiş gözlemlerdir ve **McNemar** testiyle
> yapılır; örtüşen Wilson aralıkları bu tasarımda bir fark testi değildir.
> Gözlenen sonuç uyuşması bağımsızlık altında beklenenin belirgin üzerindedir
> (ör. 85.2% vs 58.0%), dolayısıyla iki-oran testi ciddi biçimde güçsüzdür.
> Çoklu karşılaştırma için Holm düzeltmesi, açıkça adlandırılmış yedi
> karşılaştırmalık aile üzerinde uygulanır.
>
> Çıkarım cihazının ölçülebilir bir etkisi vardır: aynı checkpoint ve aynı
> tohumlarla CPU–CUDA farkı n=2000'de ortalama 0.69 pp'dir ve hiçbir modelde
> tek başına anlamlı değildir. İşaretler mimari ailesi içinde tutarlı, aileler
> arasında zıttır; QMIX 2000 episode'da yalnızca 2 sonuç değiştirir, sürekli
> kontrol yöntemleri 200–299 — ayrık `argmax` seçiminin kayan nokta
> farklarına kararlılığıyla tutarlı. Yöntem sıralaması ve tüm anlamlılık
> hükümleri iki cihazda özdeştir. Sonuçlar eşleşen bir runtime
> konfigürasyonunda bit-tekrarlanabilirdir; konfigürasyonlar arasında
> bildirilen aralık içinde uyuşurlar, birebir değil. Her çıktı cihaz, kütüphane
> sürümleri, BLAS backend'i, thread sayısı ve donanımı içeren bir `runtime`
> bloğu taşır.
