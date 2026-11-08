# Eval Dosyaları

Tüm resmi değerlendirme sonuçları.  Sıra numaraları tezdeki model karşılaştırma
tablosuyla eşleşiyor.

Yeni (schema v1) dosyalar `v0.1/` altında; orijinal dosyalar bu dizinde.

---

## 0a. Modelleri karşılaştırırken — Wilson aralıkları bir fark testi DEĞİLDİR

Her model **aynı episode akışıyla** değerlendirilir: episode *i* her modele
aynı başlangıç koşulunu sunar. İki model arasındaki karşılaştırma bu yüzden
bağımsız iki örneklem değil, **eşleştirilmiş** gözlemlerdir.

**Örtüşen Wilson aralıklarına bakarak "fark anlamlı değil" sonucuna varmayın.**
Bu, bu tablodaki en olası yanlış okumadır. Wilson aralıkları her modelin sabit
rakibe karşı *kendi* kazanma oranını tanımlar — ayrı ve doğru bir nicelik, ama
bir fark testi değil. Eşleştirilmiş tasarımda iki-oran z-testi de yanlıştır:
iki modelin aynı sonucu verdiği episode'ları — verinin çoğunu — atar.

Ne kadar bilgi atıldığı ölçülebilir:

| Karşılaştırma | bağımsızlık altında beklenen uyuşma | gözlenen |
|---|---|---|
| 02 ↔ 03 | 58.0% | 85.2% |
| 05 ↔ 06 | 74.3% | 85.3% |
| 01 ↔ 02 | 49.2% | 74.9% |

**Baseline'a karşı karşılaştırma yapan katkıcılar `compare.py` kullanmalıdır:**

```bash
python -m aircombat_bench.eval.compare \
    docs/evals/v1.0/02_mappo_gat_2000ep.json \
    docs/evals/v1.0/03_mappo_gat_om_role_2000ep.json
```

McNemar (süreklilik düzeltmeli), tam binom işaret testi, episode indeksleri
üzerinden paired bootstrap CI'lar ve bağımsızlığa karşı uyuşma oranı verir.
Önkoşullar (senaryo, rakip, n, seed, seeding, `env_version`, episode kimliği
ve **`runtime.device`**) uyarı değil **hata** olarak uygulanır — bir CPU
dosyası ile bir CUDA dosyası karşılaştırılamaz.

---

## 0b. Bilinen sınırlama — ortam yan-simetrik değildir

[dogfight_env.py:308](../../aircombat_bench/envs/dogfight_env.py#L308):

```python
if aid in self.blue_ids:
    recovery = self._critical_recovery_blue(state)
```

OOB/zemin kurtarma yalnızca blue tarafına uygulanır. Scripted red kendi
kurtarmasına sahiptir (`HeuristicAgent._critical_recovery`), ama **checkpoint
sürülen red hiçbirini almaz.**

**Kapsam.** Yalnızca agent-vs-agent sonuçlarını etkiler (`07a`/`07b` ve
gelecekteki self-play veya checkpoint-vs-checkpoint koşuları).
**`01`–`06` etkilenmez:** orada red her zaman scripted rakiptir ve altı
yöntemin hepsi birebir aynı asimetriyle karşılaşır, dolayısıyla o
karşılaştırmalar kendi içinde geçerlidir.

**Ölçülen yan etkisi** (kanonik n=2000):

| Model | blue iken | red iken | fark |
|---|---|---|---|
| FACMAC+OM+R | 73.30% | 69.80% | +3.50 pp |
| MAPPO+GAT | 16.55% | 13.35% | +3.20 pp |

İki model de blue tarafından yaklaşık aynı miktarda kazanıyor (+3.50 ve
+3.20 pp) — **iki-yönlü ortalamayı savunulabilir kılan budur.** Head-to-head
sayıları tam da bu etkiyi telafi etmek için iki yanın ortalaması olarak
raporlanır; tek yönlü bir sayı yanlıdır.

`compare.combine_head_to_head()` aynalanmamış bir çift için `SideBiasError`
fırlatır ve tek yönlü bir `compare()` çıktısı raporda açıkça yan-yanlı olarak
etiketlenir.

**Harita-kenarı dalı ölü koddur** (ölçüm, 300 ep — `logs/map_edge_probe.json`).
Her iki kurtarma yolunda da bir harita-kenarı dalı var ve hiçbiri tetiklenmiyor;
sebep koşulun yanlışlığı değil, uçakların marja hiç yaklaşmaması. ±25 km'lik
arenada blue'nun sınıra en yakın yaklaşması 14.036 m (marjın 2,8× uzağı),
red'inki 9.194 m (3,1×). Kanonik eval'de gerçek OOB de olmuyor — altı modelin
hepsinde `metrics.oob_rate = 0.0`. Yüksek OOB oranları yalnızca self-play
*eğitiminde* görülür (0,34–0,42); alt sistem o rejim için yazılmıştır.

**Kurtarma yollarının kapsamı** (ölçüm, 150 ep, ajan-adımı başına tetiklenme):

| Dal | env (blue) | heuristic (scripted red) |
|---|---|---|
| zemin `h < 300 m` | %7.886 | %1.457 |
| stall `\|α\| > 22°` | **yok** (gerekseydi %5.011) | %20.651 |
| düşük hız `v < 80 m/s` | **yok** (gerekseydi %8.181) | %23.261 |
| harita kenarı | %0.000 (5000 m marj) | %0.000 (3000 m marj) |
| **toplam** | **%7.886** | **%45.369** |

Bulgu olduğu gibi: `01`–`06`'da kurtarma desteği **scripted red lehinedir**,
blue lehine değil — red 5.8× daha sık kurtarma alır ve blue'da stall ile
düşük hız koruması hiç yoktur. Harita kenarı dalı Faz-4 spawn'ında hiçbir
tarafta tetiklenmez.

v0.2 için planlanan kırıcı değişiklik: [ENV_CHANGELOG.md](../../ENV_CHANGELOG.md).

---

## 0. Yeniden Üretilebilirlik ve Kanonik Runtime

**Açık ifade:** Sonuçlar **eşleşen bir runtime konfigürasyonunda
bit-tekrarlanabilirdir.** Farklı konfigürasyonlar arasında sonuçlar bildirilen
güven aralığı içinde uyar — **birebir değil.** Bundan fazlası iddia edilmemektedir.

Bunun nedeni ölçülmüştür (bkz. `docs/EVAL_PROTOCOL_AUDIT.md`): FACMAC ve MAPPO
politikaları sürekli kontrol çıktısı üretir; `nn.Linear` yığınının CPU ve CUDA
sonuçları son bitlerde ayrılır, bu fark RK4 dinamiğiyle büyür ve bir eşiği
(WEZ konisi kenarı, cooldown anı, timeout sınırı) geçtiğinde ayrık sonuç
farkına dönüşür. QMIX bundan etkilenmez, çünkü 162 ayrık aksiyon üzerinde
`argmax` ULP düzeyindeki farklara kararlıdır.

**Kanonik konfigürasyon (T2.5 Step 4b):**

| Alan | Değer |
|---|---|
| device | **cpu** |
| num_threads | **1** — `aircombat_bench.eval.runner` tarafından sabitlenir |
| seeding | `legacy` (döngü öncesi tek `env.seed(seed)`) |
| seed | 42 |
| deterministic | true |
| ENV_VERSION | 1.0.0 |
| opponent | `heuristic-v1` |

Thread sayısı runner tarafından pin'lenir (`pin_threads()`; `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` vb. modül import'unda `1` yapılır), **kullanıcının shell
ortamı tarafından değil** — böylece sonuç sessizce değiştirilemez.

Ölçüm notu: bu modellerde thread sayısı sonucu değiştirmiyor (batch=1'de
küçük MLP'ler zaten tek thread koşuyor). CPU 1-thread ve CPU 6-thread FACMAC
n=500 sonucu bit-aynı: 84.00%, kill/ep 1.7460, 420/74/6. Pin bir güvence,
sayı değişimi değil.

**`runtime` bloğu.** Schema v1 artık her koşuda `runtime` kaydeder: `device`,
`torch_version`, `blas_backend`, `num_threads`, `cpu_model`, `os`,
`python_version` ve CUDA koşularında `gpu_model`, `cuda_version`. Mevcut
dosyalara bu blok geriye dönük eklendi ve `reconstructed: true` ile
işaretlendi — cihaz atamaları ölçümle kanıtlandı, diğer alanlar o dönemde
kaydedilmediği için `"unknown"`.

**Cihaz atamaları:**
- Bu dizindeki `01–09` (yayım dönemi): **CUDA**
- `v0.1/01–06` (T2.2 yeniden üretimi): **CPU**, thread pin'i yoktu

İki set aynı şeyin iki farklı cihazdaki ölçümüdür; hiçbiri diğerinin
düzeltmesi değildir. **Cihaz, seed ile aynı statüde bir eval koşuludur ve tek
tabloda karıştırılamaz.**

---

## 0c. KANONİK TABLO — v1.0 (n=2000, seed=42, CPU 1 thread, `heuristic-v1`)

`docs/evals/v1.0/`, commit `3b70591`, `git_dirty=false`, `ENV_VERSION 1.0.0`.
Tez ve bildiri için kullanılacak sayılar bunlardır.

| # | Model | Win% | 95% Wilson CI | Kill/ep | Draw% | W/D/L |
|---|---|---|---|---|---|---|
| 01 | MAPPO baseline | 47.80% | [45.62, 49.99] | 1.0700 | 50.20% | 956/1004/40 |
| 02 | MAPPO+GAT | 68.75% | [66.68, 70.74] | 1.5165 | 29.75% | 1375/595/30 |
| 03 | MAPPO+GAT+OM+Role | 71.40% | [69.38, 73.34] | 1.5600 | 26.55% | 1428/531/41 |
| 04 | QMIX | 64.05% | [61.92, 66.12] | 1.4215 | 33.40% | 1281/668/51 |
| 05 | Pure FACMAC | 83.75% | [82.07, 85.30] | 1.7565 | 15.50% | 1675/310/15 |
| 06 | **FACMAC+OM+R** | **86.05%** | [84.46, 87.50] | **1.8045** | 13.25% | 1721/265/14 |

**Doğrulayıcı aile** — 5 yöntem karşılaştırması (McNemar, süreklilik
düzeltmeli). **Holm yalnızca bu aile içinde uygulanır**; üyelik
`compare.CONFIRMATORY_FAMILY` içinde sabittir:


> **Bildiri metninde p-değerleri 0.001'de kesilir** (`p < 0.001`).
> Aşağıdaki tam değerler bilerek korunmuştur: bir yeniden üretilebilirlik
> sağlamasıdır — `compare.py`'yi koşturan biri aynı sayıyı aldığını
> doğrulayabilmelidir. 10⁻⁷⁷ mertebesindeki bir sayı gerçek bir olasılık
> değildir; asimptotik yaklaşım çok öncesinde geçerliliğini yitirir.
> Ayrıntı: `PAPER_DELTA_v0.1.md` §A4f.

| Karşılaştırma | Δ win | paired bootstrap CI95 | χ² | ham p | Holm p |
|---|---|---|---|---|---|
| MAPPO → MAPPO+GAT | +20.95 pp | [+18.85, +23.05] | 347.36 | 1.6e-77 | 7.9e-77 |
| MAPPO+GAT → +OM+Role | +2.65 pp | [+1.00, +4.35] | 9.10 | 0.00255 | **0.0051** |
| QMIX → MAPPO+GAT | +4.70 pp | [+3.00, +6.40] | 29.22 | 6.5e-08 | 1.9e-07 |
| FACMAC → FACMAC+OM+R | +2.30 pp | [+0.65, +4.00] | 6.89 | 0.00868 | **0.0087** |
| MAPPO+GAT → FACMAC | +15.00 pp | [+13.10, +16.90] | 210.85 | 9.0e-48 | 3.6e-47 |

**Beşi de Holm sonrası anlamlıdır.**

**Keşifsel** — yapısal ablasyonlar mekanizma araştırmasıdır, bir performans
iddiası taşımazlar; ham p ve CI ile raporlanır, **Holm uygulanmaz**:

| Karşılaştırma | Δ win | paired bootstrap CI95 | ham p |
|---|---|---|---|
| FACMAC+OM+R → rol sıfırlanmış (08) | −2.25 pp | [−4.05, −0.50] | 0.01436 |
| MAPPO+OM+Role → rol sıfırlanmış (09) | +1.65 pp | [+0.00, +3.25] | 0.05452 |

Aile yapısının gerekçesi: [`../EVALUATION_METHODOLOGY.md`](../EVALUATION_METHODOLOGY.md) §6.

### Head-to-head (07) — iki-yönlü ortalama

`07a` + `07b`, n=4000 (2×2000). Tek yönlü sayı yanlıdır; §0b'ye bakınız.

| Model | iki-yönlü | blue iken | red iken |
|---|---|---|---|
| FACMAC+OM+R | **71.55%** | 73.30% | 69.80% |
| MAPPO+GAT | **14.95%** | 16.55% | 13.35% |
| berabere | 13.50% | — | — |

### Yapısal ablasyon (08/09) — rol vektörü çıkarım anında sıfırlanır

**Bu bir *bağımlılık* (reliance) ölçümüdür, bir nedensellik ölçümü değil.**
Zaten eğitilmiş bir politikada rol vektörü çıkarım anında sıfırlanır; bu,
politikanın rol sinyaline ne kadar *dayandığını* gösterir. Rol atamasının
eğitim kazanımlarına *yol açtığını* göstermez — o, rol mekanizması olmadan
eğitim yapmayı gerektirir ve yapılmamıştır (bkz. §0d).

Taşıyıcı nicelik **ajan-düzeyi asimetri** `|mean(k₀) − mean(k₁)|`:

| Koşul | ajan başına kill | asimetri |
|---|---|---|
| Pure FACMAC (05) | [0.831, 0.925] | 0.0945 |
| FACMAC+OM+R normal (06) | [0.733, 1.071] | **0.3385** |
| FACMAC+OM+R rol sıfır (08) | [0.845, 0.934] | **0.0895** |
| MAPPO+OM+Role normal (03) | [0.768, 0.791] | 0.0230 |
| MAPPO+OM+Role rol sıfır (09) | [0.754, 0.818] | 0.0635 |

| Ablasyon | Δ asimetri | paired bootstrap CI95 | sonuç |
|---|---|---|---|
| FACMAC+OM+R 06 → 08 | −0.2490 | [−0.3320, −0.1645] | **anlamlı** |
| MAPPO+OM+Role 03 → 09 | +0.0405 | [−0.0380, +0.1025] | null (rol atıl) |

**Mekanizma sonucu.** Rol vektörü sıfırlandığında FACMAC+OM+R, rol mekanizması
hiç olmayan Pure FACMAC'ten **ayırt edilemez** hâle gelir — hem kazanma
oranında (83.80% vs 83.75%; Δ=+0.05 pp, CI95 [−1.85, +2.00], McNemar p=1.00)
hem de rol asimetrisinde (0.0895 vs 0.0945). OM+Role yığınının Pure FACMAC
üzerindeki ölçülen +2.30 pp'lik kazancının tamamı, rol sinyalinin çıkarım
anında mevcut olmasına atfedilebilir.

*Bağlam:* 06 (7.6M adım) 05'ten (622k adım) ~12× daha uzun eğitilmiştir. Rol
sıfırlandığında 06'nın tam olarak 05 düzeyine dönmesi, bu ek eğitimin rol
sinyali olmadan ölçülebilir bir getiri bırakmadığı anlamına gelir.

**Rol sinyali ne yapıyor** (T2.8.2, enstrümanlı n=2000 koşular): kalıcı bir
**işbölümü** kuruyor — kill payı 40.62/59.38'den rol sıfırlanınca
47.49/52.51'e çöküyor (fark +0.2490, CI95 [+0.1550, +0.3435]), yoğunlaşma ise
değişmiyor. Episode bazında kimin skor yapacağını *dinamik atadığı* iddiası
desteklenmiyor: pursuit boyutunun öngörü gücü ablasyondan sağ çıkıyor
(57.27% → 55.69%), yani atayıcı büyük ölçüde geometriyi betimliyor. Tek temiz
ablasyon imzası defensive boyutunda (57.49% → 50.58%, şans düzeyine iniyor).

> **İki "kill imbalance" tanımı karıştırılmamalıdır.** Yukarıdaki tablo
> ajan-düzeyi asimetriyi `|mean(k₀) − mean(k₁)|` kullanır — yayımlanan 08/09
> dosyalarının ve tezin taşıyıcı metriği budur. Episode-içi yoğunlaşma
> `mean(|k₀ − k₁|)` FARKLI bir niceliktir ve rol sinyaline duyarsızdır
> (06: 1.2705, 08: 1.2655, 05: 1.2715 — hepsi null). Tek bir episode'da
> kill'lerin bir ajanda toplanması, ajanların ortalamada simetrik olmasıyla
> çelişmez.

---

## 0c2. Robustluk eki — CUDA (kanonik DEĞİL)

`docs/evals/robustness_cuda/` — n=2000, seed 42, CUDA, aynı checkpoint ve aynı
tohumlar. **Kanonik tablo CPU'dur;** bu set yalnızca cihaz duyarlılığını
belgelemek içindir ve aynı tabloda CPU sayılarıyla karıştırılamaz.

| Model | CPU | CUDA | Δ | paired CI95 | McNemar p | sonucu değişen |
|---|---|---|---|---|---|---|
| MAPPO baseline | 47.80% | 48.35% | +0.55 | [−1.15, +2.25] | 0.563 | 299 |
| MAPPO+GAT | 68.75% | 69.50% | +0.75 | [−0.75, +2.25] | 0.359 | 233 |
| MAPPO+GAT+OM+Role | 71.40% | 72.15% | +0.75 | [−0.70, +2.20] | 0.340 | 215 |
| QMIX | 64.05% | 64.05% | +0.00 | [−0.15, +0.15] | 1.000 | **2** |
| Pure FACMAC | 83.75% | 82.85% | −0.90 | [−2.25, +0.50] | 0.229 | 200 |
| FACMAC+OM+R | 86.05% | 84.85% | −1.20 | [−2.75, +0.35] | 0.151 | 256 |

Ortalama |Δ| = **0.69 pp**; hiçbir model McNemar altında anlamlı değil.
İşaretler **mimari ailesi içinde tutarlı, aileler arasında zıt** (üç MAPPO
türevi +, iki FACMAC türevi −, QMIX 0) — "karışık" ya da "yönsüz" demek ölçümü
yanlış aktarır. QMIX 2000 episode'da 2 sonuç değiştirir, sürekli kontrol
yöntemleri 200–299; bu, argmax kararlılığı açıklamasını bağımsız doğrular.

**Sıralama ve anlamlılık hükümleri cihazdan bağımsızdır:** altı modelin
sıralaması iki cihazda aynı; beş yöntem karşılaştırmasının tamamı — marjinal
olan 02↔03 ve 05↔06 dâhil — iki cihazda aynı hükmü veriyor.

Cihaz karşılaştırması için `--device-comparison` (opt-in) gerekir.

---

## 0c3. Rol müdahale deneyi — sabit vs öğrenilmiş atama

`docs/evals/role_experiment/` — çıkarım-anı müdahaleleri, `facmac_omr_ep27000.pt`,
n=2000, seed 42, CPU. **Eğitim yok.** Kanonik tablodan ayrıdır.

**Seri KEŞİFSELdir** — ham p ve CI raporlanır, Holm uygulanmaz (§0a, metodoloji
§6). Tahminler koşulardan önce kaydedildi:
[`../PREREGISTRATION_T2.10.md`](../PREREGISTRATION_T2.10.md) (commit `a94b944`).

| | Koşul | L2 | Win% | Wilson CI95 | rol asimetrisi |
|---|---|---|---|---|---|
| C0 | öğrenilmiş | 1.00 | 86.05% | [84.46, 87.50] | 0.3385 |
| C1 | sıfır | 0.00 | 83.80% | [82.12, 85.35] | 0.0895 |
| C2 | sabit ayrık (a0=support, a1=pursuit) | 1.00 | 85.35% | [83.73, 86.83] | **0.4370** |
| C3 | sabit özdeş (ikisi pursuit) | 1.00 | **86.90%** | [85.35, 88.31] | 0.0515 |
| C4 | takas | 1.00 | 85.40% | [83.78, 86.88] | 0.1260 |
| **C5** | **uniform (bilgi taşımayan)** | **0.50** | **86.45%** | [84.88, 87.88] | 0.0550 |
| C6 | ölçekli, ε=0.5 | 0.50 | 86.75% | [85.19, 88.17] | 0.0615 |
| C6 | ölçekli, ε=0.1 | 0.10 | 84.20% | [82.54, 85.73] | 0.1050 |
| C6 | ölçekli, ε=0.01 | 0.01 | 83.05% | [81.34, 84.63] | 0.0215 |

### Sıfır-vektör confound'u — kapatıldı

Atayıcı `F.one_hot(...)` üretir, yani aktör eğitimde daima L2 normu 1.0 olan
bir one-hot görmüştür; **sıfır vektör eğitim dağılımı dışıdır.** C5 bunu test
eder: C1 kadar az rol bilgisi taşır ama sıfır değildir.

| | Δ | paired CI95 | ham p | hüküm |
|---|---|---|---|---|
| C0 → C5 | +0.40 pp | [−1.25, +2.10] | 0.677 | **ayırt edilemez** |
| C3 → C5 | −0.45 pp | [−2.10, +1.25] | 0.639 | ayırt edilemez |
| C1 → C5 | +2.65 pp | [+0.90, +4.40] | 0.0034 | anlamlı |

C6 ε küçüldükçe düzgün bozuluyor (86.90 → 86.75 → 84.20 → 83.05) — **dağılım
kayması imzası**. L2-eşleşmiş çift C5 ↔ C6@0.5: Δ=+0.30 pp, p=0.768, yani
aynı büyüklükte **biçim sayılmıyor**.

**Sonuç.** Kazanma oranı rol yolunun etkinleşme **büyüklüğüne** bağlıdır;
içeriğine, biçimine ya da ajanlara atanmasına değil. **"Rol vektörü kazanma
oranı için taşıyıcıdır" iddiası geri çekilmiştir.**

**Ayakta kalan:** rol *içeriği* davranışsal uzmanlaşmayı belirler — ve bu
**iki ayrık küme** hâlinde, sürekli bir aralık olarak değil:

| Rol girdisi | Koşullar | Asimetri |
|---|---|---|
| **ajana hizalanmış** | C2 (0.4370), C0 (0.3385) | **0.34 – 0.44** |
| hizalanmamış | C4 (0.1260), C6@0.1 (0.1050), C1 (0.0895), C6@0.5 (0.0615), C5 (0.0550), C3 (0.0515), C6@0.01 (0.0215) | **0.02 – 0.13** |

**Boşluk 0.21 ve bulgu budur**; paired bootstrap CI'larla bile kapanmıyor
(C0 [0.2710, 0.4070] vs C4 [0.0570, 0.1955]).

**Yöneten değişken ayrıklık değil, hizalanma.** C4 ajanlara *farklı* vektörler
verir (atayıcı çıktısı, takas edilmiş) ve yine de alt kümeye düşer. Tutarlı
hizalama uzmanlaşmayı artırır (C2 > C0, +0.0985 CI [+0.0240, +0.1730]); ters
hizalama ayrıklığa rağmen yok eder. **Ayrıklık gerekli ama yeterli değil.**

Bu koşulların hepsi olağan büyüklükte ve en yüksek kazanma oranlarında —
**davranış ile sonuç ayrışıyor.**

Ayrıntı: [`../PAPER_DELTA_v0.1.md`](../PAPER_DELTA_v0.1.md) §A4d–A4e.

---

## 0d. Eğitim-zamanı ablasyon — mevcut değil

Rol mekanizması olmadan eğitim yapan bir ablasyon **yoktur**. Mevcut olanlar
farklı şeyleri ablasyona sokar ve yalnızca MAPPO hattındadır:

| Run | Bayrak | Ne ablasyona sokuluyor |
|---|---|---|
| Run C, B, B2 | `--no-role-reward` | rol *koşullu ödül* — rol mekanizması yerinde kalır |
| Run B, D, D2, D3 | `--argmax-role` | rol *örnekleme yöntemi* (Gumbel → argmax) |

Pure FACMAC ile FACMAC+OM+R arasındaki fark bir eğitim-zamanı karşıtlığıdır
ama **karıştırıcı**: 622k vs 7.6M adım ve farklı havuz zamanlaması.

→ Rol mekanizmasız eğitim, gelecek çalışma olarak listelenmelidir.

---

## 1. Ana Model Karşılaştırması — ESKİ (500 ep, CUDA)

### Yayımlanan (authoritative) sonuçlar

| Dosya | Model | Checkpoint | Win% | 95% CI | Kill/ep | Draw% |
|---|---|---|---|---|---|---|
| `01_mappo_baseline_500ep.json` | MAPPO baseline | `mappo_final.pt` | 46.4% | [42.1, 50.8] | 1.06 | 52.0% |
| `02_mappo_gat_500ep.json` | MAPPO+GAT | `mappo_gat_ep37000.pt` | 69.8% | [65.6, 73.7] | 1.52 | 28.4% |
| `03_mappo_gat_om_role_500ep.json` | MAPPO+GAT+OM+Role | `mappo_gat_ep44000.pt` | 70.8% | [66.7, 74.6] | 1.55 | 27.2% |
| `04_qmix_ep2000_500ep.json` | QMIX | `qmix_ep2000.pt` | 65.0% | [60.7, 69.0] | 1.42 | 33.0% |
| `05_facmac_base_500ep.json` | Pure FACMAC | `facmac_base_ep2000.pt` | 82.0% | [78.4, 85.1] | 1.73 | 17.4% |
| `06_facmac_omr_500ep.json` | **FACMAC+OM+R** | `facmac_omr_ep27000.pt` | **84.2%** | [80.7, 87.1] | 1.78 | 15.2% |

**CI yöntemi:** Wilson score (Wilson, 1927), z=1.96 (%95 iki-kuyruklu).

**Eval koşulları:** 500 epizot, seed=42, deterministik çıkarım, curriculum Faz-4
(2v2 normal spawn), heuristic rakip (Red), `ENV_VERSION=1.0.0`,
opponent=`heuristic-v1`.

**Stale değerler — kullanılmamalı:** 52.8%, 69.6%, 62.4%, 70.2%.
GAT artışı **+23.4 pp** (69.8−46.4), OM+Role vs GAT farkı istatistiksel olarak
ayırt edilemez (bkz. §3 aşağı).

### Schema v1 yeniden üretim dosyaları (T2.2)

`v0.1/` altındaki dosyalar, unified runner (`aircombat_bench.eval.runner`) ile
yeniden üretilen sonuçları içerir.  Her dosya `schema_version`, `env_version`,
`git_commit` ve `checkpoint.sha256` içerir; per-episode kayıtlar tüm modeller
için mevcuttur.

Herhangi bir sayının yayımlanan CI dışına çıktığı tespit edilirse:
**dosyanın üzerine yazma; farklılığı raporla.**

Farkın nedenini varsayma. T2.5'te bu iki set arasındaki 1.4–3.8 pp
farklılıkların tamamı **cihaz farkından** (CPU vs CUDA) kaynaklandı; ortam,
protokol, seeding şeması ve politika yolu değişmemişti. İlk kontrol edilecek
şey her iki dosyanın `runtime.device` alanıdır (§0).

---

## 2. Çapraz Model Karşılaştırması (FACMAC vs MAPPO head-to-head)

| Dosya | Açıklama | Epizot |
|---|---|---|
| `07_facmac_vs_mappo_500ep.json` | FACMAC+OM+R vs MAPPO+GAT — her takım sırayla Blue ve Red | 500 (250+250) |

**Sonuçlar:** FACMAC win=68.0%, MAPPO win=17.2% (MAPPO Blue iken).
Format: `half_a_mappo_blue` (ilk 250) + `half_b_facmac_blue` (son 250) iç içe JSON.

---

## 3. Yapısal Hipotez Testi — Rol Sıfırlama Deneyi (200 ep, seed=42)

| Dosya | Model | Koşul | kill_imbalance | Win% |
|---|---|---|---|---|
| `08_structural_role_zeroed_facmac.json` | FACMAC+OM+R (ep27000) | normal | 0.47 | 81.5% |
| `08_structural_role_zeroed_facmac.json` | FACMAC+OM+R (ep27000) | zeroed | 0.06 | 83.5% |
| `09_structural_role_zeroed_mappo.json` | MAPPO+GAT+OM+Role (ep44000) | normal | 0.06 | 69.0% |
| `09_structural_role_zeroed_mappo.json` | MAPPO+GAT+OM+Role (ep44000) | zeroed | 0.03 | 72.0% |

**Script:** `evaluation/eval_structural_hypothesis.py`

### Sonuç yorumu

**kill_imbalance (yük taşıyan bulgu):**
- FACMAC: Rol sıfırlanınca kill asimetrisi yok oluyor (0.47 → 0.06) → rol davranışı
  **aktif olarak farklılaştırıyor** (güçlü kanıt, N=200).
- MAPPO: Fark yok (0.06 → 0.03) → rol **davranışsal olarak etkisiz** (güçlü kanıt).

**⚠️ T2.4 — İstatistiksel uyarı (win rate farkı):**  
Win rate değişimleri (FACMAC: 81.5%→83.5%, MAPPO: 69.0%→72.0%) istatistiksel
olarak ayırt **edilemez**.

- N=200 üzerinde 95% CI yaklaşık **±6.5 pp**'dir.
- Her iki fark da (+2.0 pp ve +3.0 pp) bu CI içindedir.
- Sadece tek bir run yapıldı; birden fazla seed ile tekrar edilmedi.

**Sonuç:** Win rate farkları bulgu olarak sunulmamalı.  Tek kayda değer bulgu,
kill_imbalance'daki asimetrik değişimdir.  Tez ve bildiride ifade şu olmalı:
> "Win rate değişimleri istatistiksel olarak ayırt edilemez (±6.5 pp, N=200);
> load-bearing kanıt kill asimetrisidir."

---

## Hangi Dosyayı Ne İçin Kullanmak Gerekir

| Tez Bölümü | Dosyalar |
|---|---|
| Model karşılaştırma tablosu (§4.x) | `01` – `06` (veya `v0.1/` eşdeğerleri) |
| FACMAC vs MAPPO cross-eval (§4.x) | `07` |
| Rol mekanizma analizi — yapısal hipotez (§3.5.3 / §4.4) | `08`, `09` |
| Intent / rol dağılım tablosu | `03`, `06` (intent_mean, role0/1_mean alanları) |

---

## Dosya Format Notları

### Orijinal dosyalar (bu dizin, `01`–`09`)

**MAPPO eval dosyaları** (`01`, `02`, `03`): `eval.py` çıktısı.
- `win_rate`, `loss_rate`, `draw_rate`, `kill_per_ep`, `mean_ep_len`, `survival_rate`
- OM modellerinde ek alanlar: `intent_mean`, `role0_mean`, `role1_mean`
- `episodes` dizisi (yalnızca `01`–`03`): her epizot için kayıt

**FACMAC eval dosyaları** (`05`, `06`): `eval_facmac.py` çıktısı.
- `win_rate`, `kill_per_ep`, `second_kill_rate`, `draw_rate`, `loss_rate`, `mean_ep_len`
- 95% CI: `win_ci95_lo`, `win_ci95_hi`

**QMIX eval dosyası** (`04`): `evaluation/eval_qmix.py` çıktısı.
- FACMAC formatıyla aynı.

**Yapısal test dosyaları** (`08`, `09`): `eval_structural_hypothesis.py` çıktısı.
- İç içe `normal` / `zeroed` blokları

### Schema v1 dosyaları (`v0.1/` dizini)

Unified runner (`aircombat_bench.eval.runner`) çıktısı.  Her dosya içerir:
- `schema_version: "1.0"`
- `env_version`, `package_version`, `git_commit`
- `checkpoint: {path, sha256}`
- `metrics`: `win_rate`, `win_ci95 [lo, hi]`, `loss_rate`, `draw_rate`,
  `kill_per_ep`, `kill_std`, `mean_ep_len`, `survival_rate`, `oob_rate`,
  `one_kill_stall_rate`
- `episodes`: tüm modeller için per-episode kayıtlar

Not: `second_kill_rate` schema v1'de kaldırıldı (`win_rate` ile özdeş olduğu için).
Yerine `one_kill_stall_rate` eklendi.

---

## QMIX Provenance Notu

Orijinal Mayıs 2026 eval'i 70.2% vermişti. Ağustos 2026'da aynı checkpoint ve
seed ile yeniden çalıştırıldığında **65.0%** elde edildi.  Fark, ortam
güncellemelerinden kaynaklanıyor (reward ağırlıkları, OOB recovery, spawn).

Diskteki `qmix_ep2000.pt` (total_steps=1,110,370) farklı bir eğitim oturumunda
üzerine yazılmış; Mayıs 2026 eval'inin 844k adımlı orijinal checkpoint'i artık
mevcut değil.

**Tezde kullanılması gereken tutarlı çift:** 65.0% + 1.11M adım.
