# Environment Changelog

`aircombat_bench.envs.dogfight_env.DogfightEnv` sürüm geçmişi.

Sürümleme semver: **MAJOR** = yörüngeyi değiştiren her şey (baseline'lar
yeniden eğitilmeli ve yeniden değerlendirilmeli), **MINOR** = yörüngeyi
değiştirmeyen davranış eklentisi, **PATCH** = yörüngeye dokunmayan düzeltme.

Bir yörünge değişikliği tüm kanonik tabloyu geçersiz kılar — bu yüzden MAJOR
değişiklikler biriktirilip tek seferde uygulanır.

---

## 1.0.0 — mevcut (2026-09)

İlk dondurulmuş sürüm. Kanonik tablo (`docs/evals/v1.0/`, n=2000, seed=42,
CPU tek thread) bu sürüme karşı üretilmiştir.

---

## 2.0.0 — planlanan (kırıcı değişiklik)

Aşağıdaki maddeler birlikte uygulanacak. Hiçbiri tez teslimi öncesinde
yapılmayacak: her biri yörüngeyi değiştirir, `ENV_VERSION` bump'ı ve
baseline'ların yeniden eğitilip yeniden değerlendirilmesini gerektirir.

### 2.0.0-1 — Yan-bağımsız (side-agnostic) kurtarma

**Durum:** planlandı, T2.7.3 kapsamında belgelendi. **Uygulanmadı.**

**Sorun.** `_critical_recovery_blue` yalnızca `blue_ids`'e uygulanıyor
([dogfight_env.py:308](aircombat_bench/envs/dogfight_env.py#L308)):

```python
if aid in self.blue_ids:
    recovery = self._critical_recovery_blue(state)
```

Scripted red kendi kurtarmasına sahip (`HeuristicAgent._critical_recovery`),
ama **checkpoint sürülen red hiçbir kurtarma almaz**. Ortam bu yüzden
yan-simetrik değildir.

**Ölçülen etki** (kanonik n=2000, `docs/evals/v1.0/07a`, `07b`):

| Model | blue iken | red iken | yan etkisi |
|---|---|---|---|
| FACMAC+OM+R | 73.30% | 69.80% | +3.50 pp |
| MAPPO+GAT | 16.55% | 13.35% | +3.20 pp |

**Kapsam.** Yalnızca agent-vs-agent değerlendirmeleri (`07a`/`07b` ve
gelecekteki self-play / checkpoint-vs-checkpoint koşuları). `01`–`06`
etkilenmez: orada red her zaman scripted rakiptir ve altı yöntemin hepsi
birebir aynı asimetriyle karşılaşır.

**Planlanan değişiklik.** Kurtarmayı takımdan bağımsız hâle getirmek: her
öğrenen ajana — hangi takımda olursa olsun — aynı kurtarma uygulanmalı.
Karar verilmesi gereken ikincil nokta: env kurtarması `HeuristicAgent`'ınkiyle
eşitlenecek mi (stall ve düşük hız dalları eklenecek mi), yoksa iki yol ayrı
mı kalacak.

**Maliyet.** Yörünge değişir → `ENV_VERSION 2.0.0`, altı baseline yeniden
eğitilir, kanonik tablo yeniden üretilir.

### Not — harita-kenarı kurtarma dalı: ÖLÜ KOD, hata değil (T2.8.3)

**Bu bir v2.0.0 maddesi DEĞİLDİR.** Ölçümle sınıflandırıldı; düzeltme
gerektirmiyor, yalnızca kayda geçiyor.

Hem `_critical_recovery_blue` hem `HeuristicAgent._critical_recovery`
harita-kenarı dalı içerir ve hiçbiri tetiklenmez. Sebep, koşulun yanlış
olması değil — uçaklar marja hiç yaklaşmıyor.

Ölçüm (300 kanonik episode, 188.549 blue + 170.270 red ajan-adımı;
`logs/map_edge_probe.json`). Arena `map_size = 50.000 m`, yarı-genişlik
25.000 m. Sınıra mesafe `half − max(|x|, |y|)` — dalın kullandığı ölçünün
birebir aynısı:

| | en yakın | p1 | p5 | medyan |
|---|---|---|---|---|
| blue | 14.036 m | 17.733 m | 19.059 m | 21.784 m |
| red | 9.194 m | 10.018 m | 11.027 m | 15.056 m |

| Eşik | marj | marj içinde geçen adım |
|---|---|---|
| blue (env) | 5.000 m | **0** (%0,0000) |
| red (heuristic) | 3.000 m | **0** (%0,0000) |

Blue'nun sınıra en yakın yaklaşması marjın **2,8×**, red'inki **3,1×**
uzağındadır. Uçaklar ±25 km'lik arenada merkezden ±11 km'yi aşmıyor.

**Ayrıca gerçek OOB de olmuyor:** ölçümde sınır aşımı 0 adım ve altı kanonik
modelin hepsinde `metrics.oob_rate = 0.0`. Kanonik değerlendirme rejiminde
(Faz-4 2v2, `heuristic-v1`) sınır işleme alt sisteminin **tamamı atıldır.**

Bu, sistemin gereksiz olduğu anlamına gelmez: yüksek OOB oranları self-play
*eğitiminde* görülür (heuristic-only ≈ 0,07; sertleşmiş havuza karşı
0,34–0,42). Alt sistem o rejim için yazılmıştır ve orada canlıdır. Kanonik
eval'de ölçülen etkisi sıfırdır.

### 2.0.0-2 — Kurtarma kapsamı eşitlemesi (opsiyonel, 2.0.0-1 ile birlikte)

İki kurtarma yolu farklı kapsamdadır (ölçüm: T2.7.3, 150 ep):

| Dal | env (blue) | heuristic (scripted red) |
|---|---|---|
| zemin `h < 300 m` | var — %7.886 tetiklenme | var — %1.457 |
| stall `\|α\| > 22°` | **yok** (gerekseydi %5.011) | var — %20.651 |
| düşük hız `v < 80 m/s` | **yok** (gerekseydi %8.181) | var — %23.261 |
| harita kenarı | var, 5000 m marj — %0.000 | var, 3000 m marj — %0.000 |
| **toplam** | **%7.886** | **%45.369** |

Harita kenarı dalı Faz-4 spawn'ında hiçbir tarafta tetiklenmiyor; 5000/3000 m
marj farkı pratikte ölü harftir.

2.0.0'da bu kapsam farkının kapatılıp kapatılmayacağına karar verilmeli.
