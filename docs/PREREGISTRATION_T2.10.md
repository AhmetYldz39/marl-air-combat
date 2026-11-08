# Ön-kayıt — T2.10 sıfır-vektör confound testi

**Tarih:** 2026-09-13
**Durum:** bu dosya **koşular başlatılmadan önce** commit'lenmiştir.
Tahminler sonuçlara bakılarak yazılmamıştır; git geçmişi bunu doğrular.

---

## 1. Test edilen confound

T2.7 ve T2.9'daki C1 koşulu rol vektörünü **sıfırlar**. Ancak atayıcı
(`CentralizedRoleAssigner.assign()`, [om_net.py:216](../aircombat_bench/models/om_net.py#L216))
`F.one_hot(...)` döndürür — yani aktör eğitim boyunca **daima L2 normu 1.0 olan
bir one-hot vektör** görmüştür. Sıfır vektörü hiç görmemiştir.

Dolayısıyla C1 rol bilgisini temiz biçimde kaldırmaz; ağa **eğitim dağılımı
dışı (OOD)** bir girdi verir. Ölçülen −2.25 pp düşüş, bilgi kaldırmadan değil
dağılım kaymasından kaynaklanabilir.

Bu, T2.9'un ötesini etkiler: T2.7'nin "rol vektörü taşıyıcıdır, −2.25 pp,
p=0.014" sonucu **aynı müdahaleye** dayanır ve aynı confound'u devralır.

## 2. Koşullar

Mevcut `facmac_omr_ep27000.pt` üzerinde, çıkarım anında, eğitim yok.
n=2000, seed 42, CPU tek thread, schema v1.

| Koşul | Aktöre verilen rol vektörü | L2 norm | Biçim |
|---|---|---|---|
| C0 (var) | atayıcı çıktısı, one-hot | 1.0 | eğitim-içi |
| C1 (var) | sıfır | 0.0 | **OOD** |
| C3 (var) | sabit one-hot, ikisine de pursuit | 1.0 | eğitim-içi biçim |
| **C5** | **uniform, ikisine de [0.25, 0.25, 0.25, 0.25]** | **0.5** | OOD biçim, sıfır değil |
| **C6** | C3 vektörü × ε, ε ∈ {0.5, 0.1, 0.01} | ε | eğitim-içi biçim, OOD büyüklük |

**C5 belirleyici olan.** C1 kadar az rol bilgisi taşır — rol tercihi yok, ajan
ayrımı yok — ama sıfır değil ve olağan büyüklükte.

**Planlanmamış bir ek kontrast (tasarımdan doğal olarak çıkıyor, kayda
geçiriyorum):** ‖uniform‖₂ = √(4·0.25²) = 0.5 ve ‖0.5 · one-hot‖₂ = 0.5.
Yani **C5 ile C6@ε=0.5 aynı L2 normuna sahip, yalnızca biçimleri farklı**
(yayılmış vs yoğunlaşmış). Bu çift, büyüklüğü sabit tutarak biçimi izole eder.

## 3. Ön-kayıtlı yorumlar

### C5 için

- **C5 ≈ C3 ≈ C0 (≈%86)** → rol vektörünün *içeriği* kazanma oranı için
  önemsizdir. C1'in ölçtüğü şey, yolun sıfırlanması — yani OOD bir girdi.
  Deflasyonist okuma **doğrulanır** ve **T2.7 ablasyon sonucu geri
  çekilmelidir** (yumuşatılmaz).
- **C5 ≈ C1 (≈%83.8)** → büyüklük tek başına performansı kurtarmıyor;
  *geçerli* bir rol vektörüne ilişkin bir şey önemli. Deflasyonist okuma
  **başarısız olur** ve C3'ün sonucu başka bir açıklama gerektirir.
  → Bu durumda **escalate**, hiçbir şey yazılmadan önce.
- **C5, C1 ile C3 arasında ve ikisinden de anlamlı biçimde farklı** → hem
  büyüklük hem içerik kısmen katkı verir; tek bir mekanizma yeterli değil.

### C6 için

- **ε küçüldükçe kazanma oranı C3'ten C1'e doğru düzgün biçimde bozulur** →
  etki yolun katkı büyüklüğüyle ölçekleniyor; bu bir **dağılım kayması
  imzasıdır**, bilgi imzası değil.
- **ε çok küçülene kadar düz kalır** → yol bir **anahtar** gibi davranıyor.
- **Düzensiz / monoton olmayan** → şemaya oturmuyor, olduğu gibi raporlanır.

### C5 ↔ C6@0.5 (L2-eşleşmiş çift)

- **Fark yok** → biçim önemsiz, yalnızca büyüklük sayılıyor.
- **Anlamlı fark** → biçim de sayılıyor; one-hot yapısı kendi başına bir şey
  taşıyor.

## 4. Raporlama taahhüdü

Ne bulunursa raporlanır — yukarıdaki kalıplardan hiçbirine uymayan bir örüntü
dâhil. `compare.py` ile eşleştirilmiş episode'lar üzerinde C0, C1 ve C3'e karşı:
Wilson aralığı, paired bootstrap CI ile eşleştirilmiş fark, McNemar, rol
asimetrisi, iki kill-imbalance tanımı.

**Aile:** bu seri **keşifsel (exploratory)** olarak raporlanır. Ham p-değerleri
ve CI'lar verilir; doğrulayıcı (confirmatory) aile ile rekabet eden ikinci bir
düzeltilmiş aile oluşturulmaz (T2.10 §3).
