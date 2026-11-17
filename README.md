# MLP ile Sınıflandırma: EDA, Ön İşleme, Hiperparametre Optimizasyonu ve Nested Cross-Validation

Yapay Sinir Ağları dersi — Yüksek Lisans Ara Sınav Projesi.
UCI Bank Marketing (`bank-additional.csv`, 4.119 gözlem) üzerinde PyTorch MLP ile ikili sınıflandırma (`y = yes` → 1).

## Veri seti
- Kaynak: Moro, S., Rita, P. ve Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Ham veri repoya yüklenmez. `01_eda.ipynb` resmî zip'i (`https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`) indirir ve içinden **yalnızca** `bank-additional.csv` dosyasını `data/` klasörüne çıkarır.
- Elle indirirsen: zip içindeki `bank-additional/bank-additional.csv` dosyasını **Excel'de açmadan** `data/bank-additional.csv` olarak koy. Excel ondalık sayıları tarihe çevirebilir (ör. `1.4` → "1.Nis"). `validate_raw()` kontrolü bu durumda çalışmayı durdurur.
- `duration` değişkeni görüşme öncesinde bilinmediği için tüm modellerden çıkarılmıştır.

## Seed politikası
Temel seed = öğrenci numarasının son 5 hanesi + 2026 (`configs/project.json`). Alt seed'ler sabit ofsetlerle türetilir (`src/config.py → SEED_OFFSETS`) ve `configs/seeds.json` dosyasına yazılır:

| Amaç | Ofset | Amaç | Ofset |
|---|---|---|---|
| Python `random` | +0 | Random Search | +6 |
| NumPy | +1 | Early-stop alt bölünmesi | +7 |
| PyTorch (birincil ağırlık seed'i) | +2 | Bootstrap | +8 |
| Train/test bölünmesi | +3 | Logistic Regression | +9 |
| Outer CV | +4 | Kararlılık seed'leri | +2, +101, +202 |
| Inner CV | +5 | | |

## Çalıştırma sırası (Google Colab)
1. Bu repoyu GitHub'a yükle. Colab'da her notebook'un ilk hücresindeki `GITHUB_REPO_URL` alanına repo adresini yaz.
2. **`01_eda.ipynb`**: veri indirme ve doğrulama, kopya kontrolü, %80/%20 stratified bölünme (`results/dataset_split.csv`), development-only EDA ve hipotezler.
3. **`02_nested_cv.ipynb`**:
   - A bölümünden sonra `configs/random_candidates.json` dosyasını **commit et** (adayların önceden kaydı).
   - Nested CV, development üzerinde yeniden seçim, nihai eğitim ve etiketsiz `results/test_predictions_locked.csv`.
   - Bitince **ayrı bir commit** at: *"Lock final model and unlabeled test predictions"*.
4. **`03_final_test.ipynb`**: kilitli dosyanın SHA-256 doğrulaması, test metrikleri, bootstrap %95 GA, ROC/PR, confusion matrix, seed kararlılığı, hata analizi.

Colab'da her notebook'un son hücresi `results/`, `figures/` ve `configs/` klasörlerini zip olarak indirir. Bunları repoya yükle.
Tüm eğitimler CPU'da yapılır (determinizm için). Bağlantı koparsa `02` yeniden çalıştırıldığında `results/cache/` sayesinde kaldığı yerden devam eder.

## Repo yapısı
| Yol | İçerik |
|---|---|
| `src/config.py` | Yollar, veri sözlüğü, seed türetme, ortam bilgisi |
| `src/data.py` | İndirme, doğrulama, `pdays=999` kuralı, kopya grupları, stratified (grup-farkında) bölünmeler |
| `src/preprocessing.py` | Kat-içi `ColumnTransformer`: median imputation, yüzdelik clipping, Standard/MinMax/Robust scaler, OneHotEncoder(`handle_unknown="ignore"`) |
| `src/model.py` | PyTorch MLP (tek logit + `BCEWithLogitsLoss`), early stopping (patience 15, best-weights restore), `ReduceLROnPlateau` |
| `src/search.py` | M1–M6, Random Search R1–R6, inner/outer CV, referans modeller, nihai seçim ve kilitleme |
| `src/metrics.py` | AP, ROC-AUC, Sensitivity, Specificity, F1, Balanced Accuracy, Mean ± SD, bootstrap GA |
| `src/plots.py` | Rapordaki tüm şekiller |
| `configs/` | `project.json`, `training.json`, `search_space.json`, `manual_candidates.json`, (üretilen) `random_candidates.json`, `seeds.json` |
| `results/` | Bölünme, deney, kat, tahmin ve nihai metrik dosyaları (CSV/JSON), modeller |
| `figures/` | Üretilen grafikler |
| `report.pdf` | Bilimsel teknik rapor |

## Ana çıktı dosyaları
`dataset_split.csv`, `manual_search_results.csv`, `random_search_results.csv`, `inner_cv_fold_results.csv`, `outer_cv_results.csv`, `outer_cv_summary.csv`, `dev_inner_cv_results.csv`, `final_hyperparameters.json`, `test_predictions_locked.csv`, `test_metrics.csv`, `test_bootstrap_ci.csv`, `seed_stability_results.csv`, `environment.json`, `requirements-lock.txt`.

## Sistem ve kütüphane sürümleri
Çalıştırma sonrasında `results/environment.json` (Python, işletim sistemi, CPU, kütüphane sürümleri) ve `results/requirements-lock.txt` otomatik oluşturulur.

## Not: kör test
Veri seti kamuya açık olduğu için test etiketleri teknik olarak erişilebilirdir. Etiketsiz tahminlerin ayrı commit'te saklanması kör test garantisi değildir; zaman sıralı bir deney kaydıdır.
