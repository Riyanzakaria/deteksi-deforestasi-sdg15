# Deteksi Aktivitas Deforestasi Hutan Tropis melalui Klasifikasi Multi-Label Citra Satelit Berbasis Deep Learning

Repository proyek Learning Problem mata kuliah Praktisi Data Science. Proyek ini membangun model pembelajaran mesin yang membaca citra satelit hutan tropis Amazon dan menebak 17 label sekaligus pada setiap citra, termasuk aktivitas manusia penyebab deforestasi seperti tebang bakar, penebangan selektif, dan pertambangan, supaya area bermasalah bisa lebih cepat ditemukan dan diperiksa di lapangan.

| Keterangan | Isi |
| --- | --- |
| Kelompok | Kelompok 10 |
| Anggota | Riyan Zakaria Zulkarnain (244311058) dan Andrian Maulana (244311036) |
| Kelas | 5B |
| Program studi | Teknologi Rekayasa Perangkat Lunak |
| Kampus | Politeknik Negeri Madiun |
| Jenis tugas | Klasifikasi multi-label citra satelit (supervised learning) |
| Metrik utama | F2-score (beta = 2), menekankan recall |
| Target | F2 minimal 0,87 pada data uji |

<a id="isi"></a>
**Isi:** [Tentang Proyek](#tentang-proyek) · [Keterkaitan dengan SDG](#keterkaitan-dengan-sdg) · [Dataset](#dataset) · [Rancangan Tiga Model](#rancangan-tiga-model) · [Kemajuan](#kemajuan) · [Struktur Repository](#struktur-repository) · [Menjalankan Ulang Persiapan Data](#menjalankan-ulang-persiapan-data) · [Referensi](#referensi)

<a id="tentang-proyek"></a>
## Tentang Proyek

Hutan tropis terus menyusut, sedangkan pemantauannya masih banyak dikerjakan secara manual. Citra satelit tersedia dalam jumlah besar, tetapi memberi label tutupan lahan dan aktivitas manusia di dalamnya satu per satu memakan waktu lama. Proyek ini mencoba cara otomatis untuk membaca citra tersebut: satu citra diberi 17 label sekaligus karena satu wilayah bisa memuat banyak hal dalam waktu yang sama, misalnya hutan primer yang berawan sebagian dan dilalui jalan. Hasil akhirnya adalah perbandingan tiga model dengan tingkat kerumitan berbeda pada data uji yang sama, supaya terlihat seberapa besar tambahan manfaat model yang lebih rumit untuk persoalan ini.

<a id="keterkaitan-dengan-sdg"></a>
## Keterkaitan dengan SDG

| Aspek | Isi |
| --- | --- |
| SDG | SDG 15 (Life on Land) |
| Target | 15.2 |
| Bunyi target | Pengelolaan hutan secara lestari, penghentian deforestasi, dan pemulihan hutan yang terdegradasi |
| Relevansi proyek | Model yang mampu menandai citra berisi aktivitas perusak hutan membantu pemantauan kawasan hutan dalam skala besar, yang merupakan langkah awal pengelolaan dan perlindungan hutan pada target tersebut |

<a id="dataset"></a>
## Dataset

| Keterangan | Isi |
| --- | --- |
| Nama | Planet: Understanding the Amazon from Space |
| Sumber | Kompetisi Kaggle tahun 2017 |
| Tautan | https://www.kaggle.com/c/planet-understanding-the-amazon-from-space |
| Citra latih berlabel | 40.479 citra |
| Ukuran citra | 256 x 256 piksel, format JPG |
| Berkas yang dipakai | train-jpg.tar.7z (629.296.207 byte) dan train_v2.csv (40.479 baris label) |
| Rata-rata label | 2,87 label per citra (paling sedikit 1, paling banyak 9) |

Ke-17 label terbagi menjadi tiga kelompok:

| Kelompok label | Label |
| --- | --- |
| Kondisi atmosfer (tepat satu pada tiap citra) | clear, partly_cloudy, haze, cloudy |
| Tutupan lahan umum | primary, water, habitation, agriculture, cultivation, road |
| Label langka terkait aktivitas manusia | bare_ground, blooming, blow_down, slash_burn, selective_logging, artisinal_mine, conventional_mine |

| Statistik kunci | Nilai |
| --- | --- |
| Label terbanyak | primary, 37.513 citra (92,68 persen) |
| Label terjarang | blow_down, 98 citra (0,24 persen) |
| Ketimpangan | Sekitar 383 banding 1 antara label terbanyak dan terjarang |

<a id="rancangan-tiga-model"></a>
## Rancangan Tiga Model

| Model | Masukan | Cara kerja | Threshold |
| --- | --- | --- | --- |
| Baseline: Logistic Regression | Fitur warna dan tekstur citra | Satu pengklasifikasi biner untuk tiap label | Tetap 0,5 |
| Advanced: Random Forest atau XGBoost | Fitur warna dan tekstur yang sama | Hutan keputusan atau boosting untuk tiap label | Tetap 0,5 |
| Deep learning: ResNet50 | Citra asli 256 x 256 piksel | Transfer learning, sigmoid pada tiap label, loss binary cross-entropy | Disetel per label pada data validasi |

Protokol evaluasi yang sudah dikunci:

| Aspek | Ketentuan |
| --- | --- |
| Metrik utama | F2-score micro pada data uji, sama untuk ketiga model |
| Metrik tambahan | F2 dan recall tiap label, dengan sorotan pada label penyebab deforestasi (slash_burn, selective_logging, artisinal_mine, conventional_mine) |
| Data penyetelan | Data validasi, tidak pernah memakai data uji |
| Target keberhasilan | F2 minimal 0,87 pada data uji |

<a id="kemajuan"></a>
## Kemajuan

| Pekerjaan | Status | Hasil atau keterangan |
| --- | --- | --- |
| Pengajuan ide dan proposal | Selesai | Formulir pengajuan, Lampiran B, dan proposal terkumpul pada 6 November 2026. Repository ini dibuat dan terbuka untuk publik |
| Pengunduhan dataset | Selesai | Arsip citra dan berkas label terunduh utuh dan ukuran berkasnya terverifikasi sama dengan sumbernya |
| Eksplorasi data | Selesai | Label terbanyak primary 37.513 citra (92,68 persen), terjarang blow_down 98 citra (0,24 persen). Ditemukan 73 baris yang token labelnya tertulis ganda, sehingga penghitungan memakai kehadiran label per citra |
| Pemeriksaan dan pembersihan data | Selesai | Seluruh 40.479 citra dibuka dan diperiksa satu per satu: tidak ada citra rusak, tidak ada berkas hilang, tidak ada duplikat. Hanya 1 baris tanpa label cuaca (train_24448) yang dibuang |
| Pembagian data | Selesai | Data latih 28.335, validasi 6.071, uji 6.072. Pembagian menjaga proporsi tiap label (deviasi terbesar 0,0014) dan ketiganya saling lepas |
| Protokol evaluasi | Selesai | F2-score micro pada data uji untuk semua model, F2 dan recall per label, threshold 0,5 untuk dua model pertama dan threshold per label untuk model deep learning |
| Penyimpanan hasil di repository | Selesai | Notebook eksplorasi yang sudah dijalankan, skrip persiapan data, data bersih, berkas pembagian, dan log pembersihan tersimpan di repository ini |
| Pelatihan tiga model | Berikutnya | Dimulai dari model baseline, lalu model advanced, terakhir model deep learning. Hasil ketiganya dibandingkan pada data uji yang sama, dilengkapi analisis kesalahan, laporan akhir, dan presentasi |

Hasil persiapan data dalam angka:

| Keterangan | Nilai |
| --- | --- |
| Data awal | 40.479 citra |
| Data bersih | 40.478 citra |
| Baris dibuang | 1 (train_24448, tanpa label cuaca) |
| Data latih | 28.335 citra |
| Data validasi | 6.071 citra |
| Data uji | 6.072 citra |

Rincian lengkap persiapan data tersedia pada laporan perkembangan di [results/LAPORAN-PERKEMBANGAN.md](results/LAPORAN-PERKEMBANGAN.md).

<a id="struktur-repository"></a>
## Struktur Repository

```
deteksi-deforestasi-sdg15/
├── README.md
├── data/
│   ├── raw/          # arsip dataset asli dari Kaggle (tidak diubah)
│   └── processed/    # data bersih, berkas pembagian, distribusi label, log pembersihan
├── notebooks/
│   └── 01_eda_dan_persiapan_data.ipynb   # eksplorasi data yang sudah dijalankan
├── scripts/
│   └── persiapan_data.py                 # pipeline ekstrak, periksa, bersihkan, bagi data
└── results/
    └── LAPORAN-PERKEMBANGAN.md           # laporan perkembangan proyek
```

<a id="menjalankan-ulang-persiapan-data"></a>
## Menjalankan Ulang Persiapan Data

Seluruh pekerjaan persiapan data bisa dicek dan dijalankan ulang dari folder proyek ini. Skripnya dibuat idempoten dengan seed terkunci, jadi hasil setiap larian sama.

```bash
python -m venv venv
venv\Scripts\activate        # Windows; di Linux/macOS: source venv/bin/activate
pip install pandas numpy Pillow scikit-learn py7zr jupyter
python scripts/persiapan_data.py
```

Keluaran skrip: data bersih dan berkas pembagian di `data/processed/`, beserta `cleaning_log.json` yang mencatat setiap tindakan pembersihan. Notebook `notebooks/01_eda_dan_persiapan_data.ipynb` menyajikan tahapan yang sama secara visual, lengkap dengan grafik distribusi label dan contoh citra.

<a id="referensi"></a>
## Referensi

1. Ji, J., Jing, W., Chen, G., Lin, J., & Song, H. (2020). Multi-label remote sensing image classification with latent semantic dependencies. *Remote Sensing*, *12*(7), 1110. https://doi.org/10.3390/rs12071110
2. Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, *342*(6160), 850–853. https://doi.org/10.1126/science.1244693
