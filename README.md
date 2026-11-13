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
**Isi:** [Tentang Proyek](#tentang-proyek) · [Learning Problem](#learning-problem) · [Keterkaitan dengan SDG](#keterkaitan-dengan-sdg) · [Dataset](#dataset) · [Rancangan Tiga Model](#rancangan-tiga-model) · [Risiko dan Pertimbangan Etika](#risiko-dan-pertimbangan-etika) · [Kemajuan](#kemajuan) · [Struktur Repository](#struktur-repository) · [Menjalankan Ulang Persiapan Data](#menjalankan-ulang-persiapan-data) · [Referensi](#referensi)

<a id="tentang-proyek"></a>
## Tentang Proyek

Hutan tropis terus menyusut, sedangkan pemantauannya masih banyak dikerjakan secara manual. Citra satelit tersedia dalam jumlah besar, tetapi memberi label tutupan lahan dan aktivitas manusia di dalamnya, seperti tebang bakar, penebangan selektif, dan pertambangan, lama bila dikerjakan orang satu per satu. Hansen dkk. (2013) mencatat 2,3 juta km² hutan hilang di seluruh dunia pada 2000 sampai 2012; pada periode yang sama deforestasi di Brasil menurun, tetapi kehilangan hutan di Indonesia dan beberapa negara tropis lain justru naik, sehingga cara otomatis membaca citra satelit layak dicoba untuk konteks Indonesia.

Proyek ini mencoba cara otomatis tersebut: satu citra diberi 17 label sekaligus karena satu wilayah bisa memuat banyak hal dalam waktu yang sama, misalnya hutan primer yang berawan sebagian dan dilalui jalan. Keberhasilan model diukur dengan F2-score yang memakai beta = 2, sehingga recall berbobot empat kali lipat dibanding precision; perhatian utama diberikan pada recall label penyebab deforestasi karena label-label itu tidak boleh terlewat. Hasil akhirnya adalah perbandingan tiga model dengan tingkat kerumitan berbeda pada data uji yang sama, supaya terlihat seberapa besar tambahan manfaat model yang lebih rumit untuk persoalan ini.

Isi bagian ini dan bagian lain README ini diselaraskan dengan formulir pengajuan ide (Lampiran B) yang terkumpul pada 6 November 2026. Kriteria pengajuan yang terpenuhi:

- Ide berasal dari mahasiswa dan tidak duplikat
- Selaras jelas dengan satu SDG dan targetnya
- Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- Dataset tersedia, legal dan etis digunakan, dan cukup besar
- Layak untuk tiga model termasuk deep learning dalam 6 minggu

<a id="learning-problem"></a>
## Learning Problem

Rumusan persoalan pembelajaran mesin proyek ini diringkas dalam tabel berikut.

| Komponen | Isian |
| --- | --- |
| Target | 17 label multi-label pada tiap citra: 4 label kondisi atmosfer, 6 label tutupan lahan umum, dan 7 label langka terkait aktivitas penyebab deforestasi; keluaran model berupa vektor biner 17 dimensi |
| Objective | F2-score (beta = 2) setinggi mungkin pada data uji, dengan penekanan pada recall label penyebab deforestasi; target nilai minimal 0,87 dan recall label langka di atas hasil baseline |
| Data | Citra satelit Planet: Understanding the Amazon from Space (Kaggle, 2017): 40.479 citra latih berukuran 256 x 256 piksel dengan 17 label; jenis data citra; akses legal dengan bergabung ke kompetisi secara gratis |

<a id="keterkaitan-dengan-sdg"></a>
## Keterkaitan dengan SDG

| Aspek | Isi |
| --- | --- |
| SDG | SDG 15 (Life on Land) |
| Target | 15.2 |
| Bunyi target | Pengelolaan hutan secara lestari, penghentian deforestasi, dan pemulihan hutan yang terdegradasi |
| Relevansi proyek | Model yang mampu menandai citra berisi aktivitas perusak hutan membantu pemantauan kawasan hutan dalam skala besar, yang merupakan langkah awal pengelolaan dan perlindungan hutan pada target tersebut |

Pemantauan adalah pintu masuknya: sebelum hutan bisa dikelola dan dilindungi, perubahan di dalamnya harus terlihat lebih dulu. Di situlah proyek ini berkontribusi pada target 15.2, yaitu menyediakan cara membaca citra satelit kawasan hutan secara otomatis dan dalam jumlah besar, sehingga tanda-tanda deforestasi bisa ditemukan lebih awal.

<a id="dataset"></a>
## Dataset

| Keterangan | Isi |
| --- | --- |
| Nama | Planet: Understanding the Amazon from Space |
| Sumber | Kompetisi Kaggle tahun 2017 |
| Tautan | https://www.kaggle.com/c/planet-understanding-the-amazon-from-space |
| Citra latih berlabel | 40.479 citra |
| Citra uji tanpa label | Sekitar 40.668 citra (tidak dipakai pada proyek ini) |
| Ukuran citra | 256 x 256 piksel, format JPG 3 kanal (tersedia juga format TIF 4 kanal) |
| Akses | Bergabung ke kompetisi di Kaggle (gratis) |
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

Dataset ini kami pakai karena labelnya memang dirancang untuk membaca isi hutan dari citra satelit, dan persoalannya nyata: satu citra bisa berisi hutan primer, sungai, dan lahan garapan sekaligus, jadi tugasnya multi-label dan bukan klasifikasi satu label biasa. Tantangan utamanya juga terlihat sejak awal dari angka di atas, yaitu ketimpangan label yang ekstrem, dan tantangan itulah yang menentukan pilihan metrik serta rancangan model kami.

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

Ketiga model sengaja dibandingkan secara berjenjang. Baseline yang sederhana menjadi patokan awal, model advanced menunjukkan seberapa jauh fitur buatan tangan bisa didorong, dan deep learning membaca citranya langsung tanpa fitur buatan. Semuanya diuji pada data uji yang sama dengan protokol yang sama, jadi selisih hasilnya berasal dari modelnya, bukan dari cara mengujinya.

<a id="risiko-dan-pertimbangan-etika"></a>
## Risiko dan Pertimbangan Etika

Empat hal berikut kami sadari sejak pengajuan ide dan kami tulis terbuka, karena keempatnya memengaruhi cara hasil proyek ini harus dibaca.

| Risiko | Catatan dan penanganan |
| --- | --- |
| Label sangat timpang | Label primary muncul pada lebih dari 90 persen citra, sedangkan beberapa label langka kurang dari 1 persen. Ditangani dengan metrik F2, pembobotan loss, dan penyetelan threshold per label |
| Bias geografis data | Data latih hanya dari hutan Amazon di Brasil, sehingga model belum tentu bekerja sebaik itu pada hutan Indonesia, misalnya Kalimantan dan Papua. Keterbatasan ini ditulis apa adanya sebagai batas penelitian, bukan disembunyikan |
| Privasi | Citra satelit beresolusi 3,7 m tidak memuat data pribadi, jadi risiko privasi kecil |
| Kebisingan label | Label dibuat manusia melalui CrowdFlower, jadi masih mungkin ada label yang keliru |

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

Posisi proyek saat ini: seluruh pekerjaan persiapan sudah tuntas dan hasilnya bisa diperiksa ulang di repository ini, dari data mentah sampai berkas pembagiannya. Yang tersisa adalah pekerjaan pemodelan dan pelaporan akhir, yang urutannya sudah kami tetapkan: melatih baseline lebih dulu sebagai patokan, lalu model advanced dan deep learning, membandingkan ketiganya, dan menutupnya dengan laporan akhir serta presentasi.

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

Dua referensi berikut menjadi pijakan ilmiah proyek ini sejak pengajuan ide.

1. Ji, J., Jing, W., Chen, G., Lin, J., & Song, H. (2020). Multi-label remote sensing image classification with latent semantic dependencies. *Remote Sensing*, *12*(7), 1110. https://doi.org/10.3390/rs12071110 — memakai dataset yang sama (Planet) dengan DenseNet121-BC, attention, dan LSTM, serta loss berbobot untuk label langka.
2. Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, *342*(6160), 850–853. https://doi.org/10.1126/science.1244693 — menunjukkan skala kehilangan hutan global dari citra satelit.
