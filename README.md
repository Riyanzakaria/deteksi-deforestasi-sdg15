# Deteksi Aktivitas Deforestasi Hutan Tropis melalui Klasifikasi Multi-Label Citra Satelit Berbasis Deep Learning

Repository proyek Learning Problem mata kuliah Praktisi Data Science, Kelompok 10, Kelas 5B, Program Studi Teknologi Rekayasa Perangkat Lunak, Politeknik Negeri Madiun.

## Anggota kelompok

- Riyan Zakaria Zulkarnain (244311058)
- Andrian Maulana (244311036)

## Ringkasan

Deforestasi hutan tropis berjalan lebih cepat daripada kemampuan pemantauan manual. Proyek ini membangun model machine learning yang memberi label isi citra satelit hutan secara otomatis, termasuk aktivitas manusia penyebab deforestasi seperti tebang bakar, penebangan selektif, dan pertambangan, agar area bermasalah bisa lebih cepat diperiksa di lapangan.

## Keterkaitan dengan SDG

Proyek ini selaras dengan SDG 15 (Life on Land), target 15.2: pengelolaan hutan secara lestari, penghentian deforestasi, dan pemulihan hutan yang terdegradasi.

## Tugas machine learning

Klasifikasi multi-label citra satelit. Setiap citra berukuran 256 x 256 piksel dipetakan ke 17 label sekaligus: 4 label kondisi atmosfer, 6 label tutupan lahan umum, dan 7 label langka. Model dinilai dengan F2-score (beta = 2) yang menekankan recall, dengan target F2 minimal 0,87 pada data uji.

## Dataset

Planet: Understanding the Amazon from Space (Kaggle, 2017), berisi 40.479 citra latih berlabel berukuran 256 x 256 piksel dalam format JPG, ditambah sekitar 40.668 citra uji tanpa label.

Tautan dataset: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

## Rencana tiga model

1. Baseline: Logistic Regression dari fitur warna dan tekstur citra.
2. Advanced: Random Forest atau XGBoost dari fitur yang sama.
3. Deep learning: CNN ResNet50 dengan transfer learning, sigmoid per label, loss binary cross-entropy, dan threshold yang disetel per label.

Eksperimen dilatih di Google Colab atau Kaggle (GPU T4).

## Status

Checkpoint 1: pengajuan ide (Formulir Lampiran B dan proposal awal) sudah selesai disusun. Pengerjaan model menyusul pada minggu-minggu berikutnya.
