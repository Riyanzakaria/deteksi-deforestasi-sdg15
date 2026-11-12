# Laporan Fase 2: Persiapan Data

Proyek Learning Problem, Kelompok 10 (Riyan Zakaria Zulkarnain 244311058, Andrian Maulana 244311036)
Judul: Deteksi Aktivitas Deforestasi Hutan Tropis melalui Klasifikasi Multi-Label Citra Satelit Berbasis Deep Learning
Dataset: Planet: Understanding the Amazon from Space (Kaggle, 2017)
Tanggal eksekusi: 12 November 2026

## 1. Ringkasan fase

Fase 2 bertujuan menyiapkan dataset Planet sampai bersih dan siap dilatih, sebelum masuk ke pembangunan model pada Fase 3. Dataset yang dipakai adalah data latih kompetisi Planet: Understanding the Amazon from Space di Kaggle, yang berisi citra satelit wilayah hutan Amazon beserta label multi-labelnya. Keadaan awal data adalah 40.479 baris label pada berkas train_v2.csv dan 40.479 berkas citra JPG. Seluruh citra dan label diperiksa satu per satu, mencakup dekode citra untuk mendeteksi berkas rusak serta pemeriksaan aturan label yang berlaku pada dataset ini. Hasil pemeriksaan menunjukkan data bawaan kompetisi sudah sangat bersih, karena dari semua kategori pemeriksaan hanya ditemukan satu baris bermasalah, yaitu train_24448 yang tidak memiliki label cuaca. Baris tersebut dibuang, sehingga data bersih akhir berjumlah 40.478 citra. Data bersih kemudian dibagi menjadi data latih, validasi, dan uji dengan proporsi 70 persen, 15 persen, dan 15 persen. Dengan keadaan ini, dataset dinyatakan siap dipakai untuk pembangunan model baseline pada Fase 3.

## 2. Langkah eksekusi

1. Pengunduhan dataset dari halaman Data kompetisi Kaggle. Pemilik akun masuk ke Kaggle memakai akun Google, kemudian dua berkas dataset diunduh memakai alat unduh browser karena unduhan dengan klik biasa tidak tersimpan ke disk mesin.
   Hasil: berkas train-jpg.tar.7z berukuran 629.296.207 byte dan berkas train_v2.csv.zip berukuran 154.618 byte tersimpan di folder data/raw. Berkas label di dalam zip adalah train_v2.csv dengan 40.479 baris data.
2. Ekstraksi arsip citra ke folder data/raw/train-jpg. Ekstraksi berjalan dua tahap, tahap pertama membuka arsip .7z dan tahap kedua membuka arsip .tar di dalamnya.
   Hasil: 40.479 berkas JPG terekstrak, sama dengan jumlah baris pada berkas label.
3. Pembacaan train_v2.csv dan pembangunan matriks label biner 17 label.
   Hasil: 40.479 baris terbaca dan setiap baris terpetakan ke 17 kolom label biner.
4. Eksplorasi data awal (EDA): distribusi tiap label, jumlah label per citra, ko-okurensi label cuaca, dan contoh citra per label.
   Hasil: lihat Bagian 3.
5. Pemeriksaan integritas: setiap berkas JPG dibuka dan didekode untuk mendeteksi citra rusak.
   Hasil: 0 citra rusak dari 40.479 citra yang diperiksa.
6. Pemeriksaan aturan label: label cuaca ganda, tanpa label cuaca, citra cloudy yang membawa label lain, duplikat nama berkas, dan ketidaksesuaian CSV dengan berkas di disk.
   Hasil: lihat Bagian 4.
7. Cleaning mengikuti kebijakan tetap yang dicatat di Bagian 4.
   Hasil: data/processed/train_v2_clean.csv berisi 40.478 baris bersih dan seluruh catatan pemeriksaan tersimpan di data/processed/cleaning_log.json.
8. Pembagian data latih, validasi, dan uji dengan rasio 70 persen, 15 persen, dan 15 persen, seed 42, memakai stratifikasi iteratif multi-label.
   Hasil: lihat Bagian 5.
9. Penguncian protokol evaluasi untuk Fase 3 sampai 5.
   Hasil: lihat Bagian 6.
10. Penyalinan seluruh hasil ke repository GitHub dan laptop waipu.
    Hasil: penyalinan berkas hasil Fase 2 ke repository GitHub proyek dan ke laptop waipu dikerjakan pada rangkaian Fase 2 yang sama.

## 3. Hasil EDA

Eksplorasi awal menunjukkan dataset ini adalah persoalan klasifikasi multi-label yang timpang. Setiap citra membawa rata-rata 2,87 label, dengan jumlah label paling sedikit 1 dan paling banyak 9 label pada satu citra. Label primary muncul pada 37.513 citra atau 92,68 persen dari seluruh data, sedangkan label paling jarang adalah blow_down yang hanya muncul pada 98 citra atau 0,24 persen, disusul conventional_mine dengan 100 citra atau 0,25 persen. Artinya, frekuensi label terbanyak dan terjarang berjarak sekitar 383 banding 1. Ketimpangan seperti ini menjadi perhatian utama untuk pemodelan berikutnya, karena label penyebab deforestasi yang justru paling penting, seperti slash_burn dengan 209 citra, selective_logging dengan 340 citra, dan artisinal_mine dengan 339 citra, semuanya berada di kelompok label jarang. Untuk label cuaca, setiap citra pada data awal memiliki tepat satu label cuaca dan tidak ditemukan citra yang membawa dua label cuaca sekaligus. Label cuaca terbanyak adalah clear dengan 28.431 citra atau 70,24 persen, diikuti partly_cloudy 17,94 persen, haze 6,66 persen, dan cloudy 5,16 persen. Grafik distribusi label dan grid contoh citra per label tersimpan di notebook notebooks/01_eda_dan_persiapan_data.ipynb. Satu catatan soal penghitungan: pada berkas label asli terdapat 73 baris yang token labelnya tertulis ganda di dalam kolom tags, contohnya label cultivation yang tertulis dua kali pada satu baris yang sama. Karena itu, seluruh penghitungan distribusi label pada laporan ini memakai kehadiran label per citra, bukan jumlah kejadian token, supaya satu citra tidak terhitung dua kali untuk label yang sama. Distribusi lengkap 17 label sebelum dan sesudah cleaning adalah sebagai berikut.

| Label | Sebelum cleaning | Sesudah cleaning | Persen sesudah cleaning |
|---|---|---|---|
| agriculture | 12.315 | 12.315 | 30,42 |
| artisinal_mine | 339 | 339 | 0,84 |
| bare_ground | 862 | 862 | 2,13 |
| blooming | 332 | 332 | 0,82 |
| blow_down | 98 | 98 | 0,24 |
| clear | 28.431 | 28.431 | 70,24 |
| cloudy | 2.089 | 2.089 | 5,16 |
| conventional_mine | 100 | 100 | 0,25 |
| cultivation | 4.477 | 4.477 | 11,06 |
| habitation | 3.660 | 3.660 | 9,04 |
| haze | 2.697 | 2.697 | 6,66 |
| partly_cloudy | 7.261 | 7.261 | 17,94 |
| primary | 37.513 | 37.513 | 92,68 |
| road | 8.071 | 8.071 | 19,94 |
| selective_logging | 340 | 340 | 0,84 |
| slash_burn | 209 | 209 | 0,52 |
| water | 7.411 | 7.410 | 18,31 |

## 4. Hasil cleaning

Kebijakan tetap yang diterapkan:

1. Citra rusak atau berkasnya tidak ada di disk: baris dibuang.
2. Citra berlabel cloudy yang membawa label lain: label non-cuaca dihapus mengikuti aturan resmi dataset, baris dipertahankan.
3. Citra dengan lebih dari satu label cuaca: baris dibuang dan dicatat terpisah.
4. Citra tanpa label cuaca: baris dibuang dan dicatat terpisah.
5. Nama berkas duplikat: kemunculan pertama disimpan.

Hasil penerapan kebijakan tersebut sangat kecil dampaknya, karena data bawaan kompetisi ternyata sudah rapi. Tidak ditemukan citra rusak, nama berkas duplikat, ketidaksesuaian antara CSV dan berkas di disk, citra cloudy yang membawa label lain, maupun citra dengan label cuaca ganda. Satu-satunya temuan adalah train_24448 yang tidak memiliki label cuaca sama sekali, sehingga barisnya dibuang sesuai kebijakan keempat. Baris tersebut membawa label water, sehingga jumlah label water turun dari 7.411 menjadi 7.410, dan perubahan itu adalah satu-satunya perbedaan distribusi label sebelum dan sesudah cleaning. Rincian hasil cleaning per kategori adalah sebagai berikut, dan perbandingan distribusi labelnya juga tersimpan di data/processed/label_distribution.csv.

| Kategori | Jumlah |
|---|---|
| Baris awal | 40.479 |
| Duplikat image_name dibuang | 0 |
| Berkas di CSV tidak ada di disk | 0 |
| Berkas di disk tidak ada di CSV | 0 |
| Citra rusak | 0 |
| Citra cloudy dengan label lain yang dihapus | 0 |
| Baris dibuang karena label cuaca ganda | 0 |
| Baris dibuang karena tanpa label cuaca | 1 |
| Total data bersih | 40.478 |

## 5. Hasil split

Data bersih dibagi menjadi tiga bagian dengan stratifikasi iteratif multi-label, agar proporsi setiap label di tiap bagian sedekat mungkin dengan proporsi keseluruhan. Metode yang dipakai adalah stratifikasi iteratif dua tahap dengan penyeimbangan tukar untuk menekan deviasi proporsi label, karena pustaka skmultilearn tidak tersedia di lingkungan kerja. Seed yang dipakai adalah 42 agar pembagian dapat diulang dengan hasil yang sama. Deviasi proporsi label maksimum antara tiap split dan keseluruhan data hanya 0,0014, sehingga pembagian dinilai setia pada distribusi asli. Metode dan angka deviasi ini juga tercatat di data/processed/cleaning_log.json. Ukuran ketiga split adalah sebagai berikut.

| Split | Jumlah citra | Persen |
|---|---|---|
| Latih | 28.335 | 70,00 |
| Validasi | 6.071 | 15,00 |
| Uji | 6.072 | 15,00 |

Ketiga berkas hasil pembagian tersimpan di data/processed/split_train.csv, data/processed/split_val.csv, dan data/processed/split_test.csv. Ketiga split sudah diverifikasi saling lepas, artinya tidak ada nama citra yang muncul di dua split, dan gabungan ketiganya sama persis dengan data bersih.

## 6. Protokol evaluasi yang dikunci

1. Metrik utama: F2-score (beta = 2) micro di data uji, sama untuk ketiga model.
2. Metrik pendamping: F2 per label dan recall per label, dengan sorotan pada label penyebab deforestasi (slash_burn, selective_logging, artisinal_mine, conventional_mine).
3. Threshold 0,5 untuk model baseline dan advanced.
4. Threshold per label untuk model deep learning, disetel hanya di data validasi, tidak pernah di data uji.
5. Target: model deep learning mencapai F2 minimal 0,87 di data uji.

## 7. Kesiapan Fase 3

Dataset bersih dan split final sudah siap dipakai untuk pembangunan model baseline pada Fase 3. Seluruh berkas hasil Fase 2 tersimpan di dalam folder proyek dengan susunan sebagai berikut. Data mentah hasil unduhan dan ekstraksi berada di data/raw, yaitu train-jpg.tar.7z, train_v2.csv, dan folder train-jpg yang berisi 40.479 berkas JPG. Data hasil pengolahan berada di data/processed, yaitu train_v2_clean.csv untuk data bersih, split_train.csv, split_val.csv, dan split_test.csv untuk hasil pembagian, cleaning_log.json untuk catatan seluruh pemeriksaan dan cleaning, serta label_distribution.csv untuk perbandingan distribusi label sebelum dan sesudah cleaning. Kode yang menjalankan seluruh rangkaian tersimpan di scripts/persiapan_data.py dan dapat dijalankan ulang dengan hasil yang sama, sedangkan eksplorasi visualnya tersimpan di notebooks/01_eda_dan_persiapan_data.ipynb. Laporan ini tersimpan di folder results bersama berkas sumbernya.
