---
title: "Deteksi Aktivitas Deforestasi Hutan Tropis melalui Klasifikasi Multi-Label Citra Satelit Berbasis Deep Learning"
subtitle: "Laporan Proyek Learning Problem"
author:
- "Kelompok 10"
- "Riyan Zakaria Zulkarnain (244311058)"
- "Andrian Maulana (244311036)"
- "Program Studi Teknologi Rekayasa Perangkat Lunak"
- "Politeknik Negeri Madiun"
date: "November 2026"
toc-title: "Daftar Isi"
---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Hutan tropis penting bagi keanekaragaman hayati dan iklim, tetapi terus menyusut akibat tebang bakar, penebangan selektif, pertambangan, dan pembukaan lahan. Pemantauannya selama ini masih manual dan lambat, karena bergantung pada inspeksi lapangan dan pembacaan citra oleh analis. Hansen dkk. (2013) mencatat kehilangan hutan dunia sebesar 2,3 juta kilometer persegi pada tahun 2000 sampai 2012; pada kurun itu deforestasi di Brasil menurun, tetapi deforestasi di Indonesia justru meningkat. Citra satelit kini tersedia melimpah, jadi persoalannya bukan lagi ketersediaan data, melainkan cara memberi label pada citra itu secara otomatis. Proyek ini menjawab persoalan tersebut dengan membangun klasifikasi multi-label pada citra satelit hutan tropis, agar aktivitas penyebab deforestasi dapat dikenali oleh model dan area yang terindikasi gundul dapat lebih dulu diperiksa di lapangan.

## 1.2 Rumusan Masalah

1. Bagaimana mengklasifikasikan 17 label tutupan lahan dan aktivitas manusia pada satu citra satelit hutan tropis secara otomatis?
2. Bagaimana menangani label langka penyebab deforestasi, yang frekuensinya di bawah 1 sampai 7 persen, agar recall-nya tidak tertutup label dominan?
3. Seberapa besar kenaikan F2-score dari machine learning klasik ke CNN transfer learning?
4. Bagaimana penyetelan threshold per label memengaruhi keseimbangan recall dan precision?

## 1.3 Tujuan Penelitian

1. Membangun dan membandingkan tiga model, yaitu model baseline, model advanced, dan model deep learning, pada data yang sama.
2. Model deep learning mencapai F2-score minimal 0,87 pada data uji, dengan patokan hasil penelitian terbitan sebesar 0,89 sampai 0,93.
3. Recall label penyebab deforestasi naik dari model baseline, yang dibuktikan dari perbandingan hasil per label dan penyetelan threshold.
4. Tersedia analisis kesalahan dan catatan etika, termasuk batas generalisasi model dari hutan Amazon ke hutan Indonesia.

## 1.4 Keterkaitan dengan Tujuan Pembangunan Berkelanjutan

Proyek ini berkaitan dengan Tujuan Pembangunan Berkelanjutan ke-15, yaitu SDG 15 (Life on Land), khususnya target 15.2 tentang pengelolaan hutan secara lestari, penghentian deforestasi, dan pemulihan hutan yang terdegradasi. Machine learning mempercepat penemuan area gundul dari citra satelit, sehingga area tersebut dapat lebih dulu diperiksa di lapangan tanpa menunggu inspeksi manual yang lambat. Dengan cara itu, hasil proyek ini mendukung upaya menjaga hutan tropis yang menjadi ruang hidup banyak makhluk dan penopang iklim.

## 1.5 Pelaksanaan Pengajuan

Pengajuan ide proyek dilaksanakan pada 6 November 2026. Pada hari itu, kelompok mengumpulkan formulir pengajuan ide beserta proposal singkat yang memuat latar belakang, rumusan masalah, tujuan, keterkaitan dengan tujuan pembangunan berkelanjutan, dan rancangan metode. Repository GitHub proyek juga dibuat pada 6 November 2026 dan dapat diakses publik, sehingga seluruh berkas dan perkembangan pekerjaan proyek tersimpan di satu tempat. Tautan repository dan tautan dataset yang dipakai dicantumkan pada lampiran laporan ini.

# BAB II DATASET DAN RANCANGAN METODE

## 2.1 Dataset

Dataset yang dipakai adalah data latih kompetisi Planet: Understanding the Amazon from Space yang diselenggarakan di Kaggle pada tahun 2017. Dataset ini berisi 40.479 citra satelit wilayah hutan Amazon, masing-masing berukuran 256 x 256 piksel dalam format JPG. Setiap citra diberi label secara multi-label, artinya satu citra dapat membawa beberapa label sekaligus dari 17 label yang tersedia. Ke-17 label itu terbagi menjadi tiga kelompok. Kelompok pertama adalah label cuaca, yaitu clear, partly_cloudy, haze, dan cloudy. Kelompok kedua adalah label tutupan lahan yang umum, yaitu primary, water, agriculture, cultivation, habitation, dan road. Kelompok ketiga adalah label yang jarang muncul dan sebagian besar berkaitan dengan aktivitas penyebab deforestasi, yaitu slash_burn, selective_logging, artisinal_mine, conventional_mine, bare_ground, blooming, dan blow_down. Berkas yang diunduh dari halaman data kompetisi ada dua, yaitu arsip citra train-jpg.tar.7z berukuran 629.296.207 byte dan berkas label train_v2.csv yang berisi 40.479 baris, dengan satu baris untuk setiap citra.

## 2.2 Rancangan Pemodelan

Perumusan persoalan proyek ini sebagai persoalan machine learning diringkas pada Tabel 2.1 berikut.

Tabel 2.1 Learning Problem proyek deteksi deforestasi

| Komponen | Isian |
|---|---|
| Target | 17 label multi-label pada tiap citra, terdiri dari 4 label cuaca, 6 label tutupan lahan umum, dan 7 label langka penyebab deforestasi; keluaran model berupa vektor biner 17 dimensi. |
| Objective | F2-score (beta 2) setinggi mungkin pada data uji dengan menekankan recall label deforestasi; target F2-score 0,87 dan recall label langka di atas model baseline. |
| Data | Citra satelit Planet dari Kaggle tahun 2017, sebanyak 40.479 citra latih berukuran 256 x 256 piksel dengan 17 label; jenis data berupa citra; aksesnya legal melalui pendaftaran kompetisi yang gratis. |

Tiga model dibangun dan dibandingkan pada data yang sama. Model pertama adalah model baseline berupa Logistic Regression yang bekerja dari fitur warna dan tekstur yang diekstrak dari setiap citra. Model kedua adalah model advanced berupa Random Forest dan XGBoost yang memakai fitur warna dan tekstur yang sama, agar terlihat seberapa besar kenaikan hasil dari algoritma yang lebih kuat pada fitur yang setara. Model ketiga adalah model deep learning berupa ResNet50 dengan transfer learning dari bobot pra-latih ImageNet, dilengkapi kepala klasifikasi multi-label dengan keluaran sigmoid untuk setiap label. Seluruh pelatihan model dijalankan di Colab atau Kaggle dengan GPU T4.

## 2.3 Protokol Evaluasi

Penilaian model memakai F2-score dengan beta 2 secara micro pada data uji sebagai metrik utama, dan metrik ini berlaku sama untuk ketiga model. Nilai beta 2 membuat recall lebih berbobot daripada precision, sesuai tujuan proyek yang menekankan penemuan label penyebab deforestasi. Selain nilai keseluruhan, hasil juga dibaca per label melalui F2-score dan recall tiap label, dengan sorotan khusus pada empat label penyebab deforestasi, yaitu slash_burn, selective_logging, artisinal_mine, dan conventional_mine. Model baseline dan model advanced memakai threshold tetap sebesar 0,5, sedangkan model deep learning memakai threshold yang disetel per label dan penyetelannya hanya dilakukan pada data validasi, tidak pernah pada data uji. Target yang ditetapkan adalah model deep learning mencapai F2-score minimal 0,87 pada data uji.

# BAB III EKSPLORASI DAN PERSIAPAN DATA

## 3.1 Eksplorasi Data Awal

Eksplorasi awal menunjukkan dataset ini adalah persoalan klasifikasi multi-label yang timpang. Setiap citra membawa rata-rata 2,87 label, dengan jumlah label paling sedikit 1 dan paling banyak 9 label pada satu citra. Label primary muncul pada 37.513 citra atau 92,68 persen dari seluruh data, sedangkan label paling jarang adalah blow_down yang hanya muncul pada 98 citra atau 0,24 persen, disusul conventional_mine dengan 100 citra atau 0,25 persen. Artinya, frekuensi label terbanyak dan terjarang berjarak sekitar 383 banding 1. Ketimpangan ini menjadi perhatian utama untuk pemodelan berikutnya, karena label penyebab deforestasi yang justru paling penting, seperti slash_burn dengan 209 citra, selective_logging dengan 340 citra, dan artisinal_mine dengan 339 citra, semuanya berada di kelompok label jarang. Untuk label cuaca, setiap citra memiliki tepat satu label cuaca dan tidak ditemukan citra yang membawa dua label cuaca sekaligus. Label cuaca terbanyak adalah clear dengan 28.431 citra atau 70,24 persen, diikuti partly_cloudy sebanyak 7.261 citra, haze sebanyak 2.697 citra, dan cloudy sebanyak 2.089 citra. Satu catatan soal penghitungan: pada berkas label asli terdapat 73 baris yang token labelnya tertulis ganda di dalam kolom tags, contohnya label cultivation yang tertulis dua kali pada satu baris yang sama. Karena itu, seluruh penghitungan distribusi label pada laporan ini memakai kehadiran label per citra, bukan jumlah kejadian token, agar satu citra tidak terhitung dua kali untuk label yang sama. Distribusi lengkap 17 label sebelum dan sesudah pembersihan disajikan pada Tabel 3.1 berikut.

Tabel 3.1 Distribusi 17 label sebelum dan sesudah pembersihan data

| Label | Sebelum pembersihan | Sesudah pembersihan | Persen sesudah pembersihan |
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

## 3.2 Pemeriksaan dan Pembersihan Data

Seluruh citra dan label diperiksa satu per satu sebelum data dipakai. Pemeriksaan citra dilakukan dengan membuka dan mendekode setiap berkas JPG untuk mendeteksi berkas yang rusak, sedangkan pemeriksaan label meliputi kesesuaian nama berkas pada berkas label dengan berkas di disk, ada tidaknya duplikasi nama berkas, dan aturan label cuaca. Pembersihan data mengikuti lima kebijakan tetap berikut ini.

1. Citra rusak atau berkasnya tidak ada di disk, baris dibuang.
2. Citra berlabel cloudy yang membawa label lain, label selain cuaca dihapus mengikuti aturan resmi dataset dan baris dipertahankan.
3. Citra dengan lebih dari satu label cuaca, baris dibuang dan dicatat terpisah.
4. Citra tanpa label cuaca, baris dibuang dan dicatat terpisah.
5. Nama berkas duplikat, kemunculan pertama disimpan.

Hasil penerapan kebijakan tersebut berdampak sangat kecil, karena data bawaan kompetisi ternyata sudah rapi. Tidak ditemukan citra rusak, nama berkas duplikat, ketidaksesuaian antara berkas label dan berkas di disk, citra cloudy yang membawa label lain, maupun citra dengan label cuaca ganda. Satu-satunya temuan adalah train_24448 yang tidak memiliki label cuaca sama sekali, sehingga barisnya dibuang sesuai kebijakan keempat. Baris tersebut membawa label water, sehingga jumlah label water turun dari 7.411 menjadi 7.410, dan perubahan itu adalah satu-satunya perbedaan distribusi label sebelum dan sesudah pembersihan. Rincian hasil pembersihan per kategori disajikan pada Tabel 3.2 berikut.

Tabel 3.2 Hasil pembersihan data per kategori

| Kategori | Jumlah |
|---|---|
| Baris awal | 40.479 |
| Duplikat nama berkas dibuang | 0 |
| Berkas pada berkas label tidak ada di disk | 0 |
| Berkas di disk tidak ada pada berkas label | 0 |
| Citra rusak | 0 |
| Citra cloudy dengan label lain yang dihapus | 0 |
| Baris dibuang karena label cuaca ganda | 0 |
| Baris dibuang karena tanpa label cuaca | 1 |
| Total data bersih | 40.478 |

## 3.3 Pembagian Data

Data bersih dibagi menjadi tiga bagian, yaitu data latih, data validasi, dan data uji, dengan perbandingan 70 persen, 15 persen, dan 15 persen. Pembagian memakai stratifikasi iteratif multi-label dua tahap yang disertai penyeimbangan tukar, agar proporsi setiap label pada tiap bagian sedekat mungkin dengan proporsi keseluruhan. Angka acuan atau seed yang dipakai adalah 42, agar pembagian dapat diulang dengan hasil yang sama. Deviasi proporsi label maksimum antara tiap bagian dan keseluruhan data hanya 0,0014, sehingga pembagian dinilai setia pada distribusi asli. Ketiga bagian sudah diverifikasi saling lepas, artinya tidak ada nama citra yang muncul pada dua bagian, dan gabungan ketiganya sama persis dengan data bersih. Ukuran ketiga bagian disajikan pada Tabel 3.3 berikut.

Tabel 3.3 Ukuran pembagian data latih, validasi, dan uji

| Bagian | Jumlah citra | Persen |
|---|---|---|
| Latih | 28.335 | 70,00 |
| Validasi | 6.071 | 15,00 |
| Uji | 6.072 | 15,00 |

# BAB IV PENUTUP

## 4.1 Kesimpulan

Persiapan data telah selesai dan dataset dinyatakan siap dilatih. Dari 40.479 citra awal, hanya satu baris yang dibuang, sehingga data bersih berjumlah 40.478 citra dan sudah terbagi menjadi data latih, data validasi, dan data uji dengan proporsi label yang terjaga. Temuan terpenting dari eksplorasi data adalah ketimpangan label yang ekstrem, sekitar 383 banding 1 antara label terbanyak dan label terjarang, dan label yang jarang itu justru label penyebab deforestasi yang menjadi fokus proyek ini. Karena itu, protokol evaluasi sudah ditetapkan sejak awal dengan menekankan recall melalui F2-score dan penyetelan threshold per label, agar keberhasilan model tidak hanya ditentukan oleh label yang dominan.

## 4.2 Rencana Lanjutan

Pekerjaan berikutnya adalah melatih ketiga model pada data yang sudah disiapkan. Urutannya dimulai dari model baseline Logistic Regression, dilanjutkan model advanced Random Forest dan XGBoost, kemudian model deep learning ResNet50 dengan penyetelan threshold per label. Setelah ketiga model selesai dilatih, hasilnya dibandingkan dengan protokol evaluasi yang sama, disertai analisis kesalahan dan catatan mengenai batas generalisasi model dari hutan Amazon ke hutan Indonesia serta catatan etika penggunaannya. Rangkaian pekerjaan proyek ini ditutup dengan penulisan laporan akhir dan presentasi hasil.

# DAFTAR PUSTAKA

Ji, J., Jing, W., Chen, G., Lin, J., & Song, H. (2020). Multi-label remote sensing image classification with latent semantic dependencies. *Remote Sensing*, *12*(7), 1110. https://doi.org/10.3390/rs12071110

Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, *342*(6160), 850-853. https://doi.org/10.1126/science.1244693

# LAMPIRAN

Tautan repository proyek: https://github.com/Riyanzakaria/deteksi-deforestasi-sdg15

Tautan dataset pada Kaggle: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space
