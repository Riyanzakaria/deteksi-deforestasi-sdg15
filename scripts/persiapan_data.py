#!/usr/bin/env python3
"""Persiapan data Fase 2 untuk proyek Learning Problem deteksi deforestasi.

Skrip ini menjalankan seluruh persiapan data secara berurutan:
1. Ekstrak data/raw/train-jpg.tar.7z bila folder train-jpg belum ada.
2. Baca data/raw/train_v2.csv dan verifikasi jumlah barisnya.
3. Cek integritas setiap citra JPG dengan PIL.
4. Cek aturan label (cloudy, jumlah label cuaca, duplikat, file hilang).
5. Cleaning dengan kebijakan tetap (lihat fungsi bersihkan), hasilnya ditulis
   ke data/processed/train_v2_clean.csv beserta cleaning_log.json dan
   label_distribution.csv.
6. Split multi-label bertingkat 70/15/15 (latih/validasi/uji) dengan
   iterative stratification, ditulis ke data/processed/split_*.csv.

Pemakaian:
    python scripts/persiapan_data.py [--root PATH_PROYEK]

Semua path relatif terhadap root proyek. Skrip idempoten: aman dijalankan
berulang, berkas keluaran ditimpa dengan hasil yang sama (seed tetap 42).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Urutan label baku dataset Planet (17 label).
LABELS = [
    "agriculture", "artisinal_mine", "bare_ground", "blooming", "blow_down",
    "clear", "cloudy", "conventional_mine", "cultivation", "habitation",
    "haze", "partly_cloudy", "primary", "road", "selective_logging",
    "slash_burn", "water",
]
WEATHER = {"clear", "partly_cloudy", "haze", "cloudy"}
JUMLAH_BARIS_DIHARAPKAN = 40479
SEED = 42
RASIO_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


def log_cetak(pesan: str) -> None:
    print(f"[persiapan_data] {pesan}", flush=True)


# ---------------------------------------------------------------------------
# 1. Ekstraksi arsip
# ---------------------------------------------------------------------------
def ekstrak_arsip(root: Path) -> str:
    """Ekstrak train-jpg.tar.7z ke data/raw/ bila folder train-jpg belum ada.

    Mengembalikan keterangan cara ekstraksi yang dipakai (atau 'dilewati').
    Arsip Kaggle bersifat dua tahap: .tar.7z berisi .tar, dan .tar berisi
    folder train-jpg/.
    """
    raw = root / "data" / "raw"
    folder = raw / "train-jpg"
    if folder.is_dir() and any(folder.glob("*.jpg")):
        return "dilewati (folder train-jpg sudah ada)"

    arsip_tar = raw / "train-jpg.tar"
    arsip_7z = raw / "train-jpg.tar.7z"

    if not arsip_tar.exists() and arsip_7z.exists():
        # Tahap 1: buka .tar.7z menjadi .tar
        cara = None
        try:
            import py7zr

            with py7zr.SevenZipFile(arsip_7z, mode="r") as z:
                z.extractall(path=raw)
            cara = "py7zr"
        except ImportError:
            for bin7z in ("7z", "7za", "7zr"):
                try:
                    hasil = subprocess.run(
                        [bin7z, "x", "-y", f"-o{raw}", str(arsip_7z)],
                        capture_output=True, text=True,
                    )
                    if hasil.returncode == 0:
                        cara = bin7z
                        break
                except FileNotFoundError:
                    continue
        if cara is None:
            raise SystemExit(
                "GAGAL EKSTRAK: folder data/raw/train-jpg belum ada dan arsip "
                "train-jpg.tar.7z tidak bisa dibuka karena library py7zr dan "
                "program 7z sama sama tidak tersedia. Pilihan: pasang py7zr "
                "(pip install py7zr) atau ekstrak arsip secara manual sampai "
                "folder data/raw/train-jpg/ berisi file .jpg, lalu jalankan "
                "ulang skrip ini."
            )
        log_cetak(f"Arsip .tar.7z dibuka dengan {cara}.")
    elif not arsip_tar.exists() and not arsip_7z.exists():
        raise SystemExit(
            "GAGAL: tidak ditemukan data/raw/train-jpg/, data/raw/train-jpg.tar, "
            "maupun data/raw/train-jpg.tar.7z. Unduh dulu train-jpg.tar.7z dari "
            "halaman Data kompetisi Kaggle ke data/raw/."
        )

    if arsip_tar.exists():
        # Tahap 2: buka .tar menjadi folder train-jpg/
        with tarfile.open(arsip_tar) as tf:
            tf.extractall(path=raw, filter="data")
        if folder.is_dir() and any(folder.glob("*.jpg")):
            return "tarfile (stdlib) setelah tahap .7z"
        raise SystemExit(
            "GAGAL EKSTRAK: train-jpg.tar berhasil dibuka tetapi folder "
            "train-jpg berisi file .jpg tidak ditemukan di data/raw/."
        )
    return "selesai"


# ---------------------------------------------------------------------------
# 3. Cek integritas citra
# ---------------------------------------------------------------------------
def _cek_satu_citra(tugas: tuple[str, str]) -> tuple[str, bool, str]:
    """Buka satu citra dengan PIL sampai tuntas. Dipakai worker Pool."""
    nama, path = tugas
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() membuat objek gambar tidak bisa dipakai lagi, jadi buka
        # ulang lalu load() untuk memastikan data piksel benar benar terbaca.
        with Image.open(path) as im:
            im.load()
        return nama, True, ""
    except Exception as exc:  # noqa: BLE001 - semua kegagalan dicatat apa adanya
        return nama, False, f"{type(exc).__name__}: {exc}"


def cek_integritas(folder: Path, nama_unik: list[str]) -> dict[str, str]:
    """Cek semua citra yang ada di disk dan tercantum di CSV.

    Mengembalikan dict {image_name: pesan_error} untuk citra yang rusak.
    """
    tugas = [(nama, str(folder / f"{nama}.jpg")) for nama in nama_unik]
    rusak: dict[str, str] = {}
    n_worker = max(1, min(len(tugas), __import__("os").cpu_count() or 1))
    with Pool(processes=n_worker) as pool:
        for nama, ok, err in pool.imap_unordered(_cek_satu_citra, tugas, chunksize=64):
            if not ok:
                rusak[nama] = err
    return rusak


# ---------------------------------------------------------------------------
# 6. Iterative stratification (fallback bila skmultilearn tidak tersedia)
# ---------------------------------------------------------------------------
def _iterative_stratification_satu_tahap(
    label_sets: list[set[str]], rasio: list[float], rng: np.random.Generator
) -> list[list[int]]:
    """Bagi indeks sampel ke beberapa fold mengikuti iterative stratification.

    label_sets: himpunan label tiap sampel. rasio: proporsi tiap fold
    (jumlahnya 1.0). Mengembalikan daftar indeks per fold.
    """
    n = len(label_sets)
    n_fold = len(rasio)
    ukuran_target = [int(round(n * r)) for r in rasio]
    # Koreksi pembulatan supaya totalnya tepat n.
    while sum(ukuran_target) > n:
        i = int(np.argmax(ukuran_target))
        ukuran_target[i] -= 1
    while sum(ukuran_target) < n:
        i = int(np.argmax([r - u / max(n, 1) for r, u in zip(rasio, ukuran_target)]))
        ukuran_target[i] += 1

    total_label = Counter()
    for s in label_sets:
        total_label.update(s)
    # Kebutuhan label per fold: berapa contoh label itu yang masih harus masuk.
    butuh_label = [
        {lab: total_label[lab] * rasio[f] for lab in total_label}
        for f in range(n_fold)
    ]
    sisa_fold = ukuran_target[:]
    folds: list[list[int]] = [[] for _ in range(n_fold)]
    belum = set(range(n))

    while belum:
        # Ambil label paling langka yang masih punya sampel belum terbagi.
        hitung_sisa = Counter()
        for i in belum:
            hitung_sisa.update(label_sets[i])
        if not hitung_sisa:  # semua sisa tanpa label (tidak terjadi di proyek ini)
            for i in sorted(belum):
                f = int(np.argmax(sisa_fold))
                folds[f].append(i)
                sisa_fold[f] -= 1
            break
        label_langka = min(hitung_sisa, key=lambda lab: (hitung_sisa[lab], lab))
        kandidat = [i for i in belum if label_langka in label_sets[i]]
        rng.shuffle(kandidat)
        for i in kandidat:
            # Fold yang paling membutuhkan label ini, dibatasi hanya fold
            # yang kapasitasnya belum penuh; seri diputus oleh fold yang
            # sisa kapasitasnya paling besar, lalu fold pertama.
            f = max(
                (g for g in range(n_fold) if sisa_fold[g] > 0),
                key=lambda g: (butuh_label[g].get(label_langka, 0.0), sisa_fold[g], -g),
            )
            folds[f].append(i)
            sisa_fold[f] -= 1
            for lab in label_sets[i]:
                butuh_label[f][lab] = butuh_label[f].get(lab, 0.0) - 1.0
            belum.discard(i)

    folds = _seimbangkan_dengan_tukar(folds, label_sets, total_label)
    return [sorted(f) for f in folds]


def _seimbangkan_dengan_tukar(
    folds: list[list[int]],
    label_sets: list[set[str]],
    total_label: Counter,
    maks_tukar: int = 1000,
) -> list[set[int]]:
    """Tekan deviasi proporsi label antar fold dengan pertukaran sampel.

    Pembagian awal iterative stratification bisa menyisakan label yang
    menumpuk di satu fold karena label yang umum diproses terakhir ketika
    kapasitas fold hampir habis. Fungsi ini berulang kali mencari label
    dengan sebaran proporsi paling timpang, lalu menukar satu sampel
    berlabel itu dari fold berlebih dengan satu sampel tanpa label itu
    dari fold yang kekurangan, selama jumlah kuadrat deviasi keseluruhan
    menurun. Ukuran fold tidak pernah berubah.
    """
    n = sum(len(f) for f in folds)
    n_fold = len(folds)
    fold_set = [set(f) for f in folds]
    ukuran = [len(f) for f in fold_set]
    hitung = [Counter() for _ in range(n_fold)]
    for g in range(n_fold):
        for i in fold_set[g]:
            hitung[g].update(label_sets[i])
    label_semua = sorted(total_label)

    def biaya_lokal(g: int, lab: str) -> float:
        if ukuran[g] == 0 or n == 0:
            return 0.0
        return (hitung[g].get(lab, 0) / ukuran[g] - total_label[lab] / n) ** 2

    for _ in range(maks_tukar):
        # Cari (label, fold berlebih, fold kekurangan) dengan sebaran
        # proporsi paling besar.
        kandidat_tukar: list[tuple[float, str, int, int]] = []
        for lab in label_semua:
            dev = [
                (hitung[g].get(lab, 0) / ukuran[g] - total_label[lab] / n)
                if ukuran[g] else 0.0
                for g in range(n_fold)
            ]
            g_lebih = int(np.argmax(dev))
            g_kurang = int(np.argmin(dev))
            # Sebaran di bawah 0,005 dianggap sudah seimbang; tidak dikejar.
            if g_lebih != g_kurang and dev[g_lebih] - dev[g_kurang] > 0.005:
                kandidat_tukar.append(
                    (dev[g_lebih] - dev[g_kurang], lab, g_lebih, g_kurang)
                )
        if not kandidat_tukar:
            break
        kandidat_tukar.sort(reverse=True)

        tertukar = False
        for _, lab, g_lebih, g_kurang in kandidat_tukar:
            # Pemindaian pasangan dibatasi 300 kandidat pertama per sisi
            # supaya waktu jalan tetap terkendali pada data besar.
            sumber = sorted(i for i in fold_set[g_lebih] if lab in label_sets[i])[:300]
            sasaran = sorted(i for i in fold_set[g_kurang] if lab not in label_sets[i])[:300]
            for i in sumber:
                label_terpengaruh = label_sets[i]
                ketemu = False
                for j in sasaran:
                    label_terpengaruh_ij = label_terpengaruh | label_sets[j]
                    biaya_sebelum = sum(
                        biaya_lokal(g, lb)
                        for g in (g_lebih, g_kurang)
                        for lb in label_terpengaruh_ij
                    )
                    # Terapkan tukar sementara pada penghitung.
                    for lb in label_sets[i]:
                        hitung[g_lebih][lb] -= 1
                        hitung[g_kurang][lb] += 1
                    for lb in label_sets[j]:
                        hitung[g_kurang][lb] -= 1
                        hitung[g_lebih][lb] += 1
                    biaya_sesudah = sum(
                        biaya_lokal(g, lb)
                        for g in (g_lebih, g_kurang)
                        for lb in label_terpengaruh_ij
                    )
                    if biaya_sesudah < biaya_sebelum - 1e-12:
                        fold_set[g_lebih].discard(i)
                        fold_set[g_lebih].add(j)
                        fold_set[g_kurang].discard(j)
                        fold_set[g_kurang].add(i)
                        tertukar = True
                        ketemu = True
                        break
                    # Batalkan tukar sementara.
                    for lb in label_sets[i]:
                        hitung[g_lebih][lb] += 1
                        hitung[g_kurang][lb] -= 1
                    for lb in label_sets[j]:
                        hitung[g_kurang][lb] += 1
                        hitung[g_lebih][lb] -= 1
                if ketemu:
                    break
            if tertukar:
                break
        if not tertukar:
            break
    return fold_set


def split_bertingkat(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    """Split 70/15/15. Pakai skmultilearn bila ada, selain itu fallback lokal."""
    label_sets = [set(t.split()) for t in df["tags"]]
    metode = ""
    idx = {"train": [], "val": [], "test": []}
    try:
        from skmultilearn.model_selection import iterative_train_test_split

        X = np.arange(len(df)).reshape(-1, 1)
        y = np.zeros((len(df), len(LABELS)), dtype=int)
        for baris, s in enumerate(label_sets):
            for lab in s:
                y[baris, LABELS.index(lab)] = 1
        idx_latih, _, idx_sisa, _ = iterative_train_test_split(
            X, y, test_size=0.30
        )
        idx["train"] = sorted(int(i) for i in np.ravel(idx_latih))
        sisa = sorted(int(i) for i in np.ravel(idx_sisa))
        sub = df.iloc[sisa]
        sub_sets = [label_sets[i] for i in sisa]
        rng = np.random.default_rng(SEED)
        f_val, f_test = _iterative_stratification_satu_tahap(
            sub_sets, [0.5, 0.5], rng
        )
        idx["val"] = sorted(sisa[i] for i in f_val)
        idx["test"] = sorted(sisa[i] for i in f_test)
        metode = "skmultilearn.iterative_train_test_split + fallback lokal untuk tahap kedua"
    except ImportError:
        rng = np.random.default_rng(SEED)
        f_latih, f_sisa = _iterative_stratification_satu_tahap(
            label_sets, [RASIO_SPLIT["train"], 0.30], rng
        )
        idx["train"] = sorted(f_latih)
        sisa = sorted(f_sisa)
        sub_sets = [label_sets[i] for i in sisa]
        f_val, f_test = _iterative_stratification_satu_tahap(
            sub_sets, [0.5, 0.5], rng
        )
        idx["val"] = sorted(sisa[i] for i in f_val)
        idx["test"] = sorted(sisa[i] for i in f_test)
        metode = (
            "fallback lokal: iterative stratification dua tahap plus "
            "penyeimbangan tukar (swap) penekan deviasi proporsi label "
            "(skmultilearn tidak tersedia)"
        )
    hasil = {nama: df.iloc[posisi].reset_index(drop=True) for nama, posisi in idx.items()}
    return hasil, metode


def hitung_deviasi_maks(
    df_semua: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> float:
    """Deviasi maksimum proporsi label antara tiap split dan keseluruhan."""
    def proporsi(frame: pd.DataFrame) -> Counter:
        c = Counter()
        for t in frame["tags"]:
            c.update(t.split())
        return c

    total = proporsi(df_semua)
    n_total = len(df_semua)
    dev_maks = 0.0
    for frame in splits.values():
        c = proporsi(frame)
        n = len(frame)
        for lab in LABELS:
            p_semua = total.get(lab, 0) / n_total if n_total else 0.0
            p_split = c.get(lab, 0) / n if n else 0.0
            dev_maks = max(dev_maks, abs(p_semua - p_split))
    return dev_maks


# ---------------------------------------------------------------------------
# Program utama
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Persiapan data Fase 2 (EDA/cleaning/split).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root proyek (default: folder induk dari scripts/).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    folder_citra = raw_dir / "train-jpg"
    log: dict = {
        "waktu_jalan": datetime.now().isoformat(timespec="seconds"),
        "root_proyek": str(root),
        "seed": SEED,
    }

    # --- a. Ekstraksi ------------------------------------------------------
    cara_ekstrak = ekstrak_arsip(root)
    log["ekstraksi"] = cara_ekstrak
    log_cetak(f"Ekstraksi: {cara_ekstrak}")

    # --- b. Baca CSV -------------------------------------------------------
    csv_path = raw_dir / "train_v2.csv"
    if not csv_path.exists():
        raise SystemExit(f"GAGAL: {csv_path} tidak ditemukan.")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["tags"] = df["tags"].astype(str).str.strip()
    n_awal = len(df)
    log["baris_csv_awal"] = n_awal
    log["baris_csv_diharapkan"] = JUMLAH_BARIS_DIHARAPKAN
    log["jumlah_baris_sesuai_harapan"] = bool(n_awal == JUMLAH_BARIS_DIHARAPKAN)
    log_cetak(
        f"train_v2.csv terbaca: {n_awal} baris "
        f"(harapan {JUMLAH_BARIS_DIHARAPKAN}, "
        f"{'sesuai' if n_awal == JUMLAH_BARIS_DIHARAPKAN else 'BERBEDA, dilaporkan apa adanya'})."
    )

    # Distribusi label sebelum cleaning (CSV apa adanya). Dihitung berbasis
    # kehadiran per baris (himpunan token per baris), bukan kejadian token,
    # karena sebagian baris memuat token label yang tertulis ganda.
    hitung_sebelum = Counter()
    for t in df["tags"]:
        hitung_sebelum.update(set(t.split()))

    # --- d (sebagian). Duplikat image_name: simpan kemunculan pertama ------
    duplikat_mask = df.duplicated(subset="image_name", keep="first")
    nama_duplikat = sorted(df.loc[duplikat_mask, "image_name"].tolist())
    log["duplikat_image_name"] = {
        "jumlah_baris_dibuang": int(duplikat_mask.sum()),
        "nama": nama_duplikat,
    }
    df = df.loc[~duplikat_mask].reset_index(drop=True)

    # --- d (sebagian). Kecocokan CSV dengan file di disk -------------------
    file_di_disk = {p.stem for p in folder_citra.glob("*.jpg")}
    nama_csv = set(df["image_name"])
    file_hilang = sorted(nama_csv - file_di_disk)
    file_tidak_di_csv = sorted(file_di_disk - nama_csv)
    log["file_tidak_ada_di_disk"] = {"jumlah": len(file_hilang), "nama": file_hilang}
    log["file_tidak_ada_di_csv"] = {
        "jumlah": len(file_tidak_di_csv),
        "nama": file_tidak_di_csv,
    }
    log_cetak(f"File hilang dari disk: {len(file_hilang)}; file yatim di disk: {len(file_tidak_di_csv)}.")

    # --- c. Integritas citra ----------------------------------------------
    nama_cek = sorted(nama_csv & file_di_disk)
    log_cetak(f"Mengecek integritas {len(nama_cek)} citra dengan PIL ...")
    rusak = cek_integritas(folder_citra, nama_cek)
    log["citra_rusak"] = {
        "jumlah": len(rusak),
        "nama": sorted(rusak),
        "detail_error": {k: rusak[k] for k in sorted(rusak)},
    }
    log_cetak(f"Citra rusak: {len(rusak)}.")

    # --- d/e. Aturan label dan cleaning ------------------------------------
    dibuang_rusak_hilang: list[str] = []
    cloudy_diperbaiki: list[str] = []
    dibuang_multi_cuaca: list[str] = []
    dibuang_tanpa_cuaca: list[str] = []
    baris_bersih: list[dict] = []

    set_hilang = set(file_hilang)
    for _, baris in df.iterrows():
        nama = baris["image_name"]
        label = baris["tags"].split()
        # Kebijakan 1: citra rusak atau file tidak ada -> baris dibuang.
        if nama in rusak or nama in set_hilang:
            dibuang_rusak_hilang.append(nama)
            continue
        cuaca = [lab for lab in label if lab in WEATHER]
        # Kebijakan 3: lebih dari satu label cuaca -> dibuang (cuaca ambigu).
        if len(cuaca) > 1:
            dibuang_multi_cuaca.append(nama)
            continue
        # Kebijakan 4: tanpa label cuaca -> dibuang (di luar skema).
        if len(cuaca) == 0:
            dibuang_tanpa_cuaca.append(nama)
            continue
        # Kebijakan 2: cloudy membawa label lain -> label non-cuaca dihapus.
        if cuaca == ["cloudy"] and len(label) > 1:
            label = ["cloudy"]
            cloudy_diperbaiki.append(nama)
        baris_bersih.append({"image_name": nama, "tags": " ".join(label)})

    df_bersih = pd.DataFrame(baris_bersih, columns=["image_name", "tags"])
    log["baris_dibuang_rusak_atau_hilang"] = {
        "jumlah": len(dibuang_rusak_hilang),
        "nama": sorted(dibuang_rusak_hilang),
    }
    log["cloudy_label_lain_dihapus"] = {
        "jumlah": len(cloudy_diperbaiki),
        "nama": sorted(cloudy_diperbaiki),
    }
    log["dibuang_label_cuaca_ganda"] = {
        "jumlah": len(dibuang_multi_cuaca),
        "nama": sorted(dibuang_multi_cuaca),
    }
    log["dibuang_tanpa_label_cuaca"] = {
        "jumlah": len(dibuang_tanpa_cuaca),
        "nama": sorted(dibuang_tanpa_cuaca),
    }
    log["total_bersih"] = len(df_bersih)

    # Label yang tidak dikenal (di luar 17 label baku): dicatat, tidak diubah.
    label_asing = sorted(
        {lab for t in df_bersih["tags"] for lab in t.split()} - set(LABELS)
    )
    log["label_di_luar_daftar_baku"] = label_asing

    # --- Tulis hasil cleaning ----------------------------------------------
    path_bersih = proc_dir / "train_v2_clean.csv"
    df_bersih.to_csv(path_bersih, index=False)

    hitung_sesudah = Counter()
    for t in df_bersih["tags"]:
        hitung_sesudah.update(set(t.split()))
    dist = pd.DataFrame(
        {
            "label": LABELS,
            "sebelum_cleaning": [hitung_sebelum.get(lab, 0) for lab in LABELS],
            "sesudah_cleaning": [hitung_sesudah.get(lab, 0) for lab in LABELS],
        }
    )
    dist["proporsi_sesudah"] = (dist["sesudah_cleaning"] / max(len(df_bersih), 1)).round(6)
    dist.to_csv(proc_dir / "label_distribution.csv", index=False)

    # --- f. Split -----------------------------------------------------------
    splits, metode_split = split_bertingkat(df_bersih)
    log["metode_split"] = metode_split
    for nama, frame in splits.items():
        frame.to_csv(proc_dir / f"split_{nama}.csv", index=False)
    dev_maks = hitung_deviasi_maks(df_bersih, splits)
    log["split"] = {
        "rasio": RASIO_SPLIT,
        "ukuran": {nama: len(frame) for nama, frame in splits.items()},
        "deviasi_proporsi_label_maksimum": round(dev_maks, 6),
    }
    log_cetak(f"Metode split: {metode_split}")
    log_cetak(f"Ukuran split: {log['split']['ukuran']}; deviasi proporsi maks {dev_maks:.4f}.")

    # --- Tulis log ----------------------------------------------------------
    (proc_dir / "cleaning_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- g. Ringkasan akhir --------------------------------------------------
    print()
    print("=== RINGKASAN PERSIAPAN DATA ===")
    print(f"Total baris CSV awal          : {n_awal}")
    print(f"Duplikat image_name dibuang   : {log['duplikat_image_name']['jumlah_baris_dibuang']}")
    print(f"File tidak ada di disk        : {len(file_hilang)}")
    print(f"Citra rusak                   : {len(rusak)}")
    print(f"Baris dibuang (rusak/hilang)  : {len(dibuang_rusak_hilang)}")
    print(f"Cloudy: label lain dihapus    : {len(cloudy_diperbaiki)} baris dipertahankan")
    print(f"Dibuang (label cuaca ganda)   : {len(dibuang_multi_cuaca)}")
    print(f"Dibuang (tanpa label cuaca)   : {len(dibuang_tanpa_cuaca)}")
    print(f"Total data bersih             : {len(df_bersih)}")
    print(f"Split latih/validasi/uji      : "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Berkas tertulis di {proc_dir}:")
    for nama_berkas in (
        "train_v2_clean.csv", "cleaning_log.json", "label_distribution.csv",
        "split_train.csv", "split_val.csv", "split_test.csv",
    ):
        print(f"  - {nama_berkas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
