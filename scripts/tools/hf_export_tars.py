#!/usr/bin/env python3
"""Exporte un dataset curriculum (`<src>/L*/`) vers le HF Hub en **tar.gz par palier**.

Un dataset de 25 000 instances = ~500 k fichiers → un upload browsable est
impraticable (HF rame/refuse au-delà de ~100 k fichiers). On archive donc CHAQUE
palier (`L0_explode/`, …) en un seul `.tar.gz`, on l'uploade, puis on **supprime
l'archive locale** avant de passer au suivant → pic disque ≈ une seule archive.

En bonus on pousse les `gnn_meta.json` (browsables, petits) et, optionnellement,
on supprime d'anciens dossiers *legacy* du repo.

Prérequis : `huggingface-cli login` (ou HF_TOKEN). `HF_HUB_ENABLE_HF_TRANSFER=1`
recommandé pour la vitesse.

Exemple :
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 scripts/tools/hf_export_tars.py \
        --src output/balanced --repo-id icimathieu/lego2hero-100mosaics \
        --path-in-repo balanced --delete-legacy clean degraded --card /tmp/hf_card.md
"""
import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="output/balanced", help="dossier contenant les paliers L*/")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--path-in-repo", default="balanced", help="préfixe cible dans le repo")
    p.add_argument("--tmp", default="output/_tars", help="dossier temporaire des archives")
    p.add_argument("--delete-legacy", nargs="*", default=[],
                   help="dossiers du repo à supprimer après upload (ex: clean degraded)")
    p.add_argument("--card", default=None, help="README.md (carte) à pousser à la racine")
    p.add_argument("--token", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=a.token)
    api.create_repo(a.repo_id, repo_type="dataset", exist_ok=True)
    src = Path(a.src)
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    paliers = sorted(d.name for d in src.iterdir() if d.is_dir() and d.name.startswith("L"))
    print(f"paliers: {paliers}")

    if a.card:
        api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md",
                        repo_id=a.repo_id, repo_type="dataset",
                        commit_message="carte : curriculum 5 paliers (remplace 100mosaics legacy)")
        print("[card] poussée")

    for L in paliers:                                   # gnn_meta browsables (petits)
        meta = src / L / "gnn_meta.json"
        if meta.exists():
            api.upload_file(path_or_fileobj=str(meta),
                            path_in_repo=f"{a.path_in_repo}/{L}.gnn_meta.json",
                            repo_id=a.repo_id, repo_type="dataset",
                            commit_message=f"meta {L}")
            print(f"[meta] {L}")

    for L in paliers:                                   # 1 archive par palier
        repo_path = f"{a.path_in_repo}/{L}.tar.gz"
        if api.file_exists(a.repo_id, repo_path, repo_type="dataset"):
            print(f"[skip] {L} (déjà sur le repo) — resume")  # relance idempotente
            continue
        tar = tmp / f"{L}.tar.gz"
        print(f"[tar] {L} …", flush=True)
        subprocess.run(["tar", "-czf", str(tar), "-C", str(src), L], check=True)
        gb = tar.stat().st_size / 1e9
        print(f"[upload] {L} ({gb:.1f} Go) …", flush=True)
        api.upload_file(path_or_fileobj=str(tar), path_in_repo=repo_path,
                        repo_id=a.repo_id, repo_type="dataset",
                        commit_message=f"palier {L} (tar.gz)")
        tar.unlink()                                    # libère le disque avant le suivant
        print(f"[done] {L}", flush=True)

    for folder in a.delete_legacy:
        try:
            api.delete_folder(path_in_repo=folder, repo_id=a.repo_id, repo_type="dataset",
                              commit_message=f"retrait legacy {folder}")
            print(f"[legacy] supprimé : {folder}")
        except Exception as e:
            print(f"[legacy] échec {folder}: {e}")

    print(f"EXPORT FINI → https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
