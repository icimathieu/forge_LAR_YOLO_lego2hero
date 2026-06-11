#!/usr/bin/env python3
"""Upload d'un dossier vers le HuggingFace Hub comme **dataset** — gratuit.

Le HF Hub héberge gratuitement les datasets (public ou privé, quotas larges).
C'est la façon recommandée de partager nos données avec les camarades : le CODE
reste sur GitHub (léger), les DONNÉES (mosaïques, datasets fragmentés) vont sur le
Hub — au lieu de gonfler le dépôt git.

Prérequis :
    pip install huggingface_hub
    huggingface-cli login          # token gratuit : https://huggingface.co/settings/tokens
    #  (ou export HF_TOKEN=hf_xxx, ou --token hf_xxx)

Exemples :
    # les 100 mosaïques pour Mathias (dataset fragmenté), public
    python3 tools/hf_upload.py --folder output/100mosaic4mathias \
        --repo-id <ton_user>/lego2hero-100mosaics

    # un dataset privé
    python3 tools/hf_upload.py --folder output/dataset \
        --repo-id <ton_user>/lego2hero-dataset --private
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Upload d'un dossier vers le HF Hub (dataset).")
    p.add_argument("--folder", required=True, help="dossier local à uploader")
    p.add_argument("--repo-id", required=True, help="ex: mathieu/lego2hero-100mosaics")
    p.add_argument("--private", action="store_true", help="dataset privé (défaut : public)")
    p.add_argument("--token", default=None, help="token HF (sinon : huggingface-cli login / HF_TOKEN)")
    p.add_argument("--commit", default="upload via tools/hf_upload.py", help="message de commit Hub")
    p.add_argument("--allow-patterns", nargs="*", default=None,
                   help="ne pousser que ces motifs (ex: '*.json' '*.png')")
    return p.parse_args()


def main():
    a = parse_args()
    from huggingface_hub import HfApi          # import tardif : message clair si absent
    api = HfApi(token=a.token)
    api.create_repo(repo_id=a.repo_id, repo_type="dataset",
                    private=a.private, exist_ok=True)
    api.upload_folder(folder_path=a.folder, repo_id=a.repo_id, repo_type="dataset",
                      commit_message=a.commit, allow_patterns=a.allow_patterns)
    print(f"OK → https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
