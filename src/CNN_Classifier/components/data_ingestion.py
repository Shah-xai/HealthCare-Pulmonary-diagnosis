from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import kagglehub
import shutil
import random
from collections import defaultdict
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> None:
        logger.info("Starting data ingestion process...")

        target_path: Path = Path(self.config.local_data_file)

        if target_path.exists():
            logger.info("Dataset already exists. Skipping download.")
        else:
            # 1) download to kaggle cache
            cache_path = Path(kagglehub.dataset_download(self.config.source_URL))
            logger.info(f"Data downloaded from Kaggle cache to {cache_path}.")

            # 2) copy into your desired project directory
            logger.info(f"Copying dataset from {cache_path} to {target_path}.")
            shutil.copytree(cache_path, target_path)

        # 1) Normalize class folder names inside raw_data
        self.clean_split_folders()

        # 2) Dedupe + re-split into root_dir/balanced
        self.dedupe_and_split()

        logger.info("Data ingestion process completed successfully.")

    def clean_split_folders(self) -> None:
        root = Path(self.config.local_data_file)
        root = find_effective_root(root)

        for split in ["train", "valid", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue

            patterns = {
                "*normal*": "normal",
                "*adeno*": "adenocarcinoma",
                "*large*": "large.cell.carcinoma",
                "*squamous*": "squamous.cell.carcinoma",
            }

            for pattern, clean_name in patterns.items():
                for folder in split_dir.glob(pattern):
                    if folder.is_dir() and folder.name != clean_name:
                        target = split_dir / clean_name
                        if not target.exists():
                            folder.rename(target)
                            logger.info(f"[{split}] {folder.name} → {clean_name}")
                        else:
                            logger.info(f"[{split}] Skipping {folder.name} (target exists)")

    def dedupe_and_split(self) -> None:
        """
        Reads from self.config.local_data_file (raw_data),
        writes deduped+resplit dataset to self.config.root_dir / 'balanced'
        and quarantines ONLY exact duplicates to self.config.root_dir / 'quarantine_exact_duplicates'
        """
        raw_root = find_effective_root(Path(self.config.local_data_file))

        output_root = Path(self.config.root_dir) / "balanced"
        quarantine_root = Path(self.config.root_dir) / "quarantine_exact_duplicates"

        cfg = SplitConfig(
            input_root=raw_root,
            output_root=output_root,
            quarantine_root=quarantine_root,
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            phash_hamming_threshold=6,
        )

        # Validate split ratios
        if abs((cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio) - 1.0) > 1e-6:
            raise ValueError("train/valid/test ratios must sum to 1.0")

        logger.info(f"Effective raw dataset root: {cfg.input_root}")
        logger.info(f"Balanced output root:       {cfg.output_root}")
        logger.info(f"Exact-dup quarantine root:  {cfg.quarantine_root}")

        items = collect_images_with_labels(cfg.input_root, cfg.exts)
        items.sort(key=lambda x: str(x[0]))  # deterministic ordering

        logger.info(f"Found {len(items)} images total (before dedupe).")

        kept_items, exact_groups = dedupe_images_exact_first(items)
        quarantine_exact_duplicates_only(cfg, exact_groups)

        logger.info(f"Kept {len(kept_items)} images after EXACT dedupe.")
        logger.info(f"Exact-duplicate groups: {len(exact_groups)}")

        splits = stratified_split(kept_items, cfg)
        export_splits(splits, cfg)

        for s in ["train", "valid", "test"]:
            n = count_images(cfg.output_root / s, cfg.exts)
            logger.info(f"[balanced] {s}: {n} images")

        logger.info(f"Balanced dataset written to: {cfg.output_root}")
        logger.info(f"Exact duplicates copied to:  {cfg.quarantine_root}")


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class SplitConfig:
    input_root: Path
    output_root: Path
    quarantine_root: Path
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    phash_hamming_threshold: int = 6
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------
# Helpers
# ----------------------------

def find_effective_root(input_root: Path) -> Path:
    subdirs = [d for d in input_root.iterdir() if d.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else input_root


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_images_with_labels(root: Path, exts: Tuple[str, ...]) -> List[Tuple[Path, str, str]]:
    """
    Returns (path, class_label, split_name)
    Expects: root/train/<class>/* etc
    """
    items: List[Tuple[Path, str, str]] = []
    for split in ["train", "valid", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    items.append((p, label, split))
    return items


# ----------------------------
# EXACT dedupe only
# ----------------------------

def dedupe_images_exact_first(
    items: List[Tuple[Path, str, str]],
) -> Tuple[
    List[Tuple[Path, str]],
    Dict[str, List[Tuple[Path, str, str]]],
]:
    """
    Exact dedupe by SHA256 across the whole dataset.

    Returns:
      kept_items: [(path, label)]
      exact_dupe_groups: sha -> [(path,label,split)...] for groups with len>1
    """
    sha_to_items: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)
    for p, lbl, sp in items:
        try:
            sha = sha256_file(p)
        except Exception:
            sha = f"__ERROR__:{p}"
        sha_to_items[sha].append((p, lbl, sp))

    exact_dupe_groups = {sha: arr for sha, arr in sha_to_items.items() if len(arr) > 1}

    # keep first representative per sha (items were sorted, so deterministic)
    keep_paths = {arr[0][0] for arr in sha_to_items.values()}
    kept_items = [(p, lbl) for (p, lbl, sp) in items if p in keep_paths]

    return kept_items, exact_dupe_groups


def safe_relpath(p: Path) -> Path:
    parts = [x for x in p.parts if x not in (p.anchor,)]
    return Path(*parts[-6:]) if len(parts) > 6 else Path(*parts)


def copy_safely(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(dst.stem + "__dupe" + dst.suffix)
    try:
        shutil.copy2(str(src), str(dst))
    except Exception:
        pass


def quarantine_exact_duplicates_only(
    cfg: SplitConfig,
    exact_dupe_groups: Dict[str, List[Tuple[Path, str, str]]],
) -> None:
    """
    Copies ONLY exact duplicates (everything except representative) into quarantine folder.
    Non-destructive to raw dataset.
    """
    cfg.quarantine_root.mkdir(parents=True, exist_ok=True)

    for sha, arr in exact_dupe_groups.items():
        # keep first, copy rest
        for p, lbl, sp in arr[1:]:
            rel = safe_relpath(p)
            dst_dir = cfg.quarantine_root / sp / lbl / rel.parent
            copy_safely(p, dst_dir / p.name)


# ----------------------------
# Splitting + export
# ----------------------------

def stratified_split(items: List[Tuple[Path, str]], cfg: SplitConfig) -> Dict[str, List[Tuple[Path, str]]]:
    rng = random.Random(cfg.seed)

    by_class: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for p, lbl in items:
        by_class[lbl].append((p, lbl))

    splits = {"train": [], "valid": [], "test": []}

    for lbl, arr in by_class.items():
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(cfg.train_ratio * n)
        n_valid = int(cfg.valid_ratio * n)
        splits["train"].extend(arr[:n_train])
        splits["valid"].extend(arr[n_train:n_train + n_valid])
        splits["test"].extend(arr[n_train + n_valid:])

    for k in splits:
        rng.shuffle(splits[k])

    return splits


def export_splits(splits: Dict[str, List[Tuple[Path, str]]], cfg: SplitConfig) -> None:
    if cfg.output_root.exists():
        shutil.rmtree(cfg.output_root)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)

    for split_name, items in splits.items():
        for src, lbl in items:
            dst_dir = cfg.output_root / split_name / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{src.stem}__{rng.randint(100000, 999999)}{src.suffix}"
            shutil.copy2(src, dst)


def count_images(root: Path, exts: Tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)