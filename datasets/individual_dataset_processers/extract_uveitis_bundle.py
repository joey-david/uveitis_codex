import argparse
import shutil
import zipfile
from pathlib import Path


INNER_ZIPS = {
    "deepdrid": "DeepDRiD.zip",
    "eyepacs": "eyepacs_1024.zip",
    "fgadr": "FGADR-Seg-set_Release.zip",
    "uwf": "UWF.zip",
}


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def ensure_empty(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"{path} exists (use --force to overwrite)")
        shutil.rmtree(path)


def merge_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists() and not target.is_dir():
                raise SystemExit(f"Cannot merge {item} into {target}")
            merge_tree(item, target)
            item.rmdir()
        else:
            if target.exists():
                raise SystemExit(f"Refusing to overwrite {target}")
            shutil.move(str(item), target)
    src.rmdir()


def process_deepdrid(zip_path: Path, out_dir: Path, force: bool) -> None:
    tmp_dir = zip_path.parent / "deepdrid_extract"
    extract_zip(zip_path, tmp_dir)
    src_dir = tmp_dir / "DeepDRiD-1.1"
    ensure_empty(out_dir, force)
    shutil.move(str(src_dir), out_dir)
    shutil.rmtree(tmp_dir)


def process_eyepacs(zip_path: Path, out_dir: Path, force: bool) -> None:
    tmp_dir = zip_path.parent / "eyepacs_extract"
    extract_zip(zip_path, tmp_dir)
    src_dir = tmp_dir / "eyepacs_1024"
    ensure_empty(out_dir, force)
    shutil.move(str(src_dir), out_dir)
    shutil.rmtree(tmp_dir)


def process_fgadr(zip_path: Path, out_dir: Path, force: bool) -> None:
    tmp_dir = zip_path.parent / "fgadr_extract"
    extract_zip(zip_path, tmp_dir)
    seg_dir = tmp_dir / "Seg-set"

    ensure_empty(out_dir, force)
    out_dir.mkdir(parents=True, exist_ok=True)

    if seg_dir.exists():
        for item in seg_dir.iterdir():
            if item.name in {".DS_Store"}:
                continue
            shutil.move(str(item), out_dir / item.name)

    for item in tmp_dir.iterdir():
        if item.name in {"Seg-set", "__MACOSX", ".DS_Store"}:
            continue
        if item.is_file():
            shutil.move(str(item), out_dir / item.name)

    shutil.rmtree(tmp_dir)


def process_uwf(
    zip_path: Path, out_dir: Path, legacy_dir: Path, force: bool
) -> None:
    tmp_dir = zip_path.parent / "uwf_extract"
    extract_zip(zip_path, tmp_dir)

    if legacy_dir.exists():
        if out_dir.exists():
            merge_tree(legacy_dir, out_dir)
        else:
            shutil.move(str(legacy_dir), out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    images_dest = out_dir / "Images"
    metadata_dest = out_dir / "Metadata"
    if images_dest.exists():
        if not force:
            raise SystemExit(f"{images_dest} exists (use --force to overwrite)")
        shutil.rmtree(images_dest)
    if metadata_dest.exists():
        if not force:
            raise SystemExit(f"{metadata_dest} exists (use --force to overwrite)")
        shutil.rmtree(metadata_dest)

    out_dir.mkdir(parents=True, exist_ok=True)

    images_src = tmp_dir / "Original UWF Image"
    if images_src.exists():
        shutil.move(str(images_src), images_dest)

    metadata_dest.mkdir(parents=True, exist_ok=True)
    for item in tmp_dir.iterdir():
        if item.name in {"Original UWF Image", "__MACOSX", ".DS_Store"}:
            continue
        if item.is_file():
            shutil.move(str(item), metadata_dest / item.name)

    shutil.rmtree(tmp_dir)


def main() -> None:
    default_zip = Path("datasets/raw/uveitis_datasets.zip")
    default_out = Path("datasets")

    parser = argparse.ArgumentParser(
        description="Extract uveitis_datasets.zip into organized dataset folders."
    )
    parser.add_argument("--zip", dest="zip_path", default=default_zip)
    parser.add_argument("--out", dest="out_root", default=default_out)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing dataset folders"
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    out_root = Path(args.out_root)
    raw_dir = zip_path.parent
    work_dir = raw_dir / "_uveitis_extract_tmp"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting inner zips from {zip_path} -> {work_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in INNER_ZIPS.values():
            if name not in zf.namelist():
                raise SystemExit(f"Missing {name} in {zip_path}")
            zf.extract(name, work_dir)

    process_deepdrid(work_dir / INNER_ZIPS["deepdrid"], out_root / "deepdrid", args.force)
    (work_dir / INNER_ZIPS["deepdrid"]).unlink()
    print("DeepDRiD extracted.")

    process_eyepacs(work_dir / INNER_ZIPS["eyepacs"], out_root / "eyepacs", args.force)
    (work_dir / INNER_ZIPS["eyepacs"]).unlink()
    print("EyePACS extracted.")

    process_fgadr(work_dir / INNER_ZIPS["fgadr"], out_root / "fgadr", args.force)
    (work_dir / INNER_ZIPS["fgadr"]).unlink()
    print("fgadr extracted.")

    process_uwf(
        work_dir / INNER_ZIPS["uwf"],
        out_root / "uwf-700",
        out_root / "UWF-700",
        args.force,
    )
    (work_dir / INNER_ZIPS["uwf"]).unlink()
    print("UWF-700 extracted.")

    shutil.rmtree(work_dir)
    print("Done.")


if __name__ == "__main__":
    main()
