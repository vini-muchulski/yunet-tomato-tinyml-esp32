from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
from PIL import Image, ImageFile

DATASET = "mamtag/tomato-village/versions/2"
VARIANT = "Variant-a(Multiclass Classification)"
TEST_DIR = (
    Path(__file__).resolve().parents[1]
    / "dataset" / "tomato_village" / "tomato_village" / VARIANT / "test"
)
CLASS_NAMES = (
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
)


def prepare_test_split(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".tomato-village-") as temp:
        prepared = Path(temp) / "test"
        count = 0
        for class_name in CLASS_NAMES:
            images = sorted((source / class_name).rglob("*.jpg"))
            if not images:
                raise FileNotFoundError(f"Nenhum JPEG encontrado em {source / class_name}")

            for image_path in images:
                target = prepared / image_path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    rgb.info.pop("icc_profile", None)
                    rgb.save(target, format="JPEG")
                count += 1

        prepared.rename(destination)
    return count


def main():
    if TEST_DIR.exists():
        raise SystemExit(f"O diretório de destino já existe: {TEST_DIR}")

    downloaded = Path(kagglehub.dataset_download(DATASET))
    test_dirs = [path for path in downloaded.rglob(f"{VARIANT}/test") if path.is_dir()]
    if len(test_dirs) != 1:
        raise SystemExit(f"Esperava um diretório {VARIANT}/test em {downloaded}")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    count = prepare_test_split(test_dirs[0], TEST_DIR)
    print(f"{count} imagens prontas em {TEST_DIR}")


if __name__ == "__main__":
    main()
