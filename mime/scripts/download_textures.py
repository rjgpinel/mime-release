import argparse
import csv
import os
import requests
import zipfile

from io import BytesIO
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

TEXTURES_BLACKLIST = [
    "sign",
    "roadlines",
    "manhole",
    "backdrop",
    "foliage",
    "TreeEnd",
    "TreeStump",
    "3DBread",
    "3DApple",
    "FlowerSet",
    "FoodSteps",
    "PineNeedles",
    "Grate",
    "PavingEdge",
    "Painting",
    "RockBrush",
    "WrinklesBrush",
    "Sticker",
    "3DRock",
    "PaintedWood001",
    "PaintedWood003",
    "PaintedWood002",
    "PaintedWood004",
    "PaintedWood005",
    "Carpet002",
    "SurfaceImperfections001",
    "SurfaceImperfections002",
    "SurfaceImperfections003",
    "SurfaceImperfections004",
    "SurfaceImperfections005",
    "SurfaceImperfections006",
    "SurfaceImperfections007",
    "SurfaceImperfections008",
    "SurfaceImperfections009",
    "SurfaceImperfections010",
    "SurfaceImperfections011",
    "SurfaceImperfections012",
    "SurfaceImperfections013",
    "Scratches001",
    "Scratches002",
    "Scratches003",
    "Scratches004",
    "Scratches005",
    "Tiles001",
    "DiamondPlate004",
    "Smear005",
    "Smear006",
    "Smear007",
    "Fingerprints004",
]


def get_args_parser():
    parser = argparse.ArgumentParser("Download textures script", add_help=False)
    parser.add_argument("output_dir", type=str)
    return parser


def main(args):

    # define compression transform
    crop_size = 512
    compress_transform = T.Compose(
        [
            T.Resize(crop_size),
            T.CenterCrop(crop_size),
        ]
    )

    # download the csv file, which contains all the download links
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "full_info.csv"
    csv_url = "https://cc0textures.com/api/v1/downloads_csv"
    # setting the default header, else the server does not allow the download
    headers = {"User-Agent": "Mozilla/5.0"}
    request = requests.get(csv_url, headers=headers)
    with open(str(csv_path), "wb") as f:
        f.write(request.content)

    # extract the download links with the asset name
    textures_url = {}
    with open(str(csv_path), "r") as csv_f:
        csv_reader = csv.DictReader(csv_f, delimiter=",")
        for line in csv_reader:
            if line["Filetype"] == "zip" and line["DownloadAttribute"] == "1K-JPG":
                textures_url[line["AssetID"]] = line["PrettyDownloadLink"]

    # download each asset and create a folder for it (unpacking + deleting the zip included)
    for idx, (asset, texture_url) in enumerate(textures_url.items()):
        blacklisted = False
        for blacklisted_asset in TEXTURES_BLACKLIST:
            if asset.lower().startswith(blacklisted_asset.lower()):
                blacklisted = True
                break
        if not blacklisted:
            print(f"Download asset: {asset} of {idx}/{len(textures_url)}")
            response = requests.get(texture_url, headers=headers)
            f = BytesIO(response.content)
            try:
                with zipfile.ZipFile(f) as zip_f:
                    zip_f.extract(f"{asset}_1K_Color.jpg", str(output_dir))
                # Process compress image
                texture_im = Image.open(str(output_dir / f"{asset}_1K_Color.jpg"))
                compressed_texture_im = compress_transform(texture_im)
                compressed_texture_im.save(
                    str(output_dir / f"{asset}_1K_Color.jpg"), optimize=True, quality=95
                )
            except (IOError, zipfile.BadZipfile) as e:
                import pudb

                pudb.set_trace()

                print("Bad zip file given as input.  %s" % e)
                raise e

    print(f"Done downloading textures, saved in {str(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download textures script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
