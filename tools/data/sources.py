"""Raw sources of the data contracts: download URL, local file and SHA-256 of that file (as used to build the
contracts). Local files live under dataset/ (interactions) and dataset/raw/metadata/ (item metadata); the same file in
a TaxPro-CL repository copy (HITAX_TAXPROCL_ROOT, config_path.taxprocl_file) is accepted too, with the same hash.

kind: 'file' (stored as downloaded), 'gunzip' (stored decompressed), 'zip:<member suffix>' (one member of a zip).
"""
LIGHTGCN = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/{}/{}"
MCAULEY = "https://mcauleylab.ucsd.edu/public_datasets/data/"

SOURCES = {
    "amazon-book": [
        dict(url=LIGHTGCN.format("amazon-book", n), path="dataset/amazon-book/" + n, kind="file", sha256=h)
        for n, h in (("train.txt", "529066026085919381acd4ce19f69e9eafa4af3fef1a10119ea5551cc2ef8530"),
                     ("test.txt", "6250754994dd953b91b48dccdd77a73f318e0634b4757c2abfcc9d33a54ab0f5"),
                     ("user_list.txt", "b59cc7e4a1826e1eb313c4f0557bdafbb70661346d9c93911b97fe0e9fbd7078"),
                     ("item_list.txt", "8eee6caf6e666d95fc24ca8e0ec089ee90b1b2d1dc17933726359a6b3ecd62cb"))] + [
        dict(url=MCAULEY + "amazon/categoryFiles/meta_Books.json.gz", path="dataset/raw/metadata/amazon/2014/meta_Books.json",
             kind="gunzip", sha256="3eb207f1f52312424016ddfbc5e04b27115326731019e5fd3d14277610d0b107"),
        dict(url=MCAULEY + "amazon_v2/metaFiles2/meta_Books.json.gz", path="dataset/raw/metadata/amazon/2018/meta_Books.json",
             kind="gunzip", sha256="6750ecf9bc5db279c20faa8d8fc161c6a9fd230d16dbdd48edcc088e52ef294e"),
        dict(url=MCAULEY + "amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
             path="dataset/raw/metadata/amazon/2023/meta_Books.jsonl", kind="gunzip",
             sha256="3b008960580e390874b6723d438a3df3558bc98b9547ed37c90dc1659109cec7")],
    "yelp2018": [
        dict(url=LIGHTGCN.format("yelp2018", n), path="dataset/yelp2018/" + n, kind="file", sha256=h)
        for n, h in (("train.txt", "1e9568725a9e833d0ac3bae01c94cd95367f73b3bb6104fd26b5f65ea5054093"),
                     ("test.txt", "9cd9b37926cd1447a06625ffa7914527573fb097c10cb0bbdb1d1b8ded3742fe"),
                     ("user_list.txt", "4931519d8b3c40d71a34cc386c3287b6d63c1be9a82edd7ee6bfad765e640778"),
                     ("item_list.txt", "1091bc824b60561906df552f2f7cb3ec61767b1e46fd4386b44400ae49712007"))] + [
        dict(url="https://zenodo.org/records/10998102/files/Yelp{}.zip?download=1".format(y),
             path="dataset/raw/metadata/yelp/{0}/Yelp{0}.item".format(y), kind="zip:Yelp{}.item".format(y), sha256=h)
        for y, h in (("2018", "67b615e2a54df36b58293bcae7f926318d024e677c226966893907d4eee9dc63"),
                     ("2021", "1640363ab0f99aacb8626640ed9f1a9a21fa4810e447f51756a855a48315e005"),
                     ("2022", "c9296ee0c9de2e334180c942484554351d6e89ad6fe8ff6eb8ad31eee424399c"))] + [
        # official category hierarchy. The page is not versioned and its markup changes with every site build, so a
        # download is accepted if its bytes equal the 2026-10-26 copy OR the hierarchy parsed from it (titles, locales,
        # parents; tools/data/parse_yelp_categories.hierarchy_sha256) equals the one used for the contracts.
        dict(url="https://docs.developer.yelp.com/docs/resources-categories",
             path="metadata/yelp2018/resources-categories.html", kind="file",
             sha256="b476bc4646424baec48b76d1b2a80ec4c07dd661c372070ed9fa20deee2b62ef", retrieved="2026-10-26",
             hierarchy_sha256="e23796e433ecd85a0fb4b2713703977a2befef86f3b88d3da8d0a287b4275f96")],
    "amazon2018-grocery": [
        dict(url=MCAULEY + "amazon_v2/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz",
             path="dataset/raw/amazon2018-grocery/Grocery_and_Gourmet_Food_5.json.gz", kind="file",
             sha256="351c026163fae6e9e0ff470ea5d43eaac02826a422e89555e3f07228c94ae98d"),
        dict(url=MCAULEY + "amazon_v2/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz",
             path="dataset/raw/amazon2018-grocery/meta_Grocery_and_Gourmet_Food.json.gz", kind="file",
             sha256="4444102d9de53e54068f252e7044997a9f8e489f6463ddf45b81f4c3c8295bf4")],
    "ml-1m": [
        dict(url="https://files.grouplens.org/datasets/movielens/ml-1m.zip", path="dataset/raw/ml-1m/ml-1m.zip",
             kind="file", sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20")],
}


def expected_sha256(rel_path):
    """Recorded SHA-256 of a source file given by its path relative to the project root."""
    for items in SOURCES.values():
        for s in items:
            if s["path"] == rel_path:
                return s["sha256"]
    raise KeyError(rel_path)
