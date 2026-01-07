import glob
import json
import os
import os.path
import argparse


def main():
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description='Process JSON files in a specified folder')
    parser.add_argument('folder_name', type=str, help='Name of the folder to process')
    args = parser.parse_args()

    folder_name = args.folder_name
    label_list = glob.glob(f"./{folder_name}/*.json")
    search_string = "_"

    print(f"Found {len(label_list)} JSON files in folder: {folder_name}")

    for fname in label_list:
        print(f"Processing file: {fname}")

        # Windows/Linux 호환 경로 처리
        # os.path.basename()을 사용하여 파일명만 추출
        json_fname = os.path.basename(fname)
        json_fname = json_fname.split(".json")[0]
        print("라벨네임 : ", json_fname)

        with open(fname, "r") as json_file:
            json_data = json.load(json_file)
            json_data["imageData"] = None

            if os.path.exists(f"./{folder_name}/{json_fname}.jpg"):
                json_data["imagePath"] = json_fname + ".jpg"
            elif os.path.exists(f"./{folder_name}/{json_fname}.png"):
                json_data["imagePath"] = json_fname + ".png"
            else:
                json_data["imagePath"] = json_fname + ".bmp"

            with open(fname, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            print("오오 : ", len(json_data["shapes"]))
            print(json_data["imagePath"])
            org_path = json_data["imagePath"]

            with open(fname, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)


if __name__ == "__main__":
    main()