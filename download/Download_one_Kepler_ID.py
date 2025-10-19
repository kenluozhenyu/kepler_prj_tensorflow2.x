# =============================================================================
# This procedure will download the light-curve files for one Kepler ID
# The store location is in DATA_FOR_PREDICTION_FOLDER (defined in
# the environment module).
# The folder structure will be same as the Kepler data files
# =============================================================================
import os
import urllib.request
from urllib.error import URLError
from bs4 import BeautifulSoup
import requests

from configure import environment


def download_one_kepler_id_files(kepid):
    kepid_formatted = "{0:09d}".format(int(kepid))  # Pad with zeros.
    subdir = "{}/{}".format(kepid_formatted[0:4], kepid_formatted)

    strfolder = os.path.join(environment.DATA_FOR_PREDICTION_FOLDER, kepid_formatted[0:4])

    if not os.path.exists(strfolder):
        os.mkdir(strfolder)

    strfolder = os.path.join(environment.DATA_FOR_PREDICTION_FOLDER, kepid_formatted[0:4], kepid_formatted)

    if not os.path.exists(strfolder):
        os.mkdir(strfolder)

    download_dir = strfolder

    url = "{}/{}/".format(environment.BASE_URL, subdir)

    try:
        page = requests.get(url).text
        # print page
        soup = BeautifulSoup(page, 'html.parser')
        # files = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('fits')]
        files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('_llc.fits')]
        # print(files)

        for file in files:
            file_url = url + '/' + file
            dest_file = os.path.join(download_dir, file)

            print("Downloading {} to {}".format(file_url, download_dir))
            if not os.path.isfile(dest_file):
                try:
                    urllib.request.urlretrieve(file_url, dest_file)
                except URLError as e2:
                    print("\nERROR => File {} download failed".format(file_url))
            else:
                print("File already downloaded. Abort this one.")
    except URLError as e1:
        print("Getting {} failed.".format(url))


def main():
    # Change this value for your propose
    KEPLER_ID_TO_DOWNLOAD = 757450

    download_one_kepler_id_files(KEPLER_ID_TO_DOWNLOAD)


if __name__ == "__main__":
    main()

