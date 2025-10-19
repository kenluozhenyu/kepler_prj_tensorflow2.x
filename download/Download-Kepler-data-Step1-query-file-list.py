# =============================================================================
# To download the ~ 80G byte training data from web, and with the consideration
# of the network performance, I decided to separate the download procedure to
# two steps:
#   1) Query all the file list with file size information (size in KB). Store
#      this information to a .json file.
#   2) Do the download according to the record in that .json file
#
# Both steps are designed to be able to re-run several times so as to be able
# to get all necessary data in case of network failure.
#
# This file implements the "step 1"
# =============================================================================
import os
import urllib.request
from urllib.error import URLError
import csv
from bs4 import BeautifulSoup
import requests
import threading
import math
import json
import re

from configure import environment

# This file is to store the whole download file list
# The file list is obtained from:
#   -- The Kepler ID in the KEPLER_CSV_FILE
#   -- The '*_llc.fits' for each Kepler ID stored online
#
# It is intended not to define this as a global const as this is only used for the download procedure
# Just need to ensure the folder exists. The file will be created automatically upon querying
KEPLER_FILE_LIST_TO_DOWNLOAD = "E:/Kepler/q1_q17_dr24_tce_file_list_with_size.json"

# You may change this value depend on your computer performance
NUM_OF_THREADS = 48

# All threads will store the queried data (file list and file size)
# to this structure
kepid_to_file_list = {}
total_num_kepids = 0

mutex = threading.Lock()

# ================================================================================
# functions' definition


def save_file_list():
    json_data = json.dumps(kepid_to_file_list)
    f = open(KEPLER_FILE_LIST_TO_DOWNLOAD, "w")
    f.write(json_data)
    f.close()


class myThread (threading.Thread):
    def __init__(self, threadID, name, kepid_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.kepid_list = kepid_list
        self.num_kepids = len(kepid_list)

    def run(self):
        print("Start thread: " + self.name)

        # Get the FITS files list
        i = 0
        for kepid in self.kepid_list:
            if i and not i % 10:
                mutex.acquire()
                save_file_list()
                print("\n{} saved data to file".format(self.name))
                mutex.release()
                print("{} queried {}/{}, totally {}\n".format(self.name, i, self.num_kepids, total_num_kepids))

            if kepid_to_file_list[kepid] == []:
                kepid_formatted = "{0:09d}".format(int(kepid))  # Pad with zeros.
                subdir = "{}/{}".format(kepid_formatted[0:4], kepid_formatted)

                url = "{}/{}/".format(environment.BASE_URL, subdir)

                try:
                    print("{} is querying {}".format(self.name, url))
                    page = requests.get(url).text
                    # print page
                    soup = BeautifulSoup(page, 'html.parser')

                    # files = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('fits')]
                    files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('_llc.fits')]
                    filesizes = [int(node[24:27]) for node in
                                 soup.find_all(text=re.compile("Long Cadence light curve file"))]

                    # Merged into this format:
                    # {'kplr007032116-2009166043257_llc.fits': 188, 'kplr007032116-2009259160929_llc.fits': 456, ...}
                    filename_with_size = dict(zip(files, filesizes))

                    # print(files)

                    mutex.acquire()
                    # Cannot use "kepid_formatted" here. Otherwise the dict will be added with new element
                    # kepid_to_file_list[kepid] = files
                    kepid_to_file_list[kepid] = filename_with_size
                    mutex.release()

                except URLError as e1:
                    print("{} getting {} failed.".format(self.name, url))
            else:
                print("{} already queried. Skip it.".format(kepid))

            i = i + 1

        print("\nTotally {} files download".format(self.num_kepids))

        print("Exit thread: " + self.name)


# ================================================================================
def main():
    # Read the stored record
    # Sometimes due to network issue, we may need to run this procedure several
    # times. So there could be some previous queried data stored.
    if os.path.isfile(KEPLER_FILE_LIST_TO_DOWNLOAD):
        with open(KEPLER_FILE_LIST_TO_DOWNLOAD) as data_file:
            data = json.load(data_file)

            if data:
                kepid_to_file_list = data

    # Read Kepler targets from the CSV file
    with open(environment.KEPLER_CSV_FILE) as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            key_id = row["kepid"]

            # If that kepid doesn't in the existing list, add it
            if not key_id in kepid_to_file_list.keys():
                kepid_to_file_list[row["kepid"]] = []

    total_num_kepids = len(kepid_to_file_list.keys())
    print("Total targets: {}".format(total_num_kepids))

    size_per_sublist = math.ceil(total_num_kepids/NUM_OF_THREADS)

    kepid_list_set = []

    for i in range(NUM_OF_THREADS):
        kepid_list_set.append([])

    # Get the key set
    kepid_keys = list(kepid_to_file_list)

    # Separate the total list to several sub-list, so as to be used
    # for multiple threads
    for i in range(total_num_kepids):
        list_index = math.trunc(i / size_per_sublist)
        if list_index > NUM_OF_THREADS-1:
            list_index = NUM_OF_THREADS-1
        kepid_list_set[list_index].append(kepid_keys[i])

    print(kepid_list_set)
    for i in range(NUM_OF_THREADS):
        print(len(kepid_list_set[i]))

    # Create multiple threads with sub-download list
    thread_set = [myThread(i+1, "Thread-{0:02d}".format(i+1), kepid_list_set[i]) for i in range(NUM_OF_THREADS)]

    for thread_process in thread_set:
        thread_process.start()

    for thread_process in thread_set:
        thread_process.join()

    print("\nExit the main process")

    save_file_list()


if __name__ == "__main__":
    main()
