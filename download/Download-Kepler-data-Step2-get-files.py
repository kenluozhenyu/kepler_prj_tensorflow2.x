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
# This file implements the "step 2"
#
# I tried to catch the download failure exception but it proved I cannot catch
# all. For those on-going download and failure seems no much good method to
# catch the exception. The thread would be killed in that case.
#
# So we may need to run this several times to re-download those failed files.
# The file size in local disk will be compared to the one online to verify if
# it is already completed downloaded.
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
NUM_OF_THREADS = 2000

# The file list and file size information is stored in this data structure.
# All threads will refer to this data for downloading.
kepid_to_file_list = {}
total_num_kepids = 0

mutex = threading.Lock()

# If we can catch the download failure exception, store the failed file here for reference
# We may not always be able to catch the exceptions. Sometimes the thread would just
# raise another exception and got killed.
download_failed_list = set()


class myThread(threading.Thread):
    def __init__(self, threadID, name, kepid_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.kepid_list = kepid_list
        self.num_kepids = len(kepid_list)

    def run(self):
        print("Start thread: " + self.name)

        # Download the FITS files
        i = 0
        for kepid in self.kepid_list:
            if i and not i % 10:
                print("{} processed {}/{}, totally {}\n".format(self.name, i, self.num_kepids, total_num_kepids))

            if kepid_to_file_list[kepid] != []:
                kepid_formatted = "{0:09d}".format(int(kepid))  # Pad with zeros.
                subdir = "{}/{}".format(kepid_formatted[0:4], kepid_formatted)

                strfolder = os.path.join(environment.KEPLER_DATA_FOLDER, kepid_formatted[0:4])

                mutex.acquire()

                if not os.path.exists(strfolder):
                    os.mkdir(strfolder)

                strfolder = os.path.join(environment.KEPLER_DATA_FOLDER, kepid_formatted[0:4], kepid_formatted)

                if not os.path.exists(strfolder):
                    os.mkdir(strfolder)

                mutex.release()

                download_dir = strfolder

                url = "{}/{}/".format(environment.BASE_URL, subdir)

                # data format is like this:
                # {'kplr007032116-2009166043257_llc.fits': 188, 'kplr007032116-2009259160929_llc.fits': 456, ...}
                # All size info is in "K" (bytes)
                # We don't have '*_llc.fits' file in M-byte format at this time
                files_with_size = kepid_to_file_list[kepid]

                # for file in files:
                for file, filesize in files_with_size.items():
                    file_url = url + '/' + file
                    dest_file = os.path.join(download_dir, file)

                    if not os.path.isfile(dest_file):
                        print("{} is downloading {} to {}".format(self.name, file_url, download_dir))
                        try:
                            urllib.request.urlretrieve(file_url, dest_file)
                        except URLError as e2:
                            print("\n{} ERROR => File {} download failed".format(self.name, file_url))

                            # Remove the file that is not completely downloaded
                            if os.path.isfile(dest_file):
                                os.remove(dest_file)

                            mutex.acquire()
                            download_failed_list.add(file_url)
                            mutex.release()
                    else:
                        # Check the downloaded file size
                        st = os.stat(dest_file)
                        size_KB = round(st.st_size / 1024)
                        # print(size_KB)

                        if size_KB < filesize:
                            # Previous download failed
                            # Need to re-download
                            print("{} is RE-downloading {} to {}".format(self.name, file_url, download_dir))

                            os.remove(dest_file)

                            try:
                                urllib.request.urlretrieve(file_url, dest_file)
                            except URLError as e2:
                                print("\n{} ERROR => File {} download failed".format(self.name, file_url))

                                # Remove the file that is not completely downloaded
                                if os.path.isfile(dest_file):
                                    os.remove(dest_file)

                                mutex.acquire()
                                download_failed_list.add(file_url)
                                mutex.release()

                        else:
                            # print("File already downloaded. Abort this one.")
                            pass

            i = i + 1
            # print("\n****** Number of running threads: {} ******\n".format(threading.active_count()))

        print("\nTotally {} targets downloaded".format(self.num_kepids))
        print("Exit thread: {}\n".format(self.name))


# ================================================================================
# main() function from here

def main():
    # Read the stored record
    if os.path.isfile(KEPLER_FILE_LIST_TO_DOWNLOAD):
        with open(KEPLER_FILE_LIST_TO_DOWNLOAD) as data_file:
            data = json.load(data_file)

            if data:
                kepid_to_file_list = data

        total_num_kepids = len(kepid_to_file_list.keys())
        print("Total targets: {}".format(total_num_kepids))

        size_per_sublist = math.ceil(total_num_kepids / NUM_OF_THREADS)

        kepid_list_set = []

        for i in range(NUM_OF_THREADS):
            kepid_list_set.append([])

        # Get the key set
        kepid_keys = list(kepid_to_file_list)

        # Separate the total list to several sub-list, so as to be used
        # for multiple threads
        for i in range(total_num_kepids):
            list_index = math.trunc(i / size_per_sublist)
            if list_index > NUM_OF_THREADS - 1:
                list_index = NUM_OF_THREADS - 1
            kepid_list_set[list_index].append(kepid_keys[i])

        print(kepid_list_set)
        for i in range(NUM_OF_THREADS):
            print(len(kepid_list_set[i]))

        thread_set = [myThread(i + 1, "Thread-{0:03d}".format(i + 1), kepid_list_set[i]) for i in range(NUM_OF_THREADS)]

        for thread_process in thread_set:
            thread_process.start()

        for thread_process in thread_set:
            thread_process.join()

        print("\nExit the main process")

        print("\n============== Failed to be downloaded:")
        print(download_failed_list)
    else:
        print("No download file list stored. Please run the (step 1) procedure first.")


if __name__ == "__main__":
    main()
