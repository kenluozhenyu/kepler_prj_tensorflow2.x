from data import training_data_io

class tce_struct:
    kepid = 0
    tce_plnt_num = 0
    tce_period = 0.0
    tce_time0bk = 0.0
    tce_duration = 0.0

def main():
    tce = tce_struct()

    tce.kepid = 1162345 # 757450
    tce.tce_plnt_num = 2
    tce.tce_period = 0.83185
    tce.tce_time0bk = 132.227
    tce.tce_duration = 2.392

    tce.tce_duration /= 24

    result_X = training_data_io.read_tce_global_view_local_view_from_file(tce)
    print(result_X)
    # read_tce_global_view_local_view_from_file(tce)

if __name__ == "__main__":
    main()
