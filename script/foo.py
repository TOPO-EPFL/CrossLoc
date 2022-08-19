import time
import datetime


def main():
    start_time = time.time()
    counted_time = 0.0
    interval = 30.0

    while True:
        elapsed_time = time.time() - start_time  # in second

        print_flag = False
        if abs(elapsed_time % interval) < 1e-3 and (elapsed_time - counted_time) > 0.8 * interval:
            print_flag = True
        elif counted_time == 0:
            print_flag = True

        if print_flag:
            counted_time = elapsed_time
            hms = str(datetime.timedelta(seconds=int(elapsed_time)))
            print('Elapsed time: {:s}'.format(hms), flush=True)
            time.sleep(interval * 0.99)


if __name__ == '__main__':
    main()
