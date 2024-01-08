import time

start = time.perf_counter()


def do_something():
    print("Sleeping 1 second ...")
    time.sleep(1)
    print("Done Sleeping ... ")


finish = time.perf_counter()
