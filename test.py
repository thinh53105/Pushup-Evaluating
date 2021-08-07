import threading
import time


def count(n, lock):
    for i in range(n):
        if i <= 3:
            lock.acquire()
            print(i + 1)
            time.sleep(1)
            lock.release()
            continue
        print(i + 1)
        time.sleep(1)


lock = threading.Lock()

t1 = threading.Thread(target=count, args=(10, lock))
t2 = threading.Thread(target=count, args=(5, lock))

t1.start()
t2.start()
t1.join()
t2.join()