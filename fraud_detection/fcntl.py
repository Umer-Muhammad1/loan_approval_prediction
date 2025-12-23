# Dummy fcntl.py for Windows
def fcntl(fd, op, arg=0):
    return 0

def ioctl(fd, op, arg=0, mutable_flag=True):
    return 0

def flock(fd, op):
    return 0

def lockf(fd, op, len=0, start=0, whence=0):
    return 0

# Common constants used by fcntl
F_GETFD = 1
F_SETFD = 2
F_GETFL = 3
F_SETFL = 4