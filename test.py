def logger(file):
    def log(msg):
        print("log")
        with open(file, 'a') as f:
            f.write(msg)
            f.write('\n')
    return log


log = logger("test.log")
log("test")
log("test2")
