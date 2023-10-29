class Logger:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        open(self.file_path, mode='w').write('')

    def log(self, cnt):
        prev_cnt = open(self.file_path, mode='r').read()
        prev_cnt += '\n' + cnt
        open(self.file_path, mode = 'w').write(prev_cnt)
