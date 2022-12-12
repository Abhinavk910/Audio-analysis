class config_:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=8000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step = int(rate/1)
        self.classes = ['tired','burping','discomfort','belly_pain','hungry']
        self.max = 72.77787783
        self.min = -70.0754891
