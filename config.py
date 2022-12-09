class config2:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=8000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step = int(rate/1)
        self.model_path = '../working/models'
        self.p_path = '../working/pickle'