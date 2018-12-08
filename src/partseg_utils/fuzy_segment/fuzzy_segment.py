import numpy as np
from .fuzzy_segment_cython import compute_FDT, compute_FDT_with_move
from itertools import product
from queue import Queue

MaximumDistance = 65000

class FuzzySegment(object):
    def __init__(self):
        """
        suffix v - vessel
        suffix b - bone
        """
        super().__init__()
        self.image = np.zeros(1)
        self.FDT = np.zeros(1)
        self.tFDT = np.zeros(1)
        self.FDTv = np.zeros(1)
        self.FDTb = np.zeros(1)
        self.SS = np.zeros(1)
        self.LM = np.zeros(1)
        self.FCb = np.zeros(1)
        self.FCv = np.zeros(1)
        self.MRb = np.zeros(1)
        self.MRv = np.zeros(1)
        self.tap = 0
        self.ta = 0
        self.tb = 1
        self.tbp = 1
        self.Flag = 0
        self.GINT = 0


    def set_image(self, image):
        self.image = image
        self.FDT = np.zeros(image.shape, dtype=np.uint16)
        self.FDTv = np.zeros(image.shape, dtype=np.uint16)
        self.FDTb = np.zeros(image.shape, dtype=np.uint16)
        self.tFDT = np.zeros(image.shape, dtype=np.uint16)
        self.SS = np.zeros(image.shape, dtype=np.uint8)
        self.LM = np.zeros(image.shape+(3,), dtype=np.uint16)
        self.FCb = np.zeros(image.shape, dtype=np.uint16)
        self.FCv = np.zeros(image.shape, dtype=np.uint16)
        self.MRb = np.zeros(image.shape, dtype=np.int16)
        self.MRv = np.zeros(image.shape, dtype=np.int16)



    def first_run(self):
        # VESSEL
        interest_region = (self.image > self.tap ) * (self.image < self.tb)
        self.FDTv[interest_region] = MaximumDistance
        self.FDTv[~interest_region] = 0
        self.FDTv[self.SS == 2] = 0
        self.Flag = 2
        compute_FDT(self.image, self.FDTv, self.SS, self.tap, self.ta, self.tb, self.tbp, 2)
        # BONE
        interest_region = (self.image > self.ta ) * (self.image < self.tbp)
        self.FDTb[interest_region] = MaximumDistance
        self.FDTb[~interest_region] = 0
        self.FDTb[self.SS == 1] = 0
        self.FDTb[self.SS == 5] = 0
        compute_FDT(self.image, self.FDTb, self.SS, self.tap, self.ta, self.tb, self.tbp, 2)

        interest_region = (self.image > self.tap) * (self.image < self.tbp)
        self.FDT[interest_region] = MaximumDistance
        self.FDT[~interest_region] = 0
        compute_FDT(self.image, self.FDT, self.SS, self.tap, self.ta, self.tb, self.tbp, 2)

        self.local_maxima_and_scale()
        self.GINT = 0
        self.run_FCMR()

    def local_maxima(self):
        l = 2

        max_arr = np.zeros(self.FDT.shape, self.FDT.dtype)
        for shift in list(product(range(-l, l+1), repeat=3))[1:]:
            if shift == (0,0,0):
                continue
            max_arr = np.max(max_arr, np.roll(self.FDT, shift))
        cut_mask = np.ones(self.FDT.shape, np.bool)
        cut_mask[l:-l, l:-l,l:-l] = 0
        max_arr[cut_mask] = 0
        # Chyba niewrażliwość na szumy
        maxima_mask = self.FDT + 10 >= max_arr
        self.tFDT[maxima_mask] = 0
        # zapisanie współrzędnych tych punktów co są lokalnymi mmaksimami
        self.LM[maxima_mask] = np.transpose(np.nonzero(maxima_mask))

    def local_scale(self):
        compute_FDT_with_move(self.image, self.tFDT, self.SS, self.tap, self.ta, self.tb, self.tbp, 2, self.LM)

    def local_maxima_and_scale(self):
        interest_region = (self.image > self.tap) * (self.image < self.tbp)
        self.tFDT[interest_region] = MaximumDistance
        self.tFDT[~interest_region] = 0
        self.LM[:] = 0
        self.local_maxima()
        self.local_scale()

    def run_FCMR(self):
        VQ = Queue()
        BQ = Queue()
        if self.GINT == 0:
            self.FCb[:] = 0
            self.FCv[:] = 0
            self.MRb[:] = 0
            self.MRv[:] = 0

        while True:
            self.Flag = 0
            if self.GINT > 0:
                mask = self.MRv < self.MRb
                self.FCv[mask] = 100
                self.FCb[mask] = 0
                self.FDTb[mask] = 0




            if True:
                break
