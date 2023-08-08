from PartSeg._roi_analysis.batch_window import (
    BatchWindow,
)


class TestBatchWindow:
    def test_create(self, part_settings):
        dlg = BatchWindow(part_settings)
        assert not dlg.is_working()
