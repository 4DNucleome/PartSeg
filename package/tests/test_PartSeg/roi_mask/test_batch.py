import pytest

from PartSeg._roi_mask.batch_proceed import BatchProceed, BatchTask


@pytest.fixture
def batch_task(stack_image, mask_threshold_profile):
    return BatchTask(stack_image, mask_threshold_profile, None)


class TestBatchProceed:
    def test_create(self, qapp):
        BatchProceed()

    def test_add_task(self, qapp, batch_task):
        thread = BatchProceed()
        thread.add_task(batch_task)
        assert thread.queue.qsize() == 1
        thread.add_task(batch_task)
        assert thread.queue.qsize() == 2
        thread.add_task([batch_task, batch_task])
        assert thread.queue.qsize() == 4

    def test_simple_run_task(self, qtbot, batch_task):
        thread = BatchProceed()
        thread.add_task(batch_task)
        assert thread.index == 0
        with qtbot.waitSignal(thread.multiple_result):
            thread.run_calculation()
