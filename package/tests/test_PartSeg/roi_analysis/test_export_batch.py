import tarfile
import zipfile

from unittest.mock import MagicMock, patch

import pytest

from PartSeg._roi_analysis.export_batch import (
    ExportProjectDialog,
    _extract_information_from_excel_to_export,
    export_to_archive,
)


@pytest.fixture()
def _dummy_tiffs(tmp_path):
    for i in range(1, 11):
        (tmp_path / f"stack1_component{i}.tif").touch()


class TestExportProjectDialog:
    def test_create(self, part_settings, qtbot):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        assert not dlg.export_btn.isEnabled()
        assert not dlg.export_to_zenodo_btn.isEnabled()

    @pytest.mark.usefixtures("_dummy_tiffs")
    def test_enable_export_btn(self, part_settings, qtbot, bundle_test_dir, tmp_path):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        assert not dlg.export_btn.isEnabled()

        dlg.excel_path.setText(str(bundle_test_dir / "sample_batch_output.xlsx"))
        assert not dlg.export_btn.isEnabled()

        assert dlg.info_box.topLevelItemCount() == 8

        dlg.base_folder.setText(str(tmp_path))
        assert dlg.export_btn.isEnabled()

    @pytest.mark.usefixtures("_dummy_tiffs")
    @patch("PartSeg._roi_analysis.export_batch.PSaveDialog")
    def test_export_to_zip(self, save_dlg_mock, part_settings, qtbot, bundle_test_dir, tmp_path, monkeypatch):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        assert not dlg.export_to_zenodo_btn.isEnabled()

        dlg.excel_path.setText(str(bundle_test_dir / "sample_batch_output.xlsx"))
        dlg.base_folder.setText(str(tmp_path))
        save_dlg_mock.exec_ = MagicMock(return_value=True)
        save_dlg_mock.selectedFiles = MagicMock(return_value=[str(tmp_path / "test.zip")])

        dlg.export_btn.click()
        qtbot.wait_until(lambda: dlg.worker is None)
        (tmp_path / "test.zip").exists()

    @pytest.mark.usefixtures("_dummy_tiffs")
    def test_enable_zenodo_export_btn(self, part_settings, qtbot, bundle_test_dir, tmp_path):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        assert not dlg.export_to_zenodo_btn.isEnabled()

        dlg.excel_path.setText(str(bundle_test_dir / "sample_batch_output.xlsx"))
        dlg.base_folder.setText(str(tmp_path))
        assert not dlg.export_to_zenodo_btn.isEnabled()

        dlg.zenodo_token.setText("test111")
        dlg.zenodo_author.setText("test")
        dlg.zenodo_affiliation.setText("test")
        dlg.zenodo_title.setText("test")
        assert not dlg.export_to_zenodo_btn.isEnabled()
        dlg.zenodo_description.setText("test")
        assert dlg.export_to_zenodo_btn.isEnabled()


def test_extract_information_from_excel_to_export_no_file(tmp_path, bundle_test_dir):
    res = _extract_information_from_excel_to_export(bundle_test_dir / "sample_batch_output.xlsx", tmp_path)
    assert all(not x[1] for x in res)


def test_extract_information_from_excel_to_export(data_test_dir, bundle_test_dir):
    res = _extract_information_from_excel_to_export(
        bundle_test_dir / "sample_batch_output.xlsx", data_test_dir / "stack1_components"
    )
    assert all(x[1] for x in res)


def all_files_in_dir(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zip_file:
            name_list = zip_file.namelist()
    else:
        with tarfile.open(path) as tar_file:
            name_list = tar_file.getnames()

    assert set(name_list) == {"sample_batch_output.xlsx", *[f"stack1_component{i}.tif" for i in range(1, 9)]}


@pytest.mark.usefixtures("_dummy_tiffs")
@pytest.mark.parametrize("ext", ".zip .tar.gz .txz .tar.bz2".split())
def test_export_to_archive(bundle_test_dir, tmp_path, ext):
    a = list(export_to_archive(bundle_test_dir / "sample_batch_output.xlsx", tmp_path, tmp_path / f"arch{ext}"))
    assert len(a) == 9
    all_files_in_dir(tmp_path / f"arch{ext}")
