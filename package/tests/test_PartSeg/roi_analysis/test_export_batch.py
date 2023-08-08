import tarfile
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from PartSeg._roi_analysis.export_batch import (
    ExportProjectDialog,
    _extract_information_from_excel_to_export,
    export_to_archive,
    export_to_zenodo,
)


@pytest.fixture()
def _dummy_tiffs(tmp_path):
    for i in range(1, 11):
        (tmp_path / f"stack1_component{i}.tif").touch()


def test_extract_information_from_excel_to_export_no_file(tmp_path, bundle_test_dir):
    res = _extract_information_from_excel_to_export(bundle_test_dir / "sample_batch_output.xlsx", tmp_path)
    assert all(not x[1] for x in res)


def test_extract_information_from_excel_to_export(data_test_dir, bundle_test_dir):
    res = _extract_information_from_excel_to_export(
        bundle_test_dir / "sample_batch_output.xlsx", data_test_dir / "stack1_components"
    )
    assert all(x[1] for x in res)


def all_files_in_archive(path):
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
    all_files_in_archive(tmp_path / f"arch{ext}")


@pytest.mark.usefixtures("_dummy_tiffs")
@patch("requests.post")
@patch("requests.put")
def test_zenodo_export(put_mock, post_mock, bundle_test_dir, tmp_path):
    post_mock.return_value.status_code = 201
    put_mock.return_value.status_code = 200
    a = list(
        export_to_zenodo(
            excel_path=bundle_test_dir / "sample_batch_output.xlsx",
            base_folder=tmp_path,
            zenodo_token="sample_token",  # noqa: S106
            title="sample title",
            author="sample author",
            affiliation="sample affiliation",
            orcid="0000-0000-0000-0000",
            description="sample description",
            zenodo_url="https://dummy_url",
        )
    )
    assert len(a) == 9
    assert post_mock.call_count == 1
    assert put_mock.call_count == 10


class TestExportProjectDialog:
    def test_create(self, part_settings, qtbot):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        assert not dlg.export_btn.isEnabled()
        assert not dlg.export_to_zenodo_btn.isEnabled()

    @pytest.mark.parametrize("ext", ["", ".xlsx"])
    def test_set_excel_file(self, monkeypatch, part_settings, qtbot, bundle_test_dir, ext):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        monkeypatch.setattr("PartSeg.common_gui.custom_load_dialog.PLoadDialog.exec_", lambda x: True)
        monkeypatch.setattr(
            "PartSeg.common_gui.custom_load_dialog.PLoadDialog.selectedFiles",
            lambda x: [str(bundle_test_dir / ("sample_batch_output" + ext))],
        )
        assert dlg.excel_path.text() == ""
        dlg.select_excel()
        assert dlg.excel_path.text() == str(bundle_test_dir / "sample_batch_output.xlsx")

    def test_set_base_folder(self, monkeypatch, part_settings, qtbot, bundle_test_dir):
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        monkeypatch.setattr("PartSeg.common_gui.custom_load_dialog.SelectDirectoryDialog.exec_", lambda x: True)
        monkeypatch.setattr(
            "PartSeg.common_gui.custom_load_dialog.SelectDirectoryDialog.selectedFiles",
            lambda x: [str(bundle_test_dir)],
        )
        assert dlg.base_folder.text() == ""
        dlg.select_folder()
        assert dlg.base_folder.text() == str(bundle_test_dir)

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

    @pytest.mark.usefixtures("_dummy_tiffs")
    @pytest.mark.parametrize(
        ("sandbox", "zenodo_url"), [(True, "https://sandbox.zenodo.org"), (False, "https://zenodo.org")]
    )
    @patch("requests.post")
    @patch("requests.put")
    def test_export_to_zenodo(
        self, put_mock, post_mock, part_settings, qtbot, bundle_test_dir, tmp_path, sandbox, zenodo_url
    ):
        post_mock.return_value.status_code = 201
        put_mock.return_value.status_code = 200
        dlg = ExportProjectDialog("", "", part_settings)
        qtbot.addWidget(dlg)
        dlg.excel_path.setText(str(bundle_test_dir / "sample_batch_output.xlsx"))
        dlg.base_folder.setText(str(tmp_path))
        dlg.zenodo_token.setText("test111")
        dlg.zenodo_author.setText("test1")
        dlg.zenodo_affiliation.setText("test2")
        dlg.zenodo_title.setText("test3")
        dlg.zenodo_description.setText("test4")
        dlg.zenodo_orcid.setText("0000-0000-0000-0001")
        dlg.zenodo_sandbox.setChecked(sandbox)

        dlg.export_to_zenodo_btn.click()
        qtbot.wait_until(lambda: dlg.worker is None)
        assert post_mock.call_count == 1
        assert post_mock.call_args[0][0].startswith(zenodo_url)
        assert put_mock.call_count == 10
        assert part_settings.get("zenodo_token") == "test111"
        assert part_settings.get("zenodo_author") == "test1"
        assert part_settings.get("zenodo_affiliation") == "test2"
        assert part_settings.get("zenodo_orcid") == "0000-0000-0000-0001"
