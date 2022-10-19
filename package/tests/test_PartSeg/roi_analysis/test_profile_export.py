from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog, ProfileDictViewer


class TestImportDialog:
    def test_create(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ImportDialog(
            {lower_threshold_profile.name: lower_threshold_profile, border_rim_profile.name: border_rim_profile},
            {lower_threshold_profile.name: lower_threshold_profile},
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)

    def test_check(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ImportDialog(
            {lower_threshold_profile.name: lower_threshold_profile, border_rim_profile.name: border_rim_profile},
            {lower_threshold_profile.name: lower_threshold_profile},
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)
        assert set(dialog.get_import_list()) == {
            (border_rim_profile.name, border_rim_profile.name),
            (lower_threshold_profile.name, lower_threshold_profile.name),
        }
        dialog.uncheck_all()
        assert not dialog.get_import_list()
        dialog.check_all()
        assert set(dialog.get_import_list()) == {
            (border_rim_profile.name, border_rim_profile.name),
            (lower_threshold_profile.name, lower_threshold_profile.name),
        }

    def test_preview(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ImportDialog(
            {lower_threshold_profile.name: lower_threshold_profile, border_rim_profile.name: border_rim_profile},
            {lower_threshold_profile.name: lower_threshold_profile},
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)
        assert dialog.viewer.toPlainText() == ""
        dialog.list_view.setCurrentItem(dialog.list_view.topLevelItem(0))
        assert dialog.viewer.toPlainText() != ""
        dialog.list_view.setCurrentItem(dialog.list_view.topLevelItem(1))
        assert dialog.viewer.toPlainText() != ""

    def test_rename(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ImportDialog(
            {lower_threshold_profile.name: lower_threshold_profile, border_rim_profile.name: border_rim_profile},
            {lower_threshold_profile.name: lower_threshold_profile, border_rim_profile.name: border_rim_profile},
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)
        item = dialog.list_view.topLevelItem(0)
        assert dialog.list_view.itemWidget(item, 1).isChecked()
        dialog.list_view.itemWidget(item, 2).setChecked(True)
        assert not dialog.list_view.itemWidget(item, 1).isChecked()


class TestExportDialog:
    def test_create(self, qtbot, lower_threshold_profile):
        dialog = ExportDialog(
            export_dict={lower_threshold_profile.name: lower_threshold_profile}, viewer=ProfileDictViewer
        )
        qtbot.addWidget(dialog)

    def test_check(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ExportDialog(
            export_dict={
                lower_threshold_profile.name: lower_threshold_profile,
                border_rim_profile.name: border_rim_profile,
            },
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)
        assert set(dialog.get_checked()) == {lower_threshold_profile.name, border_rim_profile.name}
        dialog.uncheck_all()
        assert not dialog.get_checked()
        dialog.check_all()
        assert set(dialog.get_checked()) == {lower_threshold_profile.name, border_rim_profile.name}

    def test_preview(self, qtbot, lower_threshold_profile, border_rim_profile):
        dialog = ExportDialog(
            export_dict={
                lower_threshold_profile.name: lower_threshold_profile,
                border_rim_profile.name: border_rim_profile,
            },
            viewer=ProfileDictViewer,
        )
        qtbot.addWidget(dialog)
        assert dialog.viewer.toPlainText() == ""
        dialog.list_view.setCurrentRow(0)
        assert dialog.viewer.toPlainText() != ""
