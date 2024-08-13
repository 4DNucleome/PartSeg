# Changelog

All notable changes to this project will be documented in this file.

## 0.15.4 - 2024-08-13

### 🚀 Features

- Add preview of image metadata ([#1154](https://github.com/4DNucleome/PartSeg/pull/1154))
- Add option to combine channels using sum and max ([#1159](https://github.com/4DNucleome/PartSeg/pull/1159))

### 🐛 Bug Fixes

- Fix selection of custom label colors for napari 0.5.0 ([#1138](https://github.com/4DNucleome/PartSeg/pull/1138))
- Add pint call to enforce initialization of unit registry ([#1146](https://github.com/4DNucleome/PartSeg/pull/1146))
- Workaround for lack of zsd support in czifile ([#1142](https://github.com/4DNucleome/PartSeg/pull/1142))
- Fix preparing data for `mahotas.haralick` to avoid overflow problem ([#1150](https://github.com/4DNucleome/PartSeg/pull/1150))
- Fix `use_convex` type from `int` to `bool` for segmentation algorithms ([#1152](https://github.com/4DNucleome/PartSeg/pull/1152))
- Prevent propagation of decreasing contrast limits set by user ([#1166](https://github.com/4DNucleome/PartSeg/pull/1166))
- Prevent error on searching component if there is no component ([#1167](https://github.com/4DNucleome/PartSeg/pull/1167))
- Fix checking if channel requested by MeasurementProfile exists ([#1165](https://github.com/4DNucleome/PartSeg/pull/1165))
- Fix trying to access to just deleted measurement profile from edit window. ([#1168](https://github.com/4DNucleome/PartSeg/pull/1168))
- Fix bug in code for checking for survey file ([#1174](https://github.com/4DNucleome/PartSeg/pull/1174))
- Fix plugin discovery in bundle to register them in napari viewer ([#1175](https://github.com/4DNucleome/PartSeg/pull/1175))
- Fix call of logging

### 📚 Documentation

- Change homepage URL ([#1139](https://github.com/4DNucleome/PartSeg/pull/1139))
- Add link for download macOS arm bundle ([#1140](https://github.com/4DNucleome/PartSeg/pull/1140))
- Add changelog for 0.15.4 release
- Update changelog ([#1176](https://github.com/4DNucleome/PartSeg/pull/1176))

### 🧪 Testing

- \[Automatic\] Constraints upgrades: `napari`, `sentry-sdk`, `sympy` ([#1128](https://github.com/4DNucleome/PartSeg/pull/1128))
- \[Automatic\] Constraints upgrades: `mahotas`, `numpy`, `sentry-sdk`, `sympy` ([#1145](https://github.com/4DNucleome/PartSeg/pull/1145))
- \[Automatic\] Constraints upgrades: `numpy`, `tifffile` ([#1163](https://github.com/4DNucleome/PartSeg/pull/1163))
- \[Automatic\] Constraints upgrades: `napari`, `sentry-sdk`, `tifffile` ([#1169](https://github.com/4DNucleome/PartSeg/pull/1169))
- \[Automatic\] Constraints upgrades: `magicgui`, `sentry-sdk` ([#1172](https://github.com/4DNucleome/PartSeg/pull/1172))
- \[Automatic\] Constraints upgrades: `sympy`, `tifffile` ([#1177](https://github.com/4DNucleome/PartSeg/pull/1177))

### ⚙️ Miscellaneous Tasks

- Speedup tests by use `tox-uv` ([#1141](https://github.com/4DNucleome/PartSeg/pull/1141))
- Get additional dict from PR branch for checking PR title ([#1144](https://github.com/4DNucleome/PartSeg/pull/1144))
- Relax numpy constraint ([#1143](https://github.com/4DNucleome/PartSeg/pull/1143))
- Allow to skip spellchecking PR title ([#1147](https://github.com/4DNucleome/PartSeg/pull/1147))
- \[pre-commit.ci\] pre-commit autoupdate ([#1149](https://github.com/4DNucleome/PartSeg/pull/1149))
- Create only archive with version in name on azures pipeline ([#1151](https://github.com/4DNucleome/PartSeg/pull/1151))
- Fix tests for napari from repository ([#1148](https://github.com/4DNucleome/PartSeg/pull/1148))
- Use python 3.11 to determine updated packages in PR description ([#1160](https://github.com/4DNucleome/PartSeg/pull/1160))
- \[pre-commit.ci\] pre-commit autoupdate ([#1164](https://github.com/4DNucleome/PartSeg/pull/1164))
- \[pre-commit.ci\] pre-commit autoupdate ([#1170](https://github.com/4DNucleome/PartSeg/pull/1170))
- Disable thumbnail generation in napari layer as it is fragile and not used ([#1171](https://github.com/4DNucleome/PartSeg/pull/1171))
- \[pre-commit.ci\] pre-commit autoupdate ([#1173](https://github.com/4DNucleome/PartSeg/pull/1173))
- \[pre-commit.ci\] pre-commit autoupdate ([#1178](https://github.com/4DNucleome/PartSeg/pull/1178))

### Build

- Remove PyOpenGL-accelerate from dependencies because of numpy incompatibility ([#1155](https://github.com/4DNucleome/PartSeg/pull/1155))
- Update install constraints on numpy and qt packages ([#1157](https://github.com/4DNucleome/PartSeg/pull/1157))
- Enforce napari 0.5.0 for Qt6 bindings ([#1161](https://github.com/4DNucleome/PartSeg/pull/1161))
- Require napari>=0.5.0 only for python 3.9+ ([#1162](https://github.com/4DNucleome/PartSeg/pull/1162))

## 0.15.3 - 2024-07-08

### 🚀 Features

- Pydantic 2 compatibility ([#1084](https://github.com/4DNucleome/PartSeg/pull/1084))

### 🐛 Bug Fixes

- Fix rendering icons in colormap preview ([#1040](https://github.com/4DNucleome/PartSeg/pull/1040))
- Fix test for validation length of message for sentry-sdk 2.0 release ([#1098](https://github.com/4DNucleome/PartSeg/pull/1098))
- When fix reader check lowercase extension for validate compatibility ([#1097](https://github.com/4DNucleome/PartSeg/pull/1097))
- Fix napari 0.5.0 compatibility ([#1116](https://github.com/4DNucleome/PartSeg/pull/1116))

### 🚜 Refactor

- Fix Qt flags ([#1041](https://github.com/4DNucleome/PartSeg/pull/1041))
- Fix qt flags in roi mask code ([#1042](https://github.com/4DNucleome/PartSeg/pull/1042))
- Fix qt flags in roi analysis ([#1043](https://github.com/4DNucleome/PartSeg/pull/1043))
- Migrate from setup.cfg to `pyproject.toml` ([#1070](https://github.com/4DNucleome/PartSeg/pull/1070))

### 📚 Documentation

- Allow to use newer release of build docs dependencies ([#1057](https://github.com/4DNucleome/PartSeg/pull/1057))

### 🧪 Testing

- \[Automatic\] Constraints upgrades: `imagecodecs`, `imageio`, `ipykernel`, `ipython`, `numpy`, `oiffile`, `pandas`, `psygnal`, `pyinstaller`, `qtconsole`, `qtpy`, `sentry-sdk`, `simpleitk`, `superqt`, `tifffile`, `xlsxwriter` ([#1020](https://github.com/4DNucleome/PartSeg/pull/1020))
- \[Automatic\] Constraints upgrades: `h5py`, `imageio`, `ipython`, `numpy`, `packaging`, `pydantic`, `pyinstaller`, `pyqt5`, `scipy`, `sentry-sdk`, `superqt`, `tifffile`, `xlsxwriter` ([#1027](https://github.com/4DNucleome/PartSeg/pull/1027))
- \[Automatic\] Constraints upgrades: `imageio`, `magicgui`, `xlsxwriter` ([#1030](https://github.com/4DNucleome/PartSeg/pull/1030))
- \[Automatic\] Constraints upgrades: `ipykernel`, `pandas`, `qtpy` ([#1032](https://github.com/4DNucleome/PartSeg/pull/1032))
- \[Automatic\] Constraints upgrades: `imageio`, `ipykernel`, `ipython`, `numpy`, `pandas`, `psygnal`, `pygments`, `pyinstaller`, `qtconsole`, `scipy`, `sentry-sdk`, `simpleitk` ([#1035](https://github.com/4DNucleome/PartSeg/pull/1035))
- \[Automatic\] Constraints upgrades: `imagecodecs`, `imageio`, `ipykernel`, `magicgui`, `pandas`, `pyinstaller`, `qtawesome`, `sentry-sdk`, `tifffile` ([#1048](https://github.com/4DNucleome/PartSeg/pull/1048))
- \[Automatic\] Constraints upgrades: `ipykernel`, `numpy`, `pandas`, `partsegcore-compiled-backend`, `pydantic`, `scipy`, `sentry-sdk` ([#1058](https://github.com/4DNucleome/PartSeg/pull/1058))
- Improve test of PartSegImage ([#1072](https://github.com/4DNucleome/PartSeg/pull/1072))
- Improve test suite for `PartSegCore` ([#1077](https://github.com/4DNucleome/PartSeg/pull/1077))
- \[Automatic\] Constraints upgrades: `imageio`, `ipykernel`, `local-migrator`, `napari`, `numpy`, `pandas`, `partsegcore-compiled-backend`, `pyinstaller`, `sentry-sdk`, `tifffile`, `vispy`, `xlsxwriter` ([#1063](https://github.com/4DNucleome/PartSeg/pull/1063))
- \[Automatic\] Constraints upgrades: `magicgui`, `packaging`, `psygnal`, `pyinstaller`, `sentry-sdk`, `superqt` ([#1086](https://github.com/4DNucleome/PartSeg/pull/1086))
- \[Automatic\] Constraints upgrades: `psygnal`, `pydantic`, `sentry-sdk`, `vispy` ([#1090](https://github.com/4DNucleome/PartSeg/pull/1090))
- \[Automatic\] Constraints upgrades: `h5py`, `ipykernel`, `mahotas`, `pandas`, `psygnal`, `pydantic`, `pyinstaller`, `qtawesome`, `scipy`, `sentry-sdk`, `superqt` ([#1092](https://github.com/4DNucleome/PartSeg/pull/1092))
- \[Automatic\] Constraints upgrades: `imageio`, `tifffile` ([#1100](https://github.com/4DNucleome/PartSeg/pull/1100))
- \[Automatic\] Constraints upgrades: `pydantic`, `sentry-sdk`, `superqt`, `tifffile` ([#1102](https://github.com/4DNucleome/PartSeg/pull/1102))
- \[Automatic\] Constraints upgrades: `psygnal`, `pygments`, `qtconsole`, `sentry-sdk`, `superqt`, `tifffile` ([#1105](https://github.com/4DNucleome/PartSeg/pull/1105))
- \[Automatic\] Constraints upgrades: `imagecodecs`, `magicgui`, `oiffile`, `openpyxl`, `packaging`, `pydantic`, `pyinstaller`, `requests`, `scipy`, `sentry-sdk`, `superqt`, `sympy`, `tifffile`, `vispy` ([#1107](https://github.com/4DNucleome/PartSeg/pull/1107))
- \[Automatic\] Constraints upgrades: `pydantic` ([#1112](https://github.com/4DNucleome/PartSeg/pull/1112))

### ⚙️ Miscellaneous Tasks

- \[pre-commit.ci\] pre-commit autoupdate ([#1019](https://github.com/4DNucleome/PartSeg/pull/1019))
- Remove plugin page preview as it is no longer maintained ([#1021](https://github.com/4DNucleome/PartSeg/pull/1021))
- \[pre-commit.ci\] pre-commit autoupdate ([#1022](https://github.com/4DNucleome/PartSeg/pull/1022))
- \[pre-commit.ci\] pre-commit autoupdate ([#1026](https://github.com/4DNucleome/PartSeg/pull/1026))
- \[pre-commit.ci\] pre-commit autoupdate ([#1031](https://github.com/4DNucleome/PartSeg/pull/1031))
- \[pre-commit.ci\] pre-commit autoupdate ([#1034](https://github.com/4DNucleome/PartSeg/pull/1034))
- Use new semgrep configuration ([#1039](https://github.com/4DNucleome/PartSeg/pull/1039))
- Upload raw coverage information ([#1044](https://github.com/4DNucleome/PartSeg/pull/1044))
- \[pre-commit.ci\] pre-commit autoupdate ([#1036](https://github.com/4DNucleome/PartSeg/pull/1036))
- Run coverage upload in separate steep ([#1053](https://github.com/4DNucleome/PartSeg/pull/1053))
- Generate local report in `Tests` workflow and use proper script for fetch report ([#1054](https://github.com/4DNucleome/PartSeg/pull/1054))
- Move coverage back to main workflow ([#1055](https://github.com/4DNucleome/PartSeg/pull/1055))
- \[pre-commit.ci\] pre-commit autoupdate ([#1056](https://github.com/4DNucleome/PartSeg/pull/1056))
- \[pre-commit.ci\] pre-commit autoupdate ([#1059](https://github.com/4DNucleome/PartSeg/pull/1059))
- Update `actions/upload-artifact` and  `actions/download-artifact` from 3 to 4 ([#1062](https://github.com/4DNucleome/PartSeg/pull/1062))
- \[pre-commit.ci\] pre-commit autoupdate ([#1064](https://github.com/4DNucleome/PartSeg/pull/1064))
- Group actions update ([#1065](https://github.com/4DNucleome/PartSeg/pull/1065))
- \[pre-commit.ci\] pre-commit autoupdate ([#1068](https://github.com/4DNucleome/PartSeg/pull/1068))
- Remove requirement of 2 builds upload to codecov.io ([#1073](https://github.com/4DNucleome/PartSeg/pull/1073))
- Re add tests to coverage report ([#1074](https://github.com/4DNucleome/PartSeg/pull/1074))
- Switch from setup.cfg to pyproject.toml in workflows ([#1076](https://github.com/4DNucleome/PartSeg/pull/1076))
- Fix compiling pyinstaller pre-deps ([#1075](https://github.com/4DNucleome/PartSeg/pull/1075))
- Add codespell to pre-commit and fix pointed bugs ([#1078](https://github.com/4DNucleome/PartSeg/pull/1078))
- Add new ruff rules and apply them ([#1079](https://github.com/4DNucleome/PartSeg/pull/1079))
- \[pre-commit.ci\] pre-commit autoupdate ([#1080](https://github.com/4DNucleome/PartSeg/pull/1080))
- \[pre-commit.ci\] pre-commit autoupdate ([#1081](https://github.com/4DNucleome/PartSeg/pull/1081))
- Fix upgrade depenecies workflow ([#1083](https://github.com/4DNucleome/PartSeg/pull/1083))
- Block using `mpmath==1.4.0a0` and `sentry-sdk` 2.0.0a1/a2 in pre-test ([#1085](https://github.com/4DNucleome/PartSeg/pull/1085))
- \[pre-commit.ci\] pre-commit autoupdate ([#1089](https://github.com/4DNucleome/PartSeg/pull/1089))
- Fix jupyter failing test by using constraints ([#1093](https://github.com/4DNucleome/PartSeg/pull/1093))
- \[pre-commit.ci\] pre-commit autoupdate ([#1091](https://github.com/4DNucleome/PartSeg/pull/1091))
- \[pre-commit.ci\] pre-commit autoupdate ([#1096](https://github.com/4DNucleome/PartSeg/pull/1096))
- Add python 3.12 testing ([#1087](https://github.com/4DNucleome/PartSeg/pull/1087))
- Exclude pyside2 on python 3.11 and 3.12 from testing ([#1099](https://github.com/4DNucleome/PartSeg/pull/1099))
- \[pre-commit.ci\] pre-commit autoupdate ([#1101](https://github.com/4DNucleome/PartSeg/pull/1101))
- \[pre-commit.ci\] pre-commit autoupdate ([#1103](https://github.com/4DNucleome/PartSeg/pull/1103))
- Bump macos runners to macos-13 (both azure and GHA) ([#1113](https://github.com/4DNucleome/PartSeg/pull/1113))
- \[pre-commit.ci\] pre-commit autoupdate ([#1108](https://github.com/4DNucleome/PartSeg/pull/1108))
- Remove pyqt5 from constraints ([#1118](https://github.com/4DNucleome/PartSeg/pull/1118))
- Add workflow for releases from GHA ([#1117](https://github.com/4DNucleome/PartSeg/pull/1117))
- Add actionlint to CI to early prevent bug in github workflows ([#1119](https://github.com/4DNucleome/PartSeg/pull/1119))
- Fix release workflow, by update permissions
- Check if release notes are properly created ([#1122](https://github.com/4DNucleome/PartSeg/pull/1122))
- Proper use enum in checking new version ([#1123](https://github.com/4DNucleome/PartSeg/pull/1123))
- Refactor and simplify menu bar creation, add workaround for macOS numpy problem ([#1124](https://github.com/4DNucleome/PartSeg/pull/1124))
- Simplify release workflow ([#1126](https://github.com/4DNucleome/PartSeg/pull/1126))
- Fix `make_release.yml` to proper detect release, attempt 3 ([#1127](https://github.com/4DNucleome/PartSeg/pull/1127))

### 🛡️ Security

- *(deps)* Bump actions/checkout from 3 to 4 ([#1029](https://github.com/4DNucleome/PartSeg/pull/1029))
- *(deps)* Bump conda-incubator/setup-miniconda from 2 to 3 ([#1038](https://github.com/4DNucleome/PartSeg/pull/1038))
- *(deps)* Bump aganders3/headless-gui from 1 to 2 ([#1047](https://github.com/4DNucleome/PartSeg/pull/1047))
- *(deps)* Bump actions/checkout from 3 to 4 ([#1045](https://github.com/4DNucleome/PartSeg/pull/1045))
- *(deps)* Bump hynek/build-and-inspect-python-package from 1 to 2 ([#1050](https://github.com/4DNucleome/PartSeg/pull/1050))
- *(deps)* Bump actions/setup-python from 4 to 5 ([#1046](https://github.com/4DNucleome/PartSeg/pull/1046))
- *(deps)* Bump github/codeql-action from 2 to 3 ([#1051](https://github.com/4DNucleome/PartSeg/pull/1051))
- *(deps)* Bump peter-evans/create-pull-request from 5 to 6 ([#1067](https://github.com/4DNucleome/PartSeg/pull/1067))
- *(deps)* Bump codecov/codecov-action from 3 to 4 ([#1066](https://github.com/4DNucleome/PartSeg/pull/1066))

### Build

- Fix not bundling `Font Awesome 6 Free-Solid-900.otf` file to executable ([#1114](https://github.com/4DNucleome/PartSeg/pull/1114))
- Update readme and release to point to GitHub releases ([#1115](https://github.com/4DNucleome/PartSeg/pull/1115))
- Do not create archive twice when create bundle ([#1120](https://github.com/4DNucleome/PartSeg/pull/1120))
- Enable macOS-arm bundle builds ([#1121](https://github.com/4DNucleome/PartSeg/pull/1121))

## 0.15.2 - 2023-08-28

### 🐛 Bug Fixes

- Fix range threshold selection of algorithms ([#1009](https://github.com/4DNucleome/PartSeg/pull/1009))
- When run batch check if file extension is supported by loader ([#1016](https://github.com/4DNucleome/PartSeg/pull/1016))
- Do not allow to select and render corrupted batch plans ([#1015](https://github.com/4DNucleome/PartSeg/pull/1015))

### 🧪 Testing

- \[Automatic\] Constraints upgrades: `imagecodecs`, `ipykernel`, `magicgui`, `psygnal`, `scipy`, `superqt`, `tifffile` ([#1011](https://github.com/4DNucleome/PartSeg/pull/1011))
- \[Automatic\] Constraints upgrades: `imageio`, `pyinstaller`, `tifffile` ([#1018](https://github.com/4DNucleome/PartSeg/pull/1018))

### ⚙️ Miscellaneous Tasks

- Use faster version of black ([#1010](https://github.com/4DNucleome/PartSeg/pull/1010))
- \[pre-commit.ci\] pre-commit autoupdate ([#1013](https://github.com/4DNucleome/PartSeg/pull/1013))

## 0.15.1 - 2023-08-08

### 🚀 Features

- Allow to save multiple napari image layers to single tiff file ([#1000](https://github.com/4DNucleome/PartSeg/pull/1000))
- Add option to export batch project with data ([#996](https://github.com/4DNucleome/PartSeg/pull/996))

### 🐛 Bug Fixes

- Fix possible problem of double registration napari plugin in PartSeg bundle ([#974](https://github.com/4DNucleome/PartSeg/pull/974))
- Bump OS versions for part of testing workflows.  ([#977](https://github.com/4DNucleome/PartSeg/pull/977))
- Bump os version for main tests workflow. ([#979](https://github.com/4DNucleome/PartSeg/pull/979))
- Ensure that the module `PartSegCore.channel_class` is present in bundle ([#980](https://github.com/4DNucleome/PartSeg/pull/980))
- Lower npe2 schema version to work with older napari version ([#981](https://github.com/4DNucleome/PartSeg/pull/981))
- Generate test report per platform ([#978](https://github.com/4DNucleome/PartSeg/pull/978))
- Importing plugins in bundle keeping proper module names ([#983](https://github.com/4DNucleome/PartSeg/pull/983))
- Fix napari repo workflow ([#985](https://github.com/4DNucleome/PartSeg/pull/985))
- Fix bug in read tiff files with double `Q` in axes but one related to dummy dimension ([#992](https://github.com/4DNucleome/PartSeg/pull/992))
- Fix bug that lead to corrupted state when saving calculation plan to excel file ([#995](https://github.com/4DNucleome/PartSeg/pull/995))
- Enable python 3.11 test on CI, fix minor errors ([#869](https://github.com/4DNucleome/PartSeg/pull/869))

### 🧪 Testing

- \[Automatic\] Constraints upgrades: `imageio`, `ipython`, `psygnal`, `scipy`, `sentry-sdk` ([#975](https://github.com/4DNucleome/PartSeg/pull/975))
- \[Automatic\] Constraints upgrades: `h5py`, `imagecodecs`, `imageio`, `ipykernel`, `napari`, `numpy`, `pandas`, `pydantic`, `pyinstaller`, `scipy`, `sentry-sdk`, `tifffile`, `vispy` ([#986](https://github.com/4DNucleome/PartSeg/pull/986))
- \[Automatic\] Constraints upgrades: `imagecodecs`, `sentry-sdk`, `tifffile` ([#997](https://github.com/4DNucleome/PartSeg/pull/997))
- \[Automatic\] Constraints upgrades: `ipykernel`, `pydantic` ([#1002](https://github.com/4DNucleome/PartSeg/pull/1002))
- \[Automatic\] Constraints upgrades: `numpy`, `pygments`, `sentry-sdk`, `superqt` ([#1007](https://github.com/4DNucleome/PartSeg/pull/1007))

### ⚙️ Miscellaneous Tasks

- \[pre-commit.ci\] pre-commit autoupdate ([#973](https://github.com/4DNucleome/PartSeg/pull/973))
- \[pre-commit.ci\] pre-commit autoupdate ([#982](https://github.com/4DNucleome/PartSeg/pull/982))
- \[pre-commit.ci\] pre-commit autoupdate ([#987](https://github.com/4DNucleome/PartSeg/pull/987))
- \[pre-commit.ci\] pre-commit autoupdate ([#988](https://github.com/4DNucleome/PartSeg/pull/988))
- \[pre-commit.ci\] pre-commit autoupdate ([#991](https://github.com/4DNucleome/PartSeg/pull/991))
- \[pre-commit.ci\] pre-commit autoupdate ([#998](https://github.com/4DNucleome/PartSeg/pull/998))
- \[pre-commit.ci\] pre-commit autoupdate ([#1004](https://github.com/4DNucleome/PartSeg/pull/1004))
- Change markdown linter from pre-commit to mdformat ([#1006](https://github.com/4DNucleome/PartSeg/pull/1006))
- \[pre-commit.ci\] pre-commit autoupdate ([#1008](https://github.com/4DNucleome/PartSeg/pull/1008))

## 0.15.0 - 2023-05-30

### 🚀 Features

- Add `PARTSEG_SENTRY_URL` env variable support and basic documentation about error reporting ([#802](https://github.com/4DNucleome/PartSeg/pull/802))
- Allow to see underlying exception when show warning caused by exception ([#829](https://github.com/4DNucleome/PartSeg/pull/829))
- Add voxel size measurement and allow to overwrite voxel size in batch ([#853](https://github.com/4DNucleome/PartSeg/pull/853))
- Add alpha support for Qt6 ([#866](https://github.com/4DNucleome/PartSeg/pull/866))
- Add option to create projection alongside z-axis ([#919](https://github.com/4DNucleome/PartSeg/pull/919))
- Add napari image custom representation for better error report via sentry ([#861](https://github.com/4DNucleome/PartSeg/pull/861))
- Add import and export operation for labels and colormaps ([#936](https://github.com/4DNucleome/PartSeg/pull/936))
- Implement napari widgets for colormap and labels control ([#935](https://github.com/4DNucleome/PartSeg/pull/935))
- Add forget all button to multiple files widget ([#942](https://github.com/4DNucleome/PartSeg/pull/942))
- Do not abort processing whole mask segmentation project during exception on single component ([#943](https://github.com/4DNucleome/PartSeg/pull/943))
- Add distance based watersheed to flow methods ([#915](https://github.com/4DNucleome/PartSeg/pull/915))
- Add napari widgets for all group of algorithms ([#958](https://github.com/4DNucleome/PartSeg/pull/958))
- Add napari widget to copy labels along z-axis ([#968](https://github.com/4DNucleome/PartSeg/pull/968))

### 🐛 Bug Fixes

- Print all exceptions instead of the latest one in exception dialog ([#799](https://github.com/4DNucleome/PartSeg/pull/799))
- Fix ROIExtractionResult `__str__`and `__repr__` to use `ROIExtractionResult` not `SegmentationResult` ([#810](https://github.com/4DNucleome/PartSeg/pull/810))
- Fix code to address changes in napari repository ([#817](https://github.com/4DNucleome/PartSeg/pull/817))
- Fix problem with resize of multiline widgets ([#832](https://github.com/4DNucleome/PartSeg/pull/832))
- Fix tox configuration to run all required tests ([#840](https://github.com/4DNucleome/PartSeg/pull/840))
- Fix MSO `step_limit` description in GUI ([#843](https://github.com/4DNucleome/PartSeg/pull/843))
- Fix `redefined-while-unused`import code for python 3.9.7 ([#844](https://github.com/4DNucleome/PartSeg/pull/844))
- Fix warnings reported by Deepsource ([#846](https://github.com/4DNucleome/PartSeg/pull/846))
- Ensure that "ROI" layer is in proper place for proper visualization ([#856](https://github.com/4DNucleome/PartSeg/pull/856))
- Fix tests of napari widgets ([#862](https://github.com/4DNucleome/PartSeg/pull/862))
- Fix build of bundle for a new psygnal release ([#863](https://github.com/4DNucleome/PartSeg/pull/863))
- Fix minimal requirements pipeline ([#877](https://github.com/4DNucleome/PartSeg/pull/877))
- Update pyinstaller configuration ([#926](https://github.com/4DNucleome/PartSeg/pull/926))
- Use text icon, not pixmap icon in colormap and labels list ([#938](https://github.com/4DNucleome/PartSeg/pull/938))
- Resolve warnings when testing custom save dialog. ([#941](https://github.com/4DNucleome/PartSeg/pull/941))
- Add padding zeros for component num when load Mask seg file to ROI GUI ([#944](https://github.com/4DNucleome/PartSeg/pull/944))
- Proper calculate bounds for watershed napari widget ([#969](https://github.com/4DNucleome/PartSeg/pull/969))
- Fix bug in the wrong order of axis saved in napari contribution ([#972](https://github.com/4DNucleome/PartSeg/pull/972))

### 🚜 Refactor

- Simplify and refactor github workflows. ([#864](https://github.com/4DNucleome/PartSeg/pull/864))
- Better load Mask project in Roi Analysis ([#921](https://github.com/4DNucleome/PartSeg/pull/921))
- Use more descriptive names in `pylint: disable` ([#922](https://github.com/4DNucleome/PartSeg/pull/922))
- Remove `pkg_resources` usage as it is deprecated ([#967](https://github.com/4DNucleome/PartSeg/pull/967))
- Convert napari plugin to npe2 ([#966](https://github.com/4DNucleome/PartSeg/pull/966))

### 📚 Documentation

- Update README and project metadata ([#805](https://github.com/4DNucleome/PartSeg/pull/805))
- Create release notes for PartSeg 0.15.0 ([#971](https://github.com/4DNucleome/PartSeg/pull/971))

### 🎨 Styling

- Change default theme to dark, remove blinking windows on startup. ([#809](https://github.com/4DNucleome/PartSeg/pull/809))

### 🧪 Testing

- \[Automatic\] Dependency upgrades: `packaging`, `pyinstaller`, `pyopengl-accelerate`, `tifffile`, `xlsxwriter` ([#932](https://github.com/4DNucleome/PartSeg/pull/932))
- \[Automatic\] Constraints upgrades: `fonticon-fontawesome6`, `imageio`, `numpy`, `partsegcore-compiled-backend`, `pygments`, `sentry-sdk` ([#937](https://github.com/4DNucleome/PartSeg/pull/937))
- \[Automatic\] Constraints upgrades: `imageio`, `ipython`, `pandas`, `requests`, `sentry-sdk` ([#948](https://github.com/4DNucleome/PartSeg/pull/948))
- \[Automatic\] Constraints upgrades: `ipython`, `nme`, `qtconsole`, `requests`, `sentry-sdk` ([#955](https://github.com/4DNucleome/PartSeg/pull/955))
- \[Automatic\] Constraints upgrades: `ipykernel`, `local-migrator`, `pyinstaller`, `sentry-sdk`, `sympy` ([#957](https://github.com/4DNucleome/PartSeg/pull/957))
- \[Automatic\] Constraints upgrades: `sentry-sdk`, `xlsxwriter` ([#959](https://github.com/4DNucleome/PartSeg/pull/959))
- \[Automatic\] Constraints upgrades: `requests` ([#961](https://github.com/4DNucleome/PartSeg/pull/961))
- \[Automatic\] Constraints upgrades: `imageio`, `pandas`, `pydantic`, `pyopengl-accelerate`, `sentry-sdk`, `xlsxwriter` ([#970](https://github.com/4DNucleome/PartSeg/pull/970))

### ⚙️ Miscellaneous Tasks

- Improve ruff configuration, remove isort ([#815](https://github.com/4DNucleome/PartSeg/pull/815))
- Use `fail_on_no_env` feature from `tox-gh-actions` ([#842](https://github.com/4DNucleome/PartSeg/pull/842))
- Add python 3.11 to list of supported versions ([#867](https://github.com/4DNucleome/PartSeg/pull/867))
- Disable python 3.11 test because of timeout ([#870](https://github.com/4DNucleome/PartSeg/pull/870))
- Bump ruff to 0.0.218, remove flake8 from pre-commit ([#880](https://github.com/4DNucleome/PartSeg/pull/880))
- Replace GabrielBB/xvfb-action@v1 by aganders3/headless-gui, part 2 ([#887](https://github.com/4DNucleome/PartSeg/pull/887))
- Better minimal requirements test ([#888](https://github.com/4DNucleome/PartSeg/pull/888))
- Improve regexp for proper generate list of packages in update report ([#894](https://github.com/4DNucleome/PartSeg/pull/894))
- Add check for PR title ([#933](https://github.com/4DNucleome/PartSeg/pull/933))
- Update codecov configuration to wait on two reports before post information ([#934](https://github.com/4DNucleome/PartSeg/pull/934))
- \[pre-commit.ci\] pre-commit autoupdate ([#945](https://github.com/4DNucleome/PartSeg/pull/945))
- Migrate from `nme` to `local_migrator` ([#951](https://github.com/4DNucleome/PartSeg/pull/951))
- \[pre-commit.ci\] pre-commit autoupdate ([#956](https://github.com/4DNucleome/PartSeg/pull/956))
- \[pre-commit.ci\] pre-commit autoupdate ([#964](https://github.com/4DNucleome/PartSeg/pull/964))

### 🛡️ Security

- *(deps)* Bump peter-evans/create-pull-request from 4 to 5 ([#928](https://github.com/4DNucleome/PartSeg/pull/928))

### Bugfix

- Fix bug with generation of form for model with hidden field ([#920](https://github.com/4DNucleome/PartSeg/pull/920))

### Dep

- \[Automatic\] Dependency upgrades ([#824](https://github.com/4DNucleome/PartSeg/pull/824))
- \[Automatic\] Dependency upgrades ([#828](https://github.com/4DNucleome/PartSeg/pull/828))
- \[Automatic\] Dependency upgrades: `ipykernel`, `packaging` ([#838](https://github.com/4DNucleome/PartSeg/pull/838))
- \[Automatic\] Dependency upgrades: `imageio`, `ipykernel`, `napari`, `numpy`, `sentry` ([#850](https://github.com/4DNucleome/PartSeg/pull/850))
- \[Automatic\] Dependency upgrades: `imagecodecs`, `ipykernel`, `numpy`, `psygnal` ([#859](https://github.com/4DNucleome/PartSeg/pull/859))
- \[Automatic\] Dependency upgrades: `pydantic`, `pygments`, `xlsxwriter` ([#874](https://github.com/4DNucleome/PartSeg/pull/874))
- \[Automatic\] Dependency upgrades: `imageio`, `packaging`, `scipy`, `xlsxwriter` ([#878](https://github.com/4DNucleome/PartSeg/pull/878))
- \[Automatic\] Dependency upgrades: `ipykernel`, `requests`, `sentry`, `xlsxwriter` ([#884](https://github.com/4DNucleome/PartSeg/pull/884))
- \[Automatic\] Dependency upgrades: `h5py`, `imagecodecs`, `imageio`, `ipykernel`, `pandas`, `sentry`, `tifffile` ([#889](https://github.com/4DNucleome/PartSeg/pull/889))
- \[Automatic\] Dependency upgrades: `ipython`, `pyqt5` ([#893](https://github.com/4DNucleome/PartSeg/pull/893))
- \[Automatic\] Dependency upgrades: `imageio`, `ipykernel`, `ipython`, `numpy`, `openpyxl`, `psygnal`, `pydantic`, `pyinstaller`, `pyqt5`, `scipy`, `sentry-sdk`, `tifffile`, `xlsxwriter` ([#897](https://github.com/4DNucleome/PartSeg/pull/897))
- \[Automatic\] Dependency upgrades: `imageio`, `psygnal` ([#905](https://github.com/4DNucleome/PartSeg/pull/905))
- \[Automatic\] Dependency upgrades: `ipython`, `magicgui`, `scipy`, `sentry-sdk`, `tifffile` ([#906](https://github.com/4DNucleome/PartSeg/pull/906))
- \[Automatic\] Dependency upgrades: `imagecodecs`, `imageio`, `ipykernel`, `openpyxl`, `pydantic`, `pyinstaller`, `qtawesome`, `qtconsole`, `sentry-sdk`, `tifffile`, `xlsxwriter` ([#908](https://github.com/4DNucleome/PartSeg/pull/908))
- \[Automatic\] Dependency upgrades: `imageio`, `ipykernel`, `ipython`, `pandas`, `psygnal`, `pydantic`, `pygments`, `pyinstaller`, `qtpy`, `sentry-sdk`, `tifffile` ([#917](https://github.com/4DNucleome/PartSeg/pull/917))

## 0.14.6 - 2022-11-13

### 🐛 Bug Fixes

- Fix bug when loading already created project causing hide of ROI layer ([#787](https://github.com/4DNucleome/PartSeg/pull/787))

## 0.14.5 - 2022-11-09

### 🚀 Features

- Add option for ensure type in EventedDict and use it to validate profiles structures ([#776](https://github.com/4DNucleome/PartSeg/pull/776))
- Add option to create issue from error report dialog ([#782](https://github.com/4DNucleome/PartSeg/pull/782))
- Add option for multiline field in algorithm parameters ([#766](https://github.com/4DNucleome/PartSeg/pull/766))

### 🐛 Bug Fixes

- Fix scalebar color ([#774](https://github.com/4DNucleome/PartSeg/pull/774))
- Fix bug when saving segmentation parameters in mask analysis ([#781](https://github.com/4DNucleome/PartSeg/pull/781))
- Fix multiple error related to loading new file in interactive mode ([#784](https://github.com/4DNucleome/PartSeg/pull/784))

### 🚜 Refactor

- Optimize CLI actions ([#772](https://github.com/4DNucleome/PartSeg/pull/772))
- Clean warnings about threshold methods ([#783](https://github.com/4DNucleome/PartSeg/pull/783))

### Build

- *(deps)* Bump chanzuckerberg/napari-hub-preview-action from 0.1.5 to 0.1.6 ([#775](https://github.com/4DNucleome/PartSeg/pull/775))

## 0.14.4 - 2022-10-24

### 🚀 Features

- Load alternatives labeling when open PartSeg projects in napari ([#731](https://github.com/4DNucleome/PartSeg/pull/731))
- Add option to toggle scale bar ([#733](https://github.com/4DNucleome/PartSeg/pull/733))
- Allow customize settings directory using the `PARTSEG_SETTINGS_DIR` environment variable ([#751](https://github.com/4DNucleome/PartSeg/pull/751))
- Separate recent algorithms from general application settings ([#752](https://github.com/4DNucleome/PartSeg/pull/752))
- Add multiple otsu as threshold method with selection range of components ([#710](https://github.com/4DNucleome/PartSeg/pull/710))
- Add function to load components from Mask Segmentation with background in ROI Analysis ([#768](https://github.com/4DNucleome/PartSeg/pull/768))

### 🐛 Bug Fixes

- Fix typos
- Fix `get_theme` calls to prepare for napari 0.4.17 ([#729](https://github.com/4DNucleome/PartSeg/pull/729))
- Fix saving pipeline from GUI ([#756](https://github.com/4DNucleome/PartSeg/pull/756))
- Fix profile export/import dialogs ([#761](https://github.com/4DNucleome/PartSeg/pull/761))
- Enable compare button if ROI is available ([#765](https://github.com/4DNucleome/PartSeg/pull/765))
- Fix bug in cut with roi to do not make black artifacts ([#767](https://github.com/4DNucleome/PartSeg/pull/767))

### 🧪 Testing

- Add new build and inspect wheel action ([#747](https://github.com/4DNucleome/PartSeg/pull/747))

### ⚙️ Miscellaneous Tasks

- Prepare pyinstaller configuration for napari 0.4.17 ([#748](https://github.com/4DNucleome/PartSeg/pull/748))
- Add ruff linter ([#754](https://github.com/4DNucleome/PartSeg/pull/754))

### Bugfix

- Fix sentry tests ([#742](https://github.com/4DNucleome/PartSeg/pull/742))
- Fix reporting error in load settings from drive ([#725](https://github.com/4DNucleome/PartSeg/pull/725))

### Build

- *(deps)* Bump actions/checkout from 2 to 3 ([#716](https://github.com/4DNucleome/PartSeg/pull/716))
- *(deps)* Bump actions/download-artifact from 1 to 3 ([#709](https://github.com/4DNucleome/PartSeg/pull/709))

## 0.14.3 - 2022-08-18

### 🐛 Bug Fixes

- Delay setting image if an algorithm is still running ([#627](https://github.com/4DNucleome/PartSeg/pull/627))
- Wrong error report when no component is found in restartable segmentation algorithm. ([#633](https://github.com/4DNucleome/PartSeg/pull/633))
- Fix process of build documentation ([#653](https://github.com/4DNucleome/PartSeg/pull/653))

### 🚜 Refactor

- Clean potential vulnerabilities  ([#630](https://github.com/4DNucleome/PartSeg/pull/630))

### 🧪 Testing

- Add more tests for common GUI elements  ([#622](https://github.com/4DNucleome/PartSeg/pull/622))
- Report coverage per package. ([#639](https://github.com/4DNucleome/PartSeg/pull/639))
- Update conda environment to not use PyQt5 in test ([#646](https://github.com/4DNucleome/PartSeg/pull/646))
- Add tests files to calculate coverage ([#655](https://github.com/4DNucleome/PartSeg/pull/655))

### Build

- *(deps)* Bump qtpy from 2.0.1 to 2.1.0 in /requirements ([#613](https://github.com/4DNucleome/PartSeg/pull/613))
- *(deps)* Bump pyinstaller from 5.0.1 to 5.1 in /requirements ([#629](https://github.com/4DNucleome/PartSeg/pull/629))
- *(deps)* Bump tifffile from 2022.4.28 to 2022.5.4 in /requirements ([#619](https://github.com/4DNucleome/PartSeg/pull/619))
- *(deps)* Bump codecov/codecov-action from 1 to 3 ([#637](https://github.com/4DNucleome/PartSeg/pull/637))
- *(deps)* Bump requests from 2.27.1 to 2.28.0 in /requirements ([#647](https://github.com/4DNucleome/PartSeg/pull/647))
- *(deps)* Bump actions/setup-python from 3 to 4 ([#648](https://github.com/4DNucleome/PartSeg/pull/648))
- *(deps)* Bump pyqt5 from 5.15.6 to 5.15.7 in /requirements ([#652](https://github.com/4DNucleome/PartSeg/pull/652))
- *(deps)* Bump sentry-sdk from 1.5.12 to 1.6.0 in /requirements ([#659](https://github.com/4DNucleome/PartSeg/pull/659))
- *(deps)* Bump numpy from 1.22.4 to 1.23.0 in /requirements ([#660](https://github.com/4DNucleome/PartSeg/pull/660))
- *(deps)* Bump lxml from 4.9.0 to 4.9.1 in /requirements ([#665](https://github.com/4DNucleome/PartSeg/pull/665))
- *(deps)* Bump mahotas from 1.4.12 to 1.4.13 in /requirements ([#662](https://github.com/4DNucleome/PartSeg/pull/662))
- *(deps)* Bump pyinstaller from 5.1 to 5.2 in /requirements ([#667](https://github.com/4DNucleome/PartSeg/pull/667))

## 0.14.2 - 2022-05-05

### 🐛 Bug Fixes

- Fix bug in save label colors between sessions ([#610](https://github.com/4DNucleome/PartSeg/pull/610))
- Register PartSeg plugins before start napari widgets. ([#611](https://github.com/4DNucleome/PartSeg/pull/611))
- Mouse interaction with components work again after highlight. ([#620](https://github.com/4DNucleome/PartSeg/pull/620))

### 🚜 Refactor

- Limit test run ([#603](https://github.com/4DNucleome/PartSeg/pull/603))
- Filter and solve warnings in tests ([#607](https://github.com/4DNucleome/PartSeg/pull/607))
- Use QAbstractSpinBox.AdaptiveDecimalStepType in SpinBox instead of hardcoded bounds ([#616](https://github.com/4DNucleome/PartSeg/pull/616))
- Clean and test `PartSeg.common_gui.universal_gui_part` ([#617](https://github.com/4DNucleome/PartSeg/pull/617))

### 📚 Documentation

- Update changelog ([#621](https://github.com/4DNucleome/PartSeg/pull/621))

### 🧪 Testing

- Speedup test by setup cache for pip ([#604](https://github.com/4DNucleome/PartSeg/pull/604))
- Setup cache for azure pipelines workflows ([#606](https://github.com/4DNucleome/PartSeg/pull/606))

### Build

- *(deps)* Bump sentry-sdk from 1.5.10 to 1.5.11 in /requirements ([#615](https://github.com/4DNucleome/PartSeg/pull/615))

## 0.14.1 - 2022-04-27

### 🚀 Features

- Use pygments for coloring code in exception window ([#591](https://github.com/4DNucleome/PartSeg/pull/591))
- Add option to calculate Measurement per Mask component ([#590](https://github.com/4DNucleome/PartSeg/pull/590))

### 🐛 Bug Fixes

- Update build wheels and sdist to have proper version tag ([#583](https://github.com/4DNucleome/PartSeg/pull/583))
- Fix removing the first measurement entry in the napari Measurement widget ([#584](https://github.com/4DNucleome/PartSeg/pull/584))
- Fix compatybility bug for conda Pyside2 version ([#595](https://github.com/4DNucleome/PartSeg/pull/595))
- Error when synchronization is loaded and new iloaded image has different dimensionality than currently loaded. ([#598](https://github.com/4DNucleome/PartSeg/pull/598))

### 🚜 Refactor

- Refactor the create batch plan widgets and add test for it ([#587](https://github.com/4DNucleome/PartSeg/pull/587))
- Drop napari below 0.4.12 ([#592](https://github.com/4DNucleome/PartSeg/pull/592))
- Update the order of ROI Mask algorithms to be the same as in older PartSeg versions ([#600](https://github.com/4DNucleome/PartSeg/pull/600))

### Build

- *(deps)* Bump partsegcore-compiled-backend from 0.13.11 to 0.14.0 in /requirements ([#582](https://github.com/4DNucleome/PartSeg/pull/582))
- *(deps)* Bump simpleitk from 2.1.1 to 2.1.1.2 in /requirements ([#589](https://github.com/4DNucleome/PartSeg/pull/589))
- *(deps)* Bump pyinstaller from 4.10 to 5.0 in /requirements ([#586](https://github.com/4DNucleome/PartSeg/pull/586))

## 0.14.0 - 2022-04-14

### 🚀 Features

- Allow to set zoom factor from interface in Search Label napari plugin ([#538](https://github.com/4DNucleome/PartSeg/pull/538))
- Add controlling of zoom factor of search ROI in main GUI ([#540](https://github.com/4DNucleome/PartSeg/pull/540))
- Better serialization mechanism allow for declaration data structure migration locally ([#462](https://github.com/4DNucleome/PartSeg/pull/462))
- Make \`\*.obsep" file possible to load in PartSeg Analysis ([#564](https://github.com/4DNucleome/PartSeg/pull/564))
- Add option to extract measurement profile or roi extraction profile from batch plan ([#568](https://github.com/4DNucleome/PartSeg/pull/568))
- Allow import calculation plan from batch result excel file ([#567](https://github.com/4DNucleome/PartSeg/pull/567))
- Improve error reporting when fail to deserialize data ([#574](https://github.com/4DNucleome/PartSeg/pull/574))
- Launch PartSeg GUI from napari ([#581](https://github.com/4DNucleome/PartSeg/pull/581))

### 🐛 Bug Fixes

- Fix "Show selected" rendering mode in PartSeg ROI Mask ([#565](https://github.com/4DNucleome/PartSeg/pull/565))

### 🚜 Refactor

- Store PartSegImage.Image channels as separated arrays ([#554](https://github.com/4DNucleome/PartSeg/pull/554))
- Remove deprecated modules. ([#429](https://github.com/4DNucleome/PartSeg/pull/429))
- Switch serialization backen to `nme` ([#569](https://github.com/4DNucleome/PartSeg/pull/569))

### 📚 Documentation

- Update changelog and add new badges to readme ([#580](https://github.com/4DNucleome/PartSeg/pull/580))

### 🧪 Testing

- Add test of creating AboutDialog ([#539](https://github.com/4DNucleome/PartSeg/pull/539))
- Setup test for python 3.10. Disable class_generator test for this python ([#570](https://github.com/4DNucleome/PartSeg/pull/570))

### Bugfix

- Add access by operator \[\] to pydantic.BaseModel base structures for keep backward compatybility ([#579](https://github.com/4DNucleome/PartSeg/pull/579))

### Build

- *(deps)* Bump sentry-sdk from 1.5.2 to 1.5.3 in /requirements ([#512](https://github.com/4DNucleome/PartSeg/pull/512))
- *(deps)* Bump ipython from 8.0.0 to 8.0.1 in /requirements ([#513](https://github.com/4DNucleome/PartSeg/pull/513))
- *(deps)* Bump pandas from 1.3.5 to 1.4.0 in /requirements ([#514](https://github.com/4DNucleome/PartSeg/pull/514))
- *(deps)* Bump oiffile from 2021.6.6 to 2022.2.2 in /requirements ([#521](https://github.com/4DNucleome/PartSeg/pull/521))
- *(deps)* Bump numpy from 1.22.1 to 1.22.2 in /requirements ([#524](https://github.com/4DNucleome/PartSeg/pull/524))
- *(deps)* Bump tifffile from 2021.11.2 to 2022.2.2 in /requirements ([#523](https://github.com/4DNucleome/PartSeg/pull/523))
- *(deps)* Bump qtpy from 2.0.0 to 2.0.1 in /requirements ([#522](https://github.com/4DNucleome/PartSeg/pull/522))
- *(deps)* Bump sentry-sdk from 1.5.3 to 1.5.4 in /requirements ([#515](https://github.com/4DNucleome/PartSeg/pull/515))
- *(deps)* Bump pyinstaller from 4.8 to 4.10 in /requirements ([#545](https://github.com/4DNucleome/PartSeg/pull/545))
- *(deps)* Bump pillow from 9.0.0 to 9.0.1 in /requirements ([#549](https://github.com/4DNucleome/PartSeg/pull/549))
- *(deps)* Bump sphinx from 4.4.0 to 4.5.0 in /requirements ([#561](https://github.com/4DNucleome/PartSeg/pull/561))
- *(deps)* Bump tifffile from 2022.2.9 to 2022.3.25 in /requirements ([#562](https://github.com/4DNucleome/PartSeg/pull/562))
- *(deps)* Bump sympy from 1.10 to 1.10.1 in /requirements ([#556](https://github.com/4DNucleome/PartSeg/pull/556))
- *(deps)* Bump sentry-sdk from 1.5.7 to 1.5.8 in /requirements ([#557](https://github.com/4DNucleome/PartSeg/pull/557))

## 0.13.15

### Bug Fixes

- Using `translation` instead of `translation_grid` for shifting layers. (#474)
- Bugs in napari plugins (#478)
- Missing mask when using roi extraction from napari (#479)
- Fix segmentation fault on macos machines (#487)
- Fixes for napari 0.4.13 (#506)

### Documentation

- Create 0.13.15 release (#511)
- Add categories and preview page workflow for the napari hub (#489)

### Features

- Assign properties to mask layer in napari measurement widget (#480)

### Build

- Bump qtpy from 1.11.3 to 2.0.0 in /requirements (#498)
- Bump pydantic from 1.8.2 to 1.9.0 in /requirements (#496)
- Bump sentry-sdk from 1.5.1 to 1.5.2 in /requirements (#497)
- Bump sphinx from 4.3.1 to 4.3.2 in /requirements (#500)
- Bump pyinstaller from 4.7 to 4.8 in /requirements (#502)
- Bump pillow from 8.4.0 to 9.0.0 in /requirements (#501)
- Bump requests from 2.26.0 to 2.27.1 in /requirements (#495)
- Bump numpy from 1.21.4 to 1.22.0 in /requirements (#499)
- Bump numpy from 1.22.0 to 1.22.1 in /requirements (#509)
- Bump sphinx from 4.3.2 to 4.4.0 in /requirements (#510)

## 0.13.14

### Bug Fixes

- ROI alternative representation (#471)
- Change additive to translucent in rendering ROI and Mask (#472)

### Features

- Add morphological watershed segmentation (#469)
- Add Bilateral image filter (#470)

## 0.13.13

### Bug Fixes

- Fix bugs in the generation process of the changelog for release. (#428)
- Restoring ROI on home button click in compare viewer (#443)
- Fix Measurement name prefix in bundled PartSeg. (#458)
- Napari widgets registration in pyinstaller bundle (#465)
- Hide points button if no points are loaded, hide Mask checkbox if no mask is set (#463)
- Replace Label data instead of adding/removing layers - fix blending layers (#464)

### Features

- Add threshold information in layer annotation in the Multiple Otsu ROI extraction method (#430)
- Add option to select rendering method for ROI (#431)
- Add callback mechanism to ProfileDict, live update of ROI render parameters (#432)
- Move the info bar on the bottom of the viewer (#442)
- Add options to load recent files in multiple files widget (#444)
- Add ROI annotations as properties to napari labels layer created by ROI Extraction widgets (#445)
- Add signals to ProfileDict, remove redundant synchronization mechanisms (#449)
- Allow ignoring updates for 21 days (#453)
- Save all components if no components selected in mask segmentation (#456)
- Add modal dialog for search ROI components (#459)
- Add full measurement support as napari widget (#460)
- Add search labels as napari widget (#467)

### Refactor

- Export common code for load/save dialog to one place (#437)
- Change most of call QFileDialog to more generic code (#440)

### Testing

- Add test for `PartSeg.common_backend` module (#433)

## 0.13.12

### Bug Fixes

- Importing the previous version of settings (#406)
- Cutting without masking data (#407)
- Save in subdirectory in batch plan (#414)
- Loading plugins for batch processing (#423)

### Features

- Add randomization option for correlation calculation (#421)
- Add Imagej TIFF writer for image. (#405)
- Mask create widget for napari (#395)
- In napari roi extraction method show information from roi extraction method (#408)
- Add `*[0-9].tif` button in batch processing window (#412)
- Better label representation in 3d view (#418)

### Refactor

- Use Font Awesome instead of custom symbols (#424)

## 0.13.11

### Bug Fixes

- Adding mask in Prepare Plan for batch (#383)
- Set proper completion mode in SearchComboBox (#384)
- Showing warnings on the error with ROI load (#385)

### Features

- Add CellFromNucleusFlow "Cell from nucleus flow" cell segmentation method  (#367)
- When cutting components in PartSeg ROI mask allow not masking outer data (#379)
- Theme selection in GUI (#381)
- Allow return points from ROI extraction algorithm (#382)
- Add measurement to get ROI annotation by name. (#386)
- PartSeg ROI extraction algorithms as napari plugins (#387)
- Add  Pearson, Mander's, Intensity, Spearman colocalization measurements (#392)
- Separate standalone napari settings from PartSeg embedded napari settings (#397)

### Performance

- Use faster calc bound function (#375)

### Refactor

- Remove CustomApplication (#389)

## 0.13.10

- change tiff save backend to ome-tiff
- add `DistanceROIROI` and `ROINeighbourhoodROI` measurements

## 0.13.9

- annotation show bugfix

## 0.13.8

- napari deprecation fixes
- speedup simple measurement
- bundle plugins initial support

## 0.13.7

- add measurements widget for napari
- fix bug in pipeline usage

## 0.13.6

- Hotfix release
- Prepare for a new napari version

## 0.13.5

- Small fixes for error reporting
- Fix mask segmentation

## 0.13.4

- Bugfix for outdated profile/pipeline preview

## 0.13.3

- Fix saving roi_info in multiple files and history

## 0.13.2

- Fix showing label in select label tab

## 0.13.1

- Add Haralick measurements
- Add obsep file support

## 0.13.0

- Add possibility of custom input widgets for algorithms
- Switch to napari Colormaps instead of custom one
- Add points visualization
- Synchronization widget for builtin (View menu) napari viewer
- Drop Python 3.6

## 0.12.7

- Fixes for napari 0.4.6

## 0.12.6

- Fix prev_mask_get
- Fix cache mechanism on mask change
- Update PyInstaller build

## 0.12.5

- Fix bug in pipeline execute

## 0.12.4

- Fix ROI Mask windows related build (signal not properly connected)

## 0.12.3

- Fix ROI Mask

## 0.12.2

- Fix windows bundle

## 0.12.1

- History of last opened files
- Add ROI annotation and ROI alternatives
- Minor bugfix

## 0.12.0

- Toggle multiple files widget in View menu
- Toggle Left panel in ROI Analysis in View Menu
- Rename Mask Segmentation to ROI Mask
- Add documentation for interface
- Add Batch processing tutorial
- Add information about errors to batch processing output file
- Load image from the batch prepare window
- Add search option in part of list and combo boxes
- Add drag and drop mechanism to load list of files to batch window.

## 0.11.5

- add side view to viewer
- fix horizontal view for Measurements result table

## 0.11.4

- bump to napari 0.3.8 in bundle
- fix bug with not presented segmentation loaded from project
- add frame (1 pix) to image cat from base one based on segmentation
- pin to Qt version to 5.14

## 0.11.3

- prepare for napari 0.3.7
- split napari io plugin on multiple part
- better reporting for numpy array via sentry
- fix setting color for mask marking

## 0.11.2

- Speedup image set in viewer using async calls
- Fix bug in long name of sheet with parameters

## 0.11.1

- Add screenshot option in View menu
- Add Voxels measurements

## 0.11.0

- Make sprawl algorithm name shorter
- Unify capitalisation of measurement names
- Add simple measurements to mask segmentation
- Use napari as viewer
- Add possibility to preview additional output of algorithms (In View menu)
- Update names of available Algorithm and Measurement to be more descriptive.

## 0.10.8

- fix synchronisation between viewers in Segmentation Analysis
- fix batch crash on error during batch run, add information about file on which calculation fails
- add changelog preview in Help > About

## 0.10.7

- in measurements, on empty list of components mean will return 0

## 0.10.6

- fix border rim preview
- fix problem with size of image preview
- zoom with scroll and moving if rectangle zoom is not marked

## 0.10.5

- make PartSeg PEP517 compatible.
- fix multiple files widget on Windows (path normalisation)

## 0.10.4

- fix slow zoom

## 0.10.3

- deterministic order of elements in batch processing.

## 0.10.2

- bugfixes

## 0.10.1

- bugfixes

## 0.10.0

- Add creating custom label coloring.
- Change execs interpreter to python 3.7.
- Add masking operation in Segmentation Mask.
- Change license to BSD.
- Allow select root type in batch processing.
- Add median filter in preview.

## 0.9.7

- fix bug in compare mask

## 0.9.6

- fix bug in loading project with mask
- upgrade PyInstaller version (bug  GHSA-7fcj-pq9j-wh2r)

## 0.9.5

- fix bug in loading project in "Segmentation analysis"

## 0.9.4

- read mask segmentation projects
- choose source type in batch
- add initial support to OIF and CZI file format
- extract utils to PartSegCore module
- add automated tests of example notebook
- reversed mask
- load segmentation parameters in mask segmentation
- allow use sprawl in segmentation tool
- add radial split of mask for measurement
- add all measurement results in batch, per component sheet

## 0.9.3

- start automated build documentation
- change color map backend and allow for user to create custom color map.
- segmentation compare
- update test engines
- support of PySide2

## 0.9.2.3

- refactor code to make easier create plugin for mask segmentation
- create class base updater for update outdated algorithm description
- fix save functions
- fix different bugs

## 0.9.2.2

- extract static data to separated package
- update marker of fix range and add mark of gauss in channel control

## 0.9.2.1

- add VoteSmooth and add choosing of smooth algorithm

## 0.9.2

- add pypi base check for update

- remove resetting image state when change state in same image

- in stack segmentation add options to picking components from segmentation's

- in mask segmentation add:

  - preview of segmentation parameters per component,
  - save segmentation parameters in save file
  - new implementation of batch mode.

## 0.9.1

- Add multiple files widget

- Add Calculating distances between segmented object and mask

- Batch processing plan fixes:

  - Fix adding pipelines to plan
  - Redesign mask widget

- modify measurement backend to allow calculate multi channel measurements.

## 0.9

Begin of changelog
