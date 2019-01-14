import argparse
from typing import Optional, Sequence, Text
from . import report_utils


class CustomParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--no_report", action="store_false", help="disable error reporting")

    def parse_args(self, args: Optional[Sequence[Text]] = None,
                   namespace: Optional[argparse.Namespace] = None):
        args = super().parse_args(args, namespace)
        report_utils.report_errors = args.no_report
        return args
