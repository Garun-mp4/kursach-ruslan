import sys
import unittest
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from planning.run_m5_verification import (
    _font,
    _legend_layout,
    _plot_bottom_for_legend,
)


class LegendLayoutTests(unittest.TestCase):
    def test_wrapped_legend_stays_above_x_axis_label(self):
        font = _font(16)
        series = [
            {"label": f"perceived object {index} with extended label", "color": "#123456"}
            for index in range(5)
        ]

        entries = _legend_layout(series, font, 118, 1215, 760)

        row_y = sorted({y for _, _, y in entries})
        self.assertGreater(len(row_y), 1)
        x_label_top = 760 - 34
        last_legend_bottom = max(y for _, _, y in entries)
        self.assertLess(last_legend_bottom, x_label_top - 20)
        self.assertEqual(row_y[-1], 760 - 61)
        bottom = _plot_bottom_for_legend(entries)
        tick_label_bottom = bottom + 10 + font.getbbox("0")[3]
        first_legend_text_top = row_y[0] - 18
        self.assertLess(tick_label_bottom, first_legend_text_top - 8)
        self.assertEqual(bottom, 635 - 26)

    def test_single_row_keeps_legacy_baseline(self):
        entries = _legend_layout(
            [{"label": "TCP path", "color": "#123456"}], _font(16), 118, 1215, 760)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][2], 760 - 61)
        self.assertEqual(_plot_bottom_for_legend(entries), 635)

    def test_annotation_only_series_are_not_legend_entries(self):
        entries = _legend_layout(
            [{"label": "phase marker", "color": "#123456", "annotation": "start"}],
            _font(16), 118, 1215, 760)

        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
