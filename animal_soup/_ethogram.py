import pandas as pd
from ipywidgets import HBox, VBox, Select, Button, Layout, RadioButtons
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector
import numpy as np
from ._behavior import BehaviorVizContainer

ETHOGRAM_COLORS = {
    "lift": "b",
    "handopen": "green",
    "grab": "r",
    "sup": "cyan",
    "atmouth": "magenta",
    "chew": "yellow"
}

class EthogramVizContainer(BehaviorVizContainer):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 start_index: int = 0,
                 ):
        super(EthogramVizContainer, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )

        self.plot = None

        self._make_ethogram_plot()

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row_ix]

        if self.plot is None:
            self.plot = Plot(size=(500, 100))

        if self._check_for_cleaned_array(row=row):
            self.ethogram_array = row["cleaned_ethograms"][self.selected_trial]
        else:
            self.ethogram_array = row["ethograms"][self.selected_trial]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=20,
                name=b
            )

            lg.colors = 0
            lg.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -5) - 1
            lg.position_y = y_pos

        self.ethogram_selector = LinearSelector(
            selection=0,
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            parent=lg,
            end_points=(y_bottom, y_pos),
        )

        self.plot.add_graphic(self.ethogram_selector)
        self.ethogram_selector.selection.add_event_handler(self.ethogram_selection_event_handler)
        self.plot.auto_scale()

    def _check_for_cleaned_array(self, row):
        if "cleaned_ethograms" not in self._dataframe.columns:
            return False
        if self.selected_trial in row["cleaned_ethograms"].keys():
            return True
        return False

    def ethogram_selection_event_handler(self, ev):
        """
        Event handler called for linear selector.
        """
        ix = ev.pick_info["selected_index"]
        self.image_widget.sliders["t"].value = ix

    def _trial_change(self, obj):
        """
        Event handler called when a trial is changed in self.trial_selector.
        Updates the behavior imagewidget and ethogram plot with new data.
        """
        super()._trial_change(obj)

        # force clearing of event handlers for selectors
        # seems to be an issue with fpl delete graphic method for selectors
        self.plot.selectors[0].selection._event_handlers.clear()
        self.plot.clear()
        self._make_ethogram_plot()

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                self.trial_selector
        ]),
            self.plot.show()])