import pandas as pd
from ipywidgets import HBox, VBox, Textarea, Layout
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

ETHOGRAM_BEHAVIOR_MAPPING = {
    "lift": 0,
    "handopen": 1,
    "grab": 2,
    "sup": 3,
    "atmouth": 4,
    "chew": 5,
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
        self.behavior_count = None

        self._make_ethogram_plot()
        self._set_behavior_frame_count()

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row_ix]

        if self.plot is None:
            self.plot = Plot(size=(700, 300))

        if self._check_for_cleaned_array(row=row):
            self.ethogram_array = row["cleaned_ethograms"]
        else:
            self.ethogram_array = row["ethograms"]

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

    def _check_for_cleaned_array(self, row: pd.Series):
        if "cleaned_ethograms" not in self._dataframe.columns:
            return False
        if row["cleaned_ethograms"] is not None:
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
        self._set_behavior_frame_count()

    def _set_behavior_frame_count(self):
        durations = self._get_behavior_frame_count()
        if self.behavior_count is None:
            self.behavior_count = Textarea(
                                        value=f'lift: {durations["lift"]}\n'
                                              f'handopen: {durations["handopen"]}\n'
                                              f'grab: {durations["grab"]}\n'
                                              f'sup: {durations["sup"]}\n'
                                              f'atmouth: {durations["atmouth"]}\n'
                                              f'chew: {durations["chew"]}',
                                        description="Frame #:",
                                        disabled=True,
                                        layout=Layout(height="65%", width="auto"))
        else:
            self.behavior_count.value = (f'lift: {durations["lift"]}\n'
            f'handopen: {durations["handopen"]}\n'
            f'grab: {durations["grab"]}\n'
            f'sup: {durations["sup"]}\n'
            f'atmouth: {durations["atmouth"]}\n'
            f'chew: {durations["chew"]}')

    def _get_behavior_frame_count(self):
        """Get the duration of each behavior in the currently selected ethogram."""
        durations = dict()
        for behavior in ETHOGRAM_BEHAVIOR_MAPPING.keys():
            durations[behavior] = int(self.ethogram_array[ETHOGRAM_BEHAVIOR_MAPPING[behavior]].sum())
        return durations

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                  VBox([self.trial_selector,
                        self.behavior_count])
                  ]),
            self.plot.show()])
