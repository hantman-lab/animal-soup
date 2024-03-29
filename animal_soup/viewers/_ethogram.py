import pandas as pd
from ipywidgets import HBox, VBox, Textarea, Layout, ToggleButtons
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector
import numpy as np
from ._behavior import BehaviorVizContainer
from .ethogram_utils import get_ethogram_from_disk

ETHOGRAM_COLORS = {
    "lift": "b",
    "handopen": "green",
    "grab": "r",
    "sup": "cyan",
    "atmouth": "magenta",
    "chew": "yellow",
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
    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: int = 0,
        mode: str = "inference"
    ):
        """
        Wraps BehaviorVizContainer, in addition to showing behavior videos, will also show corresponding ethograms.

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe used to create the ipydatagrid. Datagrid will show a subset of columns from the original
            dataframe.
        start_index: int, default 0
            Row index in datagrid that will start out as being selected initially. Default is first row.
        mode: str, default 'inference'
            One of ['ground', 'inference']. The locations of ethograms can either be stored on disk if they
            have been inferred or in the dataframe if they are hand-labels. Mode argument can be passed to the
            ethogram viewer to set where to look for available ethograms.
        """
        super(EthogramVizContainer, self).__init__(
            dataframe=dataframe, start_index=start_index
        )

        self.plot = None
        self.behavior_count = None

        if mode not in ["inference", "ground"]:
            raise ValueError("'mode' argument must be one of ['inference', 'ground']")

        self.mode = mode

        self.mode_selector = ToggleButtons(
                                options=["slow", "medium", "fast"],
                                description="Mode",
                                disabled=False,
                                button_style='',
                                value="fast",
                                layout=Layout(width='200px')
        )

        self.mode_selector.observe(self._toggle_mode, "value")

        self._make_ethogram_plot()
        self._set_behavior_frame_count()

    def _toggle_mode(self, obj):
        """Toggle ethogram for a trial based on prediction mode."""
        # force clearing of event handlers for selectors
        # seems to be an issue with fpl delete graphic method for selectors
        if len(self.plot.selectors) > 0:
            self.plot.selectors[0].selection._event_handlers.clear()
        self.plot.clear()
        self._make_ethogram_plot()
        self._set_behavior_frame_count()

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row_ix]

        if self.plot is None:
            self.plot = Plot(size=(700, 300))

        # if mode is ground
        if self.mode == "ground":
            if self._check_for_cleaned_array(row=row):
                self.ethogram_array = row["cleaned_ethograms"]
            else:
                self.ethogram_array = row["ethograms"]
        else: # mode must be inference
            self.ethogram_array = get_ethogram_from_disk(row=row, mode=self.mode_selector.value)

        # will return an empty plot when a mode has been selected that inference hasn't been run for
        # allows for toggling between modes without throwing errors
        if self.ethogram_array is None:
            return

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.plot.add_line(
                data=np.column_stack([xs, ys]), thickness=20, name=b
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
        self.ethogram_selector.selection.add_event_handler(
            self._ethogram_selection_event_handler
        )
        self.plot.auto_scale()

    def _check_for_cleaned_array(self, row: pd.Series):
        """Checks whether there is a cleaned array to be shown instead."""
        if "cleaned_ethograms" not in self._dataframe.columns:
            return False
        if row["cleaned_ethograms"] is not None:
            return True
        return False

    def _ethogram_selection_event_handler(self, ev):
        """Event handler called for linear selector."""
        ix = ev.pick_info["selected_index"]
        self.image_widget.sliders["t"].value = ix

    def _row_changed(self, *args):
        """Event handler for when a row in the datagrid is changed."""
        super()._row_changed()

        # force clearing of event handlers for selectors
        # seems to be an issue with fpl delete graphic method for selectors
        if len(self.plot.selectors) > 0:
            self.plot.selectors[0].selection._event_handlers.clear()
        self.plot.clear()
        self._make_ethogram_plot()
        self._set_behavior_frame_count()

    def _set_behavior_frame_count(self):
        """Sets the behavior duration for each behavior."""
        # no durations to calculate if inference has not been run with currently selected mode
        # blank plot currently being displayed
        if self.ethogram_array is None:
            self.behavior_count = Textarea(
                value=f'lift: {None}\n'
                      f'handopen: {None}\n'
                      f'grab: {None}\n'
                      f'sup: {None}\n'
                      f'atmouth: {None}\n'
                      f'chew: {None}',
                description="Duration:",
                disabled=True,
                layout=Layout(height="65%", width="auto"),
            )
            return
        durations = self._get_behavior_frame_count()
        if self.behavior_count is None:
            self.behavior_count = Textarea(
                value=f'lift: {durations["lift"]}\n'
                f'handopen: {durations["handopen"]}\n'
                f'grab: {durations["grab"]}\n'
                f'sup: {durations["sup"]}\n'
                f'atmouth: {durations["atmouth"]}\n'
                f'chew: {durations["chew"]}',
                description="Duration:",
                disabled=True,
                layout=Layout(height="65%", width="auto"),
            )
        else:
            self.behavior_count.value = (
                f'lift: {durations["lift"]}\n'
                f'handopen: {durations["handopen"]}\n'
                f'grab: {durations["grab"]}\n'
                f'sup: {durations["sup"]}\n'
                f'atmouth: {durations["atmouth"]}\n'
                f'chew: {durations["chew"]}'
            )

    def _get_behavior_frame_count(self):
        """Get the duration of each behavior in the currently selected ethogram."""
        durations = dict()
        for behavior in ETHOGRAM_BEHAVIOR_MAPPING.keys():
            durations[behavior] = int(
                self.ethogram_array[ETHOGRAM_BEHAVIOR_MAPPING[behavior]].sum()
            )
        return durations

    def show(self):
        """Shows the widget."""
        trial_buttons = HBox([self.previous_button, self.next_button])

        return VBox(
            [
                self.datagrid,
                HBox(
                    [
                        self.image_widget.show(sidecar=False),
                        VBox([trial_buttons, self.mode_selector, self.behavior_count]),
                    ]
                ),
                self.plot.show(sidecar=False),
            ]
        )
