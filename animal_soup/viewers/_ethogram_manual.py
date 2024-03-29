import pandas as pd
from ipywidgets import HBox, VBox, Textarea, Layout
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearRegionSelector
import numpy as np
from ._behavior import BehaviorVizContainer
from .ethogram_utils import _get_clean_ethogram, save_ethogram_to_disk

BEHAVIORS = ["lift", "handopen", "grab", "sup", "atmouth", "chew"]

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


class EthogramManualVizContainer(BehaviorVizContainer):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: int = 0,
    ):
        """
        Wraps BehaviorVizContainer, in addition to showing behavior videos, will allow for manual annotation of
        ethograms.

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe used to create the ipydatagrid. Datagrid will show a subset of columns from the original
            dataframe.
        start_index: int, default 0
            Row index in datagrid that will start out as being selected initially. Default is first row.
        """
        super(EthogramManualVizContainer, self).__init__(
            dataframe=dataframe, start_index=start_index
        )

        self.plot = None
        self.behavior_count = None

        self._make_ethogram_plot()
        self._set_behavior_frame_count()

    @property
    def current_behavior(self):
        """Current behavior selected in ethogram."""
        return self._current_behavior

    @current_behavior.setter
    def current_behavior(self, behavior: str):
        """Set the currently selected behavior."""
        self._current_behavior = self.plot[behavior]
        self.current_highlight = f"{behavior}_highlight"

    @property
    def current_highlight(self):
        """Current graphic that is highlighted."""
        return self._current_highlight

    @current_highlight.setter
    def current_highlight(self, behavior_highlight: str):
        """Set the currently selected highlight."""
        self._current_highlight = self.plot[behavior_highlight]
        self.current_highlight.colors = "white"
        if self.current_behavior.name != behavior_highlight.split("_")[0]:
            self.current_behavior = behavior_highlight.split("_")[0]

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row_ix]

        if self.plot is None:
            self.plot = Plot(size=(700, 300))
            self.plot.renderer.add_event_handler(
                self._ethogram_key_event_handler, "key_down"
            )

        # assumes that no ethogram exists and user is doing manual annotation
        # check if manual ethogram has been saved to disk
        self.ethogram_array = _get_clean_ethogram(row=row)
        # else initialize empty array of zeros to annotate from
        if self.ethogram_array is None:
            self.ethogram_array = np.zeros(shape=(len(ETHOGRAM_COLORS.keys()), self.image_widget.sliders["t"].max))

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg_data = self.plot.add_line(
                data=np.column_stack([xs, ys]), thickness=20, name=b
            )

            lg_highlight = self.plot.add_line(
                data=np.column_stack([xs, ys]), thickness=21, name=f"{b}_highlight"
            )

            lg_highlight.colors = 0
            lg_highlight.position_z = 1

            lg_data.colors = 0
            lg_data.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -5) - 1
            lg_data.position_y = y_pos
            lg_highlight.position_y = y_pos

        # default initial selected behavior will always be lift
        self.current_behavior = "lift"

        self.ethogram_region_selector = LinearRegionSelector(
            bounds=(0, 50),
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            origin=(y_bottom, -25),
            fill_color=(0, 0, 0, 0),
            parent=lg_data,
            size=(55),
            name="ethogram_selector",
        )

        self.plot.add_graphic(self.ethogram_region_selector)
        self.ethogram_region_selector.selection.add_event_handler(
            self._ethogram_selection_event_handler
        )
        self.plot.auto_scale()

    def _ethogram_selection_event_handler(self, ev):
        """
        Event handler called for linear region to slide through video data using the left or right bound
        of the linear region selector.
        """
        source = ev.pick_info["move_info"].source
        if source is self.ethogram_region_selector.edges[0]:
            self.image_widget.sliders["t"].value = ev.pick_info["selected_indices"][0]
        elif source is self.ethogram_region_selector.edges[1]:
            self.image_widget.sliders["t"].value = ev.pick_info["selected_indices"][-1]
        else:
            self.image_widget.sliders["t"].value = ev.pick_info["selected_indices"][0]

    def _ethogram_key_event_handler(self, obj):
        """Event handler for handling keyboard events to clean up ethograms."""
        # index of current highlight graphic
        current_ix = BEHAVIORS.index(self.current_behavior.name)
        # selected indices of linear region selector
        selected_ixs = self.plot.selectors[0].get_selected_indices(
            self.current_behavior
        )

        # move `down` a behavior in the current ethogram
        if obj.key == "s":
            # set current highlight behavior to black
            self.current_highlight.colors = "black"
            # if current graphic is last behavior, should circle around to first behavior
            if current_ix + 1 == len(BEHAVIORS):
                self.current_behavior = BEHAVIORS[0]
            else:
                self.current_behavior = BEHAVIORS[current_ix + 1]

        # move `up` a behavior in the current ethogram
        elif obj.key == "q":
            self.current_highlight.colors = "black"
            # if already at first behavior, should loop to last behavior
            if current_ix - 1 < 0:
                self.current_behavior = BEHAVIORS[-1]
            else:
                self.current_behavior = BEHAVIORS[current_ix - 1]

        # set selected indices of current behavior to '1'
        elif obj.key == "1":
            self.current_behavior.colors[
            selected_ixs[0]: selected_ixs[-1]
            ] = ETHOGRAM_COLORS[self.current_behavior.name]
            self.save_ethogram()
        # set selected indices of current behavior to `0`
        elif obj.key == "2":
            self.current_behavior.colors[selected_ixs[0]: selected_ixs[-1]] = "black"
            self.save_ethogram()

        # reset entire ethogram
        elif obj.key == "r":
            self.reset_ethogram()

        # reset current selected behavior
        elif obj.key == "t":
            self.reset_ethogram(current_behavior=True)

        # save ethogram
        elif obj.key == "y":
            self.save_ethogram()

    def save_ethogram(self):
        """
        Saves an ethogram to the clean dataframe or to disk depending on the
        mode specified at instantiation.
        """
        # create new ethogram based off of indices that are not black
        row = self._dataframe.iloc[self.current_row_ix]
        trial_length = self.ethogram_array[0].shape[0]
        new_ethogram = np.zeros(shape=(6, trial_length))
        for i, g in enumerate(ETHOGRAM_COLORS.keys()):
            non_zero_ixs = np.where(self.plot[g].colors[:] != np.array([0, 0, 0, 1]))[0]
            new_ethogram[i][non_zero_ixs] = 1

        save_ethogram_to_disk(row, new_ethogram)

    def reset_ethogram(self, current_behavior: bool = False):
        """Will reset the current behavior selected or the entire cleaned ethogram back to the original ethogram."""
        if current_behavior:  # reset only current behavior to original
            current_ix = BEHAVIORS.index(self.current_behavior.name)
            self.current_behavior.colors[:] = "black"
            self.current_behavior.colors[
                self.ethogram_array[current_ix] == 1
                ] = ETHOGRAM_COLORS[self.current_behavior.name]
        else:  # reset all behaviors to original
            for i, g in enumerate(ETHOGRAM_COLORS.keys()):
                self.plot[g].colors[:] = "black"
                self.plot[g].colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[g]
        # save ethogram to disk after reset
        self.save_ethogram()

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
                        VBox([trial_buttons, self.behavior_count]),
                    ]
                ),
                self.plot.show(sidecar=False),
            ]
        )
