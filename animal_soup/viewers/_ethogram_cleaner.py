import pandas as pd
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearRegionSelector
import numpy as np
from ._ethogram import EthogramVizContainer, ETHOGRAM_COLORS
from .ethogram_utils import get_ethogram_from_disk, save_ethogram_to_disk

BEHAVIORS = ["lift", "handopen", "grab", "sup", "atmouth", "chew"]


class EthogramCleanerVizContainer(EthogramVizContainer):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: int = 0,
        mode: str = "inference"
    ):
        """
        Creates container for editing ethograms and saving them to a new dataframe.

        Parameters
        ----------
        dataframe: pandas.Dataframe
            Dataframe to create ipydatagrid viewer from.
        start_index: int, default 0
            Row of the dataframe that will initially be selected to view videos and corresponding ethograms.
        mode: str, default 'inference'
            One of ['ground', 'inference']. Indicates the location in which to load ethograms for cleaning from
            as well as where to save cleaned ethograms.
        """
        super(EthogramCleanerVizContainer, self).__init__(
            dataframe=dataframe, start_index=start_index, mode=mode
        )
        # add column for cleaned ethograms to df if not exists
        if self.mode == "ground" and "cleaned_ethograms" not in self._dataframe.columns:
            self._dataframe.insert(loc=5, column="cleaned_ethograms", value=None)

            self._dataframe.behavior.save_to_disk()

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

        if self.mode == "ground":
            # if an ethogram has been cleaned, want to make sure to show it
            if self._check_for_cleaned_array(row=row):
                self.ethogram_array = row["cleaned_ethograms"]
            else:
                self.ethogram_array = row["ethograms"]
        else: # mode is inference look for cleaned ethogram to show and if not show predicted
            self.ethogram_array = get_ethogram_from_disk(row=row, mode=self.mode_selector.value)

        # will return an empty plot when a mode has been selected that inference hasn't been run for
        # allows for toggling between modes without throwing errors
        if self.ethogram_array is None:
            return

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
                selected_ixs[0] : selected_ixs[-1]
            ] = ETHOGRAM_COLORS[self.current_behavior.name]
            self.save_ethogram()
        # set selected indices of current behavior to `0`
        elif obj.key == "2":
            self.current_behavior.colors[selected_ixs[0] : selected_ixs[-1]] = "black"
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

        if self.mode == "ground":
            # save new ethogram to cleaned_ethograms column
            row["cleaned_ethograms"] = new_ethogram
            # save df to disk
            self._dataframe.behavior._unsafe_save()
        else: # save ethogram to disk
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
