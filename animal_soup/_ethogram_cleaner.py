import os.path

import pandas as pd
from ipywidgets import HBox, VBox, Select, Button, Layout, RadioButtons
from fastplotlib import Plot
from fastplotlib.graphics.selectors import LinearSelector, LinearRegionSelector
import numpy as np
from ._ethogram import EthogramVizContainer, ETHOGRAM_COLORS
from pathlib import Path
from typing import Union
from .batch_utils import validate_path
import warnings
from fastplotlib.graphics import LineGraphic

HIGHLIGHT_GRAPHICS = ["lift_highlight",
                      "handopen_highlight",
                      "grab_highlight",
                      "sup_highlight",
                      "atmouth_highlight",
                      "chew_highlight"]

class EthogramCleaner(EthogramVizContainer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
    ):
        """
        Creates container for editing ethograms and saving them to a new dataframe. 
        
        Parameters
        ----------
        dataframe: ``pandas.Dataframe``
            Dataframe for ethograms that need to be cleaned. Should be organized in terms of `animal_id` and
            `session_id`. Ethograms that need to be cleaned for a given `animal_id`/`session_id` pairing should be
            stored in the `ethograms` column as a ``dict`` of `{trial: ethogram}`. 
        start_index: ``int``, default 0
            Row of the dataframe that will initially be selected to view videos and corresponding ethograms. 
        clean_df_path: ``str`` or ``pathlib.Path``, default ``None``
            Path to dataframe where clean ethograms will be stored. Should have same structure as `dataframe` arg. If
            ``None``, an existing clean dataframe will be loaded if it exists or a new clean dataframe will be
            created to store cleaned ethograms. 
        """
        super(EthogramCleaner, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )
        # add column for cleaned ethograms to df if not exists
        if "cleaned_ethograms" not in self._dataframe.columns:
            self._dataframe.insert(
                loc=3,
                column="cleaned_ethograms",
                value=[dict() for i in range(len(self._dataframe.index))]
            )

        self._dataframe.behavior.save_to_disk()


    @property
    def current_behavior(self):
        """Current behavior selected in ethogram."""
        return self._current_behavior

    @current_behavior.setter
    def current_behavior(self, graphic: LineGraphic):
        """Set the currently selected behavior."""
        self._current_behavior = graphic
        self._current_highlight = self.plot[f"{self.current_behavior.name}_highlight"]

    @ property
    def current_highlight(self):
        """Current graphic that is highlighted."""
        return self._current_highlight

    @current_highlight.setter
    def current_highlight(self, graphic: LineGraphic):
        """Set the currently selected highlight."""
        self._current_highlight = graphic
        self.current_highlight.colors = "white"
        self.current_behavior = self.plot[self.current_highlight.name.split('_')[0]]

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row_ix]

        if self.plot is None:
            self.plot = Plot(size=(500, 100))
            self.plot.renderer.add_event_handler(self.ethogram_key_event_handler, "key_down")

        if self._check_for_cleaned_array(row=row):
            self.ethogram_array = row["cleaned_ethograms"][self.selected_trial]
        else:
            self.ethogram_array = row["ethograms"][self.selected_trial]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg_data = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=20,
                name=b
            )

            lg_highlight = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=21,
                name=f"{b}_highlight"
            )

            lg_highlight.colors = 0
            lg_highlight.position_z = 1


            lg_data.colors = 0
            lg_data.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -5) - 1
            lg_data.position_y = y_pos
            lg_highlight.position_y = y_pos

        # default initial selected behavior will always be lift
        self.current_behavior = self.plot["lift"]
        self.current_highlight = self.plot["lift_highlight"]
        self.current_highlight.colors = "white"

        self.ethogram_region_selector = LinearRegionSelector(
            bounds=(0, 50),
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            origin=(y_bottom, -25),
            fill_color=(0, 0, 0, 0),
            parent=lg_data,
            size=(55),
            name="ethogram_selector"
        )

        self.plot.add_graphic(self.ethogram_region_selector)
        self.ethogram_region_selector.selection.add_event_handler(self.ethogram_selection_event_handler)
        self.plot.auto_scale()

    def ethogram_selection_event_handler(self, ev):
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

    def ethogram_key_event_handler(self, obj):
        """Event handler for handling keyboard events to clean up ethograms."""
        # index of current highlight graphic
        current_ix = HIGHLIGHT_GRAPHICS.index(self.current_highlight.name)
        # selected indices of linear region selector
        selected_ixs = self.plot.selectors[0].get_selected_indices(self.current_behavior)

        # move `down` a behavior in the current ethogram
        if obj.key == 's':
            for g in HIGHLIGHT_GRAPHICS:
                self.plot[g].colors = "black"
            # if current graphic is last behavior, should circle around
            if current_ix + 1 == len(HIGHLIGHT_GRAPHICS):
                self.current_highlight = self.plot[HIGHLIGHT_GRAPHICS[0]]
            else:
                self.current_highlight = self.plot[HIGHLIGHT_GRAPHICS[current_ix + 1]]

        # move `up` a behavior in the current ethogram
        elif obj.key == 'q':
            for g in HIGHLIGHT_GRAPHICS:
                self.plot[g].colors = "black"
            if current_ix - 1 < 0:
                self.current_highlight = self.plot[HIGHLIGHT_GRAPHICS[-1]]
            else:
                self.current_highlight = self.plot[HIGHLIGHT_GRAPHICS[current_ix - 1]]

        # set selected indices of current behavior to '1'
        elif obj.key == '1':
            self.current_behavior.colors[selected_ixs[0]:selected_ixs[-1]] = ETHOGRAM_COLORS[self.current_behavior.name]
            self.save_ethogram()
        # set selected indices of current behavior to `0`
        elif obj.key == '2':
            self.current_behavior.colors[selected_ixs[0]:selected_ixs[-1]] = "black"
            self.save_ethogram()

        # reset entire ethogram
        elif obj.key == 'r':
            self.reset_ethogram()

        # reset current selected behavior
        elif obj.key == 't':
            self.reset_ethogram(current_behavior=True)

        # save ethogram
        elif obj.key == 'y':
            self.save_ethogram()

    def save_ethogram(self):
        """Saves an ethogram to the clean dataframe."""
        # create new ethogram based off of indices that are not black
        row = self._dataframe.iloc[self.current_row_ix]
        trial_length = row["ethograms"][self.selected_trial][0].shape[0]
        new_ethogram = np.zeros(shape=(6, trial_length))
        for i, g in enumerate(ETHOGRAM_COLORS.keys()):
            non_zero_ixs = np.where(self.plot[g].colors[:] != np.array([0, 0, 0, 1]))[0]
            new_ethogram[i][non_zero_ixs] = 1
        # check if key already in clean_df
        row["cleaned_ethograms"][self.selected_trial] = new_ethogram
        # save clean_df to disk
        self._dataframe.behavior.save_to_disk()

    def reset_ethogram(self, current_behavior: bool = False):
        """Will reset the current behavior selected or the entire cleaned ethogram back to the original ethogram."""
        if current_behavior: # reset only current behavior to original
            current_ix = HIGHLIGHT_GRAPHICS.index(self.current_highlight.name)
            self.current_behavior.colors[:] = "black"
            self.current_behavior.colors[self.ethogram_array[current_ix] == 1] = ETHOGRAM_COLORS[self.current_behavior.name]
        else: # reset all behaviors to original
            for i, g in enumerate(ETHOGRAM_COLORS.keys()):
                self.plot[g].colors[:] = "black"
                self.plot[g].colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[g]
        # save ethogram to disk after reset
        self.save_ethogram()
