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

HIGHLIGHT_GRAPHICS = ["lift_highlight", "handopen_highlight", "grab_highlight", "sup_highlight", "atmouth_highlight",
                   "chew_highlight"]

class EthogramCleaner(EthogramVizContainer):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
            clean_df: Union[str, Path] = None
    ):
        super(EthogramCleaner, self).__init__(
            dataframe=dataframe,
            start_index=start_index
        )
        # create or load dataframe where ethograms are stored after cleaning
        if clean_df is None:
            df_dir, relative_path = self._dataframe.paths.split(self._dataframe.paths.get_df_path())
            self.clean_df_path = df_dir.joinpath(relative_path.stem).with_name(f'{relative_path.stem}_cleaned').with_suffix('.hdf')
            # check if clean_df exists and user didn't
            if os.path.exists(self.clean_df_path):
                self.clean_df = pd.read_hdf(self.clean_df_path)
            # copy original df to clean, and then as ethograms are cleaned will overwrite
            self._clean_df = self._dataframe.copy(deep=True)
            # save clean df to disk in same place as current dataframe
            self._clean_df.to_hdf(self.clean_df_path, key='df')
        elif isinstance(clean_df, str):
            self.clean_df_path = Path(clean_df)
            validate_path(self.clean_df_path)
            self._clean_df = pd.read_hdf(self.clean_df_path)
        else:
            self.clean_df_path = clean_df
            validate_path(self.clean_df_path)
            self._clean_df = pd.read_hdf(self.clean_df_path)

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]

        if self.plot is None:
            self.plot = Plot(size=(500, 100))
            self.plot.renderer.add_event_handler(self.ethogram_key_event_handler, "key_down")

        self.ethogram_array = row["ethograms"][self.selected_trial]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg_data = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=10,
                name=b
            )

            lg_highlight = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=11,
                name=f"{b}_highlight"
            )

            lg_highlight.colors = 0
            lg_highlight.position_z = 1


            lg_data.colors = 0
            lg_data.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -10) - 1
            lg_data.position_y = y_pos
            lg_highlight.position_y = y_pos

        self.plot["lift_highlight"].colors = "white"
        self.currently_selected_behavior = self.plot["lift_highlight"]

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

    def ethogram_key_event_handler(self, obj):
        """Event handler for handling keyboard events to clean up ethograms."""
        current_highlight_name = self.currently_selected_behavior.name
        current_ix = HIGHLIGHT_GRAPHICS.index(current_highlight_name)
        current_behavior_name = self.currently_selected_behavior.name.split('_')[0]
        selected_ixs = self.plot.selectors[0].get_selected_indices(self.plot[current_behavior_name])
        if obj.key == 's':
            for g in HIGHLIGHT_GRAPHICS:
                self.plot[g].colors = "black"
            if current_ix + 1 == len(HIGHLIGHT_GRAPHICS):
                self.plot[HIGHLIGHT_GRAPHICS[0]].colors = "white"
                self.currently_selected_behavior = self.plot[HIGHLIGHT_GRAPHICS[0]]
            else:
                self.plot[HIGHLIGHT_GRAPHICS[current_ix + 1]].colors = "white"
                self.currently_selected_behavior = self.plot[HIGHLIGHT_GRAPHICS[current_ix + 1]]
        elif obj.key == 'w':
            for g in HIGHLIGHT_GRAPHICS:
                self.plot[g].colors = "black"
            if current_ix - 1 < 0:
                self.plot[HIGHLIGHT_GRAPHICS[-1]].colors = "white"
                self.currently_selected_behavior = self.plot[HIGHLIGHT_GRAPHICS[-1]]
            else:
                self.plot[HIGHLIGHT_GRAPHICS[current_ix - 1]].colors = "white"
                self.currently_selected_behavior = self.plot[HIGHLIGHT_GRAPHICS[current_ix - 1]]
        elif obj.key == '1':
            self.plot[current_behavior_name].colors[selected_ixs[0]:selected_ixs[-1]] = ETHOGRAM_COLORS[current_behavior_name]
            self.save_ethogram()
        elif obj.key == '2':
            self.plot[current_behavior_name].colors[selected_ixs[0]:selected_ixs[-1]] = "black"
            self.save_ethogram()
        elif obj.key == 'r':
            self.reset_ethogram()

    def save_ethogram(self):
        """Saves an ethogram to the clean dataframe."""
        # create new ethogram based off of indices that are not black
        row = self._dataframe.iloc[self.current_row]
        trial_length = row["ethograms"][self.selected_trial][0].shape[0]
        new_ethogram = np.zeros(shape=(6, trial_length))
        for i, g in enumerate(ETHOGRAM_COLORS.keys()):
            non_zero_ixs = np.where(self.plot[g].colors[:] != np.array([0, 0, 0, 1]))[0]
            new_ethogram[i][non_zero_ixs] = 1
        # check if key already in clean_df
        clean_df_row = self._clean_df[(self._clean_df["animal_id"] == row["animal_id"]) &
                                      (self._clean_df["session_id"] == row["session_id"])]
        # add or update dictionary with ethogram
        self._clean_df.loc[:, 'ethograms'].loc[clean_df_row.index[0]][self.selected_trial] = new_ethogram
        # save clean_df to disk
        self._clean_df.to_hdf(self.clean_df_path, key='df')

    def reset_ethogram(self):
        """Will reset currently cleaned ethogram back to the original ethogram."""
        for i, g in enumerate(ETHOGRAM_COLORS.keys()):
            self.plot[g].colors[:] = "black"
            self.plot[g].colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[g]
        # save ethogram to disk after reset
        self.save_ethogram()

