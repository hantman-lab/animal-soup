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
            # check if clean_df exists and user forgot to pass
            if os.path.exists(self.clean_df_path):
                self.clean_df = pd.read_hdf(self.clean_df_path)
            self._clean_df = self._dataframe.copy(deep=True)
            # set ethograms to empty dicts() in order to clean ethograms
            self._clean_df["ethograms"] = [dict() for i in range(len(self._dataframe.index))]
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

        # radio buttons to click which behavior needs to be changed
        self.behavior_buttons = RadioButtons(options=["lift", "handopen", "grab", "sup", "atmouth", "chew"],
                                        layout=Layout(width='auto'))

        # radio buttons to check whether the alpha value should be changed to zero or 1
        self.fill_values = RadioButtons(options=["0", "1"], layout=Layout(width='auto'))

        # button to clean the ethogram based on the current values of the radio buttons
        self.clean_button = Button(value=False,
                                   disabled=False,
                                   icon='broom',
                                   layout=Layout(width='auto'),
                                   tooltip='clean ethogram')

        # save button ethogram to self._clean_df
        self.save_button = Button(value=False,
                                  disabled=False,
                                  icon='save',
                                  layout=Layout(width='auto'),
                                  tooltip='save ethogram')

        # check if key exists, and overwrite, otherwise add
        self.reset_button = Button(value=False,
                                   disabled=False,
                                   icon='history',
                                   layout=Layout(width='auto'),
                                   tooltip='reset ethogram')

        # setting event handlers
        self.clean_button.on_click(self.clean_ethogram)
        self.save_button.on_click(self.save_ethogram)
        self.reset_button.on_click(self.reset_ethogram)

        # group buttons together to make it easier to visualize
        self.radio_box = HBox([self.behavior_buttons, self.fill_values])
        self.clean_options = HBox([self.reset_button, self.clean_button, self.save_button])

    def _make_ethogram_plot(self):
        """
        Instantiates the ethogram plot.
        """
        row = self._dataframe.iloc[self.current_row]

        if self.plot is None:
            self.plot = Plot(size=(500, 100))

        self.ethogram_array = row["ethograms"][self.selected_trial]

        y_bottom = 0
        for i, b in enumerate(ETHOGRAM_COLORS.keys()):
            xs = np.arange(self.ethogram_array.shape[1], dtype=np.float32)
            ys = np.zeros(xs.size, dtype=np.float32)

            lg = self.plot.add_line(
                data=np.column_stack([xs, ys]),
                thickness=10,
                name=b
            )

            lg.colors = 0
            lg.colors[self.ethogram_array[i] == 1] = ETHOGRAM_COLORS[b]

            y_pos = (i * -10) - 1
            lg.position_y = y_pos

        self.ethogram_region_selector = LinearRegionSelector(
            bounds=(0, 50),
            limits=(0, self.ethogram_array.shape[1]),
            axis="x",
            origin=(y_bottom, -25),
            fill_color=(0, 0, 0, 0),
            parent=lg,
            size=(55),
        )

        self.plot.add_graphic(self.ethogram_region_selector)
        self.ethogram_region_selector.selection.add_event_handler(self.ethogram_selection_event_handler)
        self.plot.auto_scale()

    def ethogram_selection_event_handler(self, ev):
        """
        Event handler called for linear region to slide through data using the left or right bound
        of the linear region selector.
        """
        source = ev.pick_info["move_info"].source
        if source is self.ethogram_region_selector.edges[0]:
            self.image_widget.sliders["t"].value = ev.pick_info["selected_indices"][0]
        elif source is self.ethogram_region_selector.edges[1]:
            self.image_widget.sliders["t"].value = ev.pick_info["selected_indices"][-1]

    def clean_ethogram(self, obj):
        """Will correct ethogram based on radio button values and indices selected with linear
        region selector."""
        # get indices of selected region
        selected_ixs = self.plot.selectors[0].get_selected_indices(self.plot.graphics[0])
        # map behavior button value to index
        behavior_ix = self.behavior_buttons.options.index(self.behavior_buttons.value)
        # set indices of selected region to black or ethogram color of selected behavior
        if self.fill_values.value == "0":
            self.plot.graphics[behavior_ix].colors[selected_ixs[0]:selected_ixs[-1]] = "black"
        else:
            self.plot.graphics[behavior_ix].colors[selected_ixs[0]:selected_ixs[-1]] = ETHOGRAM_COLORS[
                self.behavior_buttons.value]

    def save_ethogram(self, obj):
        """Saves an ethogram to the clean dataframe."""
        # create new ethogram based off of indices that are not black
        row = self._dataframe.iloc[self.current_row]
        trial_length = row["ethograms"][self.selected_trial][0].shape[0]
        new_ethogram = np.zeros(shape=(6, trial_length))
        for i, graphic in enumerate(self.plot.graphics):
            non_zero_ixs = np.where(self.plot.graphics[i].colors[:] != np.array([0, 0, 0, 1]))[0]
            new_ethogram[i][non_zero_ixs] = 1
        # check if key already in clean_df
        clean_df_row = self._clean_df[(self._clean_df["animal_id"] == row["animal_id"]) &
                                      (self._clean_df["session_id"] == row["session_id"])]
        # add or update dictionary with ethogram
        self._clean_df.loc[:, 'ethograms'].loc[clean_df_row.index[0]][self.selected_trial] = new_ethogram
        # save clean_df to disk
        self._clean_df.to_hdf(self.clean_df_path, key='df')

    def reset_ethogram(self, obj):
        """Will reset currently cleaned ethogram back to the original ethogram."""
        row = self._dataframe.iloc[self.current_row]
        old_ethogram = row["ethograms"][self.selected_trial]
        for i, graphic in enumerate(self.plot.graphics):
            graphic.colors[:] = "black"
            graphic.colors[old_ethogram[i] == 1] = list(ETHOGRAM_COLORS.values())[i]

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                self.trial_selector]),
            HBox([self.plot.show(),
                 VBox([
                     self.radio_box,
                     self.clean_options
                 ])
                 ])
                  ])
