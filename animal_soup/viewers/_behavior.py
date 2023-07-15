import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import HBox, VBox, Select

from fastplotlib import ImageWidget
from warnings import warn
from mesmerize_core.arrays import LazyVideo
from typing import *

from animal_soup.df_utils import get_parent_raw_data_path

from decord import gpu as gpu_context

DECORD_CONTEXT = "cpu"

class BehaviorVizContainer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
    ):
        self._dataframe = dataframe
        self.local_parent_path = get_parent_raw_data_path()

        hide_columns = ["ethograms",
                        "cleaned_ethograms",
                        "notes",
                        "exp_type",
                        "deg_preds"]

        columns = dataframe.columns

        default_widths = {
            'animal_id': 200,
            'session_id': 100,
            'trial_id': 200
        }

        df_show = self._dataframe[[c for c in columns if c not in hide_columns]]

        self.datagrid = DataGrid(
            df_show,
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths)

        self.current_row_ix: int = start_index
        self.trial_selector = None
        self.image_widget = None

        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )
        self._set_trial_selector(self.current_row_ix)

        self.datagrid.observe(self._row_changed, names="selections")

    def _get_selection_row(self) -> Union[int, None]:
        """Returns the index of the current selected row."""
        r1 = self.datagrid.selections[0]["r1"]
        r2 = self.datagrid.selections[0]["r2"]

        if r1 != r2:
            warn("Only single row selection is currently allowed")
            return

        index = self.datagrid.get_visible_data().index[r1]

        return index

    def _row_changed(self, *args):
        """Event handler for when a row in the datagrid is changed."""
        index = self._get_selection_row()

        if index is None or self.current_row_ix == index:
            return

        self.current_row_ix = index
        self._set_trial_selector(index)

    def _set_trial_selector(self, index):
        """Creates trial selector widget for a given session."""
        row = self._dataframe.iloc[index]
        options = [row["trial_id"]]

        if self.trial_selector is None:
            self.trial_selector = Select(options=options)
            self.trial_selector.observe(self._trial_change, "value")
        else:
            self.trial_selector.options = options

        self.selected_trial = self.trial_selector.value

        if self.image_widget is None:
            self._make_image_widget()

    def _make_image_widget(self):
        """
        Instantiates image widget to view behavior videos.
        """
        row = self._dataframe.iloc[self.current_row_ix]
        vid_path = self.local_parent_path.joinpath(row['animal_id'],
                                                   row['session_id'],
                                                   self.selected_trial).with_suffix('.avi')

        if self.image_widget is None:
            if DECORD_CONTEXT == "gpu":
                self.image_widget = ImageWidget(
                                        data=LazyVideo(vid_path, ctx=gpu_context(0)),
                                        grid_plot_kwargs={"size": (700, 300)}
                                        )
            else:
                self.image_widget = ImageWidget(
                                        data=LazyVideo(vid_path),
                                        grid_plot_kwargs={"size": (700, 300)}
                                        )
        # most the time video is rendered upside down, default flip camera
        self.image_widget.gridplot[0, 0].camera.world.scale_y *= -1

    def _trial_change(self, obj):
        """
        Event handler called when a trial is changed in self.trial_selector.
        Updates the behavior imagewidget with new data.
        """
        row = self._dataframe.iloc[self.current_row_ix]
        self.selected_trial = self.trial_selector.value

        session_path = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])
        selected_video = session_path.joinpath(self.selected_trial).with_suffix('.avi')

        if DECORD_CONTEXT == "gpu":
            self.image_widget.set_data([LazyVideo(selected_video, ctx=gpu_context(0))], reset_vmin_vmax=True)
        else:
            self.image_widget.set_data([LazyVideo(selected_video)], reset_vmin_vmax=True)

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                  self.trial_selector])])



