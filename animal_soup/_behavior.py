import pandas as pd
from ipydatagrid import DataGrid
from ipywidgets import HBox, VBox, Select

from fastplotlib import ImageWidget
from warnings import warn
from mesmerize_core.arrays import LazyVideo
from typing import *

from .batch_utils import get_parent_raw_data_path


class BehaviorVizContainer:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            start_index: int = 0,
    ):
        self._dataframe = dataframe
        self.local_parent_path = get_parent_raw_data_path()

        hide_columns = ["mat_path",
                        "deg_path",
                        "session_vids",
                        "notes"]

        columns = dataframe.columns

        default_widths = {
            'animal_id': 200,
            'session_id': 200
        }

        df_show = self._dataframe[[c for c in columns if c not in hide_columns]]

        self.datagrid = DataGrid(
            df_show,
            selection_mode="cell",
            layout={"height": "250px", "width": "750px"},
            base_row_size=24,
            index_name="index",
            column_widths=default_widths)

        self.current_row: int = start_index
        self.trial_selector = None
        self.image_widget = None

        self.datagrid.select(
            row1=start_index,
            column1=0,
            row2=start_index,
            column2=len(df_show.columns),
            clear_mode="all"
        )
        self._set_trial_selector(self.current_row)

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

        if index is None or self.current_row == index:
            return

        self.current_row = index
        self._set_trial_selector(index)

    def _set_trial_selector(self, index):
        """Creates trial selector widget for a given session."""
        row = self._dataframe.iloc[index]
        options = [item.stem for item in row['session_vids']]

        if self.trial_selector is None:
            self.trial_selector = Select(options=options)
            self.trial_selector.observe(self._trial_change, "value")
        else:
            self.trial_selector.options = options

        self.selected_trial_ix = int(self.trial_selector.value.split('_v')[-1])

        if self.image_widget is None:
            self._make_image_widget()

    def _make_image_widget(self):
        """
        Instantiates image widget to view behavior videos.
        """
        row = self._dataframe.iloc[self.current_row]
        vid_path = self.local_parent_path.joinpath(row['session_vids'][int(self.selected_trial_ix)])

        if self.image_widget is None:
            self.image_widget = ImageWidget(data=LazyVideo(vid_path))


    def _trial_change(self, obj):
        """
        Event handler called when a trial is changed in self.trial_selector.
        Updates the behavior imagewidget and ethogram plot with new data.
        """
        row = self._dataframe.iloc[self.current_row]

        session_path = self.local_parent_path.joinpath(row['animal_id'], row['session_id'])
        selected_video = session_path.joinpath(self.trial_selector.value).with_suffix('.avi')

        self.image_widget._data = [LazyVideo(selected_video)]
        self.image_widget.current_index["t"] = 0
        self.image_widget.sliders["t"].value = 0
        self.image_widget.plot.graphics[0].data = self.image_widget._data[0][0]

    def show(self):
        """
        Shows the widget.
        """
        return VBox([
            self.datagrid,
            HBox([self.image_widget.show(),
                  self.trial_selector])])



